#include "risk.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace QuantRisk {

RiskCalculator::RiskCalculator(unsigned int seed)
    : rng_(seed), normal_dist_(0.0, 1.0) {
}

double RiskCalculator::calculateVaR(const Eigen::VectorXd& returns,
                                   const Eigen::MatrixXd& cov_matrix,
                                   const Eigen::VectorXd& weights,
                                   double confidence_level,
                                   int num_simulations,
                                   int time_horizon) {
    
    if (returns.size() != cov_matrix.rows() || cov_matrix.rows() != cov_matrix.cols()) {
        throw std::invalid_argument("Dimension mismatch in returns and covariance matrix");
    }
    
    if (weights.size() != returns.size()) {
        throw std::invalid_argument("Weights dimension doesn't match returns");
    }
    
    // Generate Monte Carlo simulations
    std::vector<double> simulated_returns = generateSimulations(
        returns, cov_matrix, weights, num_simulations, time_horizon);
    
    // Sort returns in ascending order (losses are negative)
    std::sort(simulated_returns.begin(), simulated_returns.end());
    
    // Find VaR at specified confidence level
    int var_index = static_cast<int>((1.0 - confidence_level) * num_simulations);
    var_index = std::max(0, std::min(var_index, num_simulations - 1));
    
    // Return VaR as positive value (loss)
    return -simulated_returns[var_index];
}

double RiskCalculator::calculateES(const Eigen::VectorXd& returns,
                                  const Eigen::MatrixXd& cov_matrix,
                                  const Eigen::VectorXd& weights,
                                  double confidence_level,
                                  int num_simulations,
                                  int time_horizon) {
    
    // Generate Monte Carlo simulations
    std::vector<double> simulated_returns = generateSimulations(
        returns, cov_matrix, weights, num_simulations, time_horizon);
    
    // Sort returns in ascending order
    std::sort(simulated_returns.begin(), simulated_returns.end());
    
    // Find the tail beyond VaR
    int var_index = static_cast<int>((1.0 - confidence_level) * num_simulations);
    var_index = std::max(0, std::min(var_index, num_simulations - 1));
    
    // Calculate Expected Shortfall as average of tail losses
    double sum_tail_losses = 0.0;
    int count = 0;
    
    for (int i = 0; i <= var_index; ++i) {
        sum_tail_losses += simulated_returns[i];
        count++;
    }
    
    if (count == 0) {
        return 0.0;
    }
    
    // Return ES as positive value (expected loss beyond VaR)
    return -sum_tail_losses / count;
}

double RiskCalculator::calculateVolatility(const Eigen::MatrixXd& cov_matrix,
                                          const Eigen::VectorXd& weights) {
    
    if (cov_matrix.rows() != cov_matrix.cols() || weights.size() != cov_matrix.rows()) {
        throw std::invalid_argument("Dimension mismatch in covariance matrix and weights");
    }
    
    // Portfolio variance: w^T * Sigma * w
    double portfolio_variance = weights.transpose() * cov_matrix * weights;
    
    // Return volatility (standard deviation)
    return std::sqrt(std::max(0.0, portfolio_variance));
}

std::vector<double> RiskCalculator::generateSimulations(const Eigen::VectorXd& returns,
                                                       const Eigen::MatrixXd& cov_matrix,
                                                       const Eigen::VectorXd& weights,
                                                       int num_simulations,
                                                       int time_horizon) {
    
    int num_assets = returns.size();
    
    // Perform Cholesky decomposition of covariance matrix
    Eigen::LLT<Eigen::MatrixXd> chol_decomp(cov_matrix);
    if (chol_decomp.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky decomposition failed - covariance matrix not positive definite");
    }
    
    Eigen::MatrixXd L = chol_decomp.matrixL();
    
    // Pre-compute portfolio expected return and time scaling
    double portfolio_return = weights.dot(returns);
    double time_scale = std::sqrt(static_cast<double>(time_horizon));
    
    std::vector<double> simulated_returns;
    simulated_returns.reserve(num_simulations);
    
    // Generate simulations
    for (int i = 0; i < num_simulations; ++i) {
        // Generate independent normal random variables
        Eigen::VectorXd random_normals(num_assets);
        for (int j = 0; j < num_assets; ++j) {
            random_normals(j) = normal_dist_(rng_);
        }
        
        // Transform to correlated random variables using Cholesky decomposition
        Eigen::VectorXd correlated_shocks = L * random_normals;
        
        // Scale by time horizon
        correlated_shocks *= time_scale;
        
        // Calculate portfolio return for this simulation
        // Portfolio return = expected return * time + portfolio shock
        double sim_return = portfolio_return * time_horizon + weights.dot(correlated_shocks);
        
        simulated_returns.push_back(sim_return);
    }
    
    return simulated_returns;
}

double RiskCalculator::calculateMaxDrawdown(const std::vector<double>& returns) {
    if (returns.empty()) {
        return 0.0;
    }
    
    double max_drawdown = 0.0;
    double peak = returns[0];
    double cumulative_return = returns[0];
    
    for (size_t i = 1; i < returns.size(); ++i) {
        cumulative_return += returns[i];
        
        if (cumulative_return > peak) {
            peak = cumulative_return;
        }
        
        double drawdown = (peak - cumulative_return) / (1.0 + peak);
        max_drawdown = std::max(max_drawdown, drawdown);
    }
    
    return max_drawdown;
}

Eigen::VectorXd RiskCalculator::calculateRiskContributions(const Eigen::MatrixXd& cov_matrix,
                                                          const Eigen::VectorXd& weights) {
    
    if (cov_matrix.rows() != cov_matrix.cols() || weights.size() != cov_matrix.rows()) {
        throw std::invalid_argument("Dimension mismatch in covariance matrix and weights");
    }
    
    // Calculate portfolio volatility
    double portfolio_vol = calculateVolatility(cov_matrix, weights);
    
    if (portfolio_vol == 0.0) {
        return Eigen::VectorXd::Zero(weights.size());
    }
    
    // Risk contributions: (weight_i * (Sigma * weights)_i) / portfolio_volatility
    Eigen::VectorXd marginal_contrib = cov_matrix * weights;
    Eigen::VectorXd risk_contrib = weights.cwiseProduct(marginal_contrib) / portfolio_vol;
    
    return risk_contrib;
}

} // namespace QuantRisk
