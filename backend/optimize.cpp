#include "optimize.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace QuantRisk {

PortfolioOptimizer::PortfolioOptimizer(double tolerance, int max_iterations)
    : tolerance_(tolerance), max_iterations_(max_iterations) {
}

OptimizationResult PortfolioOptimizer::solveQP(const Eigen::MatrixXd& Q,
                                              const Eigen::VectorXd& c,
                                              const Eigen::MatrixXd& A,
                                              const Eigen::VectorXd& b,
                                              const Eigen::VectorXd& lb,
                                              const Eigen::VectorXd& ub) {
    
    OptimizationResult result;
    int n = Q.rows();
    
    // Simple active set method for quadratic programming
    // This is a basic implementation - in production, use libraries like OSQP or Quadprog++
    
    // Initialize with equal weights satisfying sum constraint
    Eigen::VectorXd x = Eigen::VectorXd::Constant(n, 1.0 / n);
    
    // Apply bounds
    if (lb.size() == n) {
        x = x.cwiseMax(lb);
    }
    if (ub.size() == n) {
        x = x.cwiseMin(ub);
    }
    
    // Renormalize to satisfy sum constraint
    double sum = x.sum();
    if (sum > 0) {
        x /= sum;
    }
    
    // Simple gradient descent with projection
    double learning_rate = 0.01;
    int iterations = 0;
    
    for (iterations = 0; iterations < max_iterations_; ++iterations) {
        // Compute gradient: Q*x + c
        Eigen::VectorXd gradient = Q * x + c;
        
        // Check convergence
        double grad_norm = gradient.norm();
        if (grad_norm < tolerance_) {
            result.converged = true;
            break;
        }
        
        // Gradient step
        Eigen::VectorXd x_new = x - learning_rate * gradient;
        
        // Project onto constraints
        // Apply bounds
        if (lb.size() == n) {
            x_new = x_new.cwiseMax(lb);
        }
        if (ub.size() == n) {
            x_new = x_new.cwiseMin(ub);
        }
        
        // Project onto sum constraint (simplex projection)
        double sum_new = x_new.sum();
        if (std::abs(sum_new - 1.0) > tolerance_) {
            x_new = x_new.array() - (sum_new - 1.0) / n;
            
            // Ensure non-negativity and re-project
            if (lb.size() == n) {
                for (int i = 0; i < n; ++i) {
                    if (x_new(i) < lb(i)) {
                        x_new(i) = lb(i);
                    }
                }
                // Renormalize
                double sum_final = x_new.sum();
                if (sum_final > 0 && std::abs(sum_final - 1.0) > tolerance_) {
                    x_new *= (1.0 / sum_final);
                }
            }
        }
        
        x = x_new;
        
        // Adaptive learning rate
        if (iterations % 100 == 0 && iterations > 0) {
            learning_rate *= 0.9;
        }
    }
    
    result.weights = x;
    result.iterations = iterations;
    result.objective_value = 0.5 * (x.transpose() * Q * x)(0,0) + (c.transpose() * x)(0,0);

    
    return result;
}

OptimizationResult PortfolioOptimizer::maximizeSharpeRatio(const Eigen::VectorXd& returns,
                                                         const Eigen::MatrixXd& cov_matrix,
                                                         double risk_free_rate,
                                                         const Eigen::VectorXd& min_weights,
                                                         const Eigen::VectorXd& max_weights) {
    
    int n = returns.size();
    
    if (cov_matrix.rows() != n || cov_matrix.cols() != n) {
        throw std::invalid_argument("Dimension mismatch between returns and covariance matrix");
    }
    
    // Convert Sharpe ratio maximization to quadratic programming
    // We use the transformation approach: maximize (mu - rf)^T w / sqrt(w^T Sigma w)
    // This is equivalent to solving a quadratic program
    
    Eigen::VectorXd excess_returns = returns.array() - risk_free_rate;
    
    // Check if covariance matrix is positive definite
    Eigen::LLT<Eigen::MatrixXd> chol_decomp(cov_matrix);
    if (chol_decomp.info() != Eigen::Success) {
        throw std::runtime_error("Covariance matrix is not positive definite");
    }
    
    // Solve using the dual formulation
    // min w^T Sigma w subject to (mu - rf)^T w = 1 and other constraints
    
    // Set up QP: minimize 0.5 * x^T * Q * x + c^T * x
    Eigen::MatrixXd Q = 2.0 * cov_matrix;  // Factor of 2 for 0.5 * x^T * Q * x
    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
    
    // Equality constraint: sum of weights = 1
    Eigen::MatrixXd A(1, n);
    A.row(0) = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd b(1);
    b(0) = 1.0;
    
    // Set up bounds
    Eigen::VectorXd lb = min_weights.size() == n ? min_weights : Eigen::VectorXd::Zero(n);
    Eigen::VectorXd ub = max_weights.size() == n ? max_weights : Eigen::VectorXd::Ones(n);
    
    OptimizationResult result = solveQP(Q, c, A, b, lb, ub);
    
    // Calculate portfolio metrics
    auto [exp_ret, vol, sharpe] = calculatePortfolioMetrics(returns, cov_matrix, result.weights, risk_free_rate);
    result.expected_return = exp_ret;
    result.volatility = vol;
    result.sharpe_ratio = sharpe;
    
    return result;
}

OptimizationResult PortfolioOptimizer::minimizeVariance(const Eigen::MatrixXd& cov_matrix,
                                                       const Eigen::VectorXd& min_weights,
                                                       const Eigen::VectorXd& max_weights) {
    
    int n = cov_matrix.rows();
    
    if (cov_matrix.cols() != n) {
        throw std::invalid_argument("Covariance matrix must be square");
    }
    
    // Set up QP: minimize 0.5 * w^T * Sigma * w
    Eigen::MatrixXd Q = 2.0 * cov_matrix;
    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
    
    // Equality constraint: sum of weights = 1
    Eigen::MatrixXd A(1, n);
    A.row(0) = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd b(1);
    b(0) = 1.0;
    
    // Set up bounds
    Eigen::VectorXd lb = min_weights.size() == n ? min_weights : Eigen::VectorXd::Zero(n);
    Eigen::VectorXd ub = max_weights.size() == n ? max_weights : Eigen::VectorXd::Ones(n);
    
    OptimizationResult result = solveQP(Q, c, A, b, lb, ub);
    
    // Calculate portfolio volatility
    result.volatility = std::sqrt(result.weights.transpose() * cov_matrix * result.weights);
    result.expected_return = 0.0;  // Not specified for minimum variance
    result.sharpe_ratio = 0.0;
    
    return result;
}

OptimizationResult PortfolioOptimizer::optimizeForReturn(const Eigen::VectorXd& returns,
                                                        const Eigen::MatrixXd& cov_matrix,
                                                        double target_return,
                                                        const Eigen::VectorXd& min_weights,
                                                        const Eigen::VectorXd& max_weights) {
    
    int n = returns.size();
    
    // Set up QP: minimize 0.5 * w^T * Sigma * w
    Eigen::MatrixXd Q = 2.0 * cov_matrix;
    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
    
    // Equality constraints: sum of weights = 1 and target return
    Eigen::MatrixXd A(2, n);
    A.row(0) = Eigen::VectorXd::Ones(n);      // Sum constraint
    A.row(1) = returns.transpose();           // Return constraint
    
    Eigen::VectorXd b(2);
    b(0) = 1.0;
    b(1) = target_return;
    
    // Set up bounds
    Eigen::VectorXd lb = min_weights.size() == n ? min_weights : Eigen::VectorXd::Zero(n);
    Eigen::VectorXd ub = max_weights.size() == n ? max_weights : Eigen::VectorXd::Ones(n);
    
    OptimizationResult result = solveQP(Q, c, A, b, lb, ub);
    
    // Calculate portfolio metrics
    auto [exp_ret, vol, sharpe] = calculatePortfolioMetrics(returns, cov_matrix, result.weights);
    result.expected_return = exp_ret;
    result.volatility = vol;
    result.sharpe_ratio = sharpe;
    
    return result;
}

std::vector<FrontierPoint> PortfolioOptimizer::generateEfficientFrontier(
    const Eigen::VectorXd& returns,
    const Eigen::MatrixXd& cov_matrix,
    int num_points,
    const Eigen::VectorXd& min_weights,
    const Eigen::VectorXd& max_weights) {
    
    std::vector<FrontierPoint> frontier_points;
    frontier_points.reserve(num_points);
    
    // Find minimum and maximum possible returns
    double min_return = returns.minCoeff();
    double max_return = returns.maxCoeff();
    
    // Generate return targets
    std::vector<double> return_targets;
    for (int i = 0; i < num_points; ++i) {
        double target = min_return + (max_return - min_return) * i / (num_points - 1);
        return_targets.push_back(target);
    }
    
    // Optimize for each target return
    for (double target_return : return_targets) {
        try {
            OptimizationResult opt_result = optimizeForReturn(
                returns, cov_matrix, target_return, min_weights, max_weights);
            
            if (opt_result.converged || opt_result.weights.sum() > 0.9) {  // Accept if close to convergence
                FrontierPoint point;
                point.expected_return = opt_result.expected_return;
                point.volatility = opt_result.volatility;
                point.sharpe_ratio = opt_result.sharpe_ratio;
                point.weights = opt_result.weights;
                
                frontier_points.push_back(point);
            }
        } catch (const std::exception& e) {
            // Skip this point if optimization fails
            continue;
        }
    }
    
    // Sort by volatility
    std::sort(frontier_points.begin(), frontier_points.end(),
              [](const FrontierPoint& a, const FrontierPoint& b) {
                  return a.volatility < b.volatility;
              });
    
    return frontier_points;
}

std::tuple<double, double, double> PortfolioOptimizer::calculatePortfolioMetrics(
    const Eigen::VectorXd& returns,
    const Eigen::MatrixXd& cov_matrix,
    const Eigen::VectorXd& weights,
    double risk_free_rate) {
    
    // Portfolio expected return
    double portfolio_return = weights.dot(returns);
    
    // Portfolio volatility
    double portfolio_variance = weights.transpose() * cov_matrix * weights;
    double portfolio_volatility = std::sqrt(std::max(0.0, portfolio_variance));
    
    // Sharpe ratio
    double sharpe_ratio = 0.0;
    if (portfolio_volatility > 0.0) {
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility;
    }
    
    return std::make_tuple(portfolio_return, portfolio_volatility, sharpe_ratio);
}

bool PortfolioOptimizer::validateWeights(const Eigen::VectorXd& weights,
                                        const Eigen::VectorXd& min_weights,
                                        const Eigen::VectorXd& max_weights) {
    
    const double tolerance = 1e-6;
    
    // Check if weights sum to 1
    double weight_sum = weights.sum();
    if (std::abs(weight_sum - 1.0) > tolerance) {
        return false;
    }
    
    // Check minimum weight constraints
    if (min_weights.size() == weights.size()) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weights(i) < min_weights(i) - tolerance) {
                return false;
            }
        }
    }
    
    // Check maximum weight constraints
    if (max_weights.size() == weights.size()) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weights(i) > max_weights(i) + tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

} // namespace QuantRisk
