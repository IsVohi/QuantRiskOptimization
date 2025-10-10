#ifndef RISK_H
#define RISK_H

#include <vector>
#include <Eigen/Dense>
#include <random>

namespace QuantRisk {

/**
 * @brief Risk metrics and Monte Carlo simulation class
 * 
 * This class implements high-performance risk calculations including:
 * - Monte Carlo Value at Risk (VaR)
 * - Expected Shortfall (ES/CVaR)
 * - Portfolio volatility and drawdown metrics
 */
class RiskCalculator {
private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    
public:
    /**
     * @brief Constructor with random seed
     * @param seed Random seed for reproducible results
     */
    explicit RiskCalculator(unsigned int seed = 12345);
    
    /**
     * @brief Calculate portfolio Value at Risk using Monte Carlo simulation
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param weights Portfolio weights
     * @param confidence_level Confidence level (e.g., 0.95 for 95% VaR)
     * @param num_simulations Number of Monte Carlo simulations
     * @param time_horizon Time horizon in days
     * @return VaR value (positive indicates loss)
     */
    double calculateVaR(const Eigen::VectorXd& returns,
                       const Eigen::MatrixXd& cov_matrix,
                       const Eigen::VectorXd& weights,
                       double confidence_level = 0.95,
                       int num_simulations = 10000,
                       int time_horizon = 1);
    
    /**
     * @brief Calculate Expected Shortfall (Conditional VaR)
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param weights Portfolio weights
     * @param confidence_level Confidence level
     * @param num_simulations Number of Monte Carlo simulations
     * @param time_horizon Time horizon in days
     * @return Expected Shortfall value
     */
    double calculateES(const Eigen::VectorXd& returns,
                      const Eigen::MatrixXd& cov_matrix,
                      const Eigen::VectorXd& weights,
                      double confidence_level = 0.95,
                      int num_simulations = 10000,
                      int time_horizon = 1);
    
    /**
     * @brief Calculate portfolio volatility (annualized)
     * @param cov_matrix Covariance matrix (annualized)
     * @param weights Portfolio weights
     * @return Portfolio volatility
     */
    double calculateVolatility(const Eigen::MatrixXd& cov_matrix,
                              const Eigen::VectorXd& weights);
    
    /**
     * @brief Generate Monte Carlo simulation paths
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param weights Portfolio weights
     * @param num_simulations Number of simulations
     * @param time_horizon Time horizon
     * @return Vector of simulated portfolio returns
     */
    std::vector<double> generateSimulations(const Eigen::VectorXd& returns,
                                          const Eigen::MatrixXd& cov_matrix,
                                          const Eigen::VectorXd& weights,
                                          int num_simulations,
                                          int time_horizon);
    
    /**
     * @brief Calculate maximum drawdown from return series
     * @param returns Vector of returns
     * @return Maximum drawdown value
     */
    double calculateMaxDrawdown(const std::vector<double>& returns);
    
    /**
     * @brief Decompose portfolio risk by asset contribution
     * @param cov_matrix Covariance matrix
     * @param weights Portfolio weights
     * @return Vector of risk contributions by asset
     */
    Eigen::VectorXd calculateRiskContributions(const Eigen::MatrixXd& cov_matrix,
                                              const Eigen::VectorXd& weights);
};

} // namespace QuantRisk

#endif // RISK_H
