#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <vector>
#include <Eigen/Dense>
#include <memory>

namespace QuantRisk {

/**
 * @brief Portfolio optimization result structure
 */
struct OptimizationResult {
    Eigen::VectorXd weights;      // Optimal portfolio weights
    double expected_return;       // Portfolio expected return
    double volatility;           // Portfolio volatility
    double sharpe_ratio;         // Sharpe ratio
    bool converged;              // Convergence flag
    int iterations;              // Number of iterations used
    double objective_value;      // Final objective function value
    
    OptimizationResult() : converged(false), iterations(0), objective_value(0.0) {}
};

/**
 * @brief Efficient frontier point structure
 */
struct FrontierPoint {
    double expected_return;
    double volatility;
    double sharpe_ratio;
    Eigen::VectorXd weights;
    
    FrontierPoint() : expected_return(0.0), volatility(0.0), sharpe_ratio(0.0) {}
};

/**
 * @brief Portfolio optimizer class implementing mean-variance optimization
 * 
 * This class provides high-performance portfolio optimization using:
 * - Markowitz mean-variance optimization
 * - Efficient frontier computation
 * - Maximum Sharpe ratio optimization
 * - Minimum variance optimization
 */
class PortfolioOptimizer {
private:
    double tolerance_;
    int max_iterations_;
    
    /**
     * @brief Internal function to solve quadratic programming problem
     * @param Q Quadratic term matrix (covariance)
     * @param c Linear term vector
     * @param A Equality constraint matrix
     * @param b Equality constraint vector
     * @param lb Lower bounds
     * @param ub Upper bounds
     * @return Optimization result
     */
    OptimizationResult solveQP(const Eigen::MatrixXd& Q,
                              const Eigen::VectorXd& c,
                              const Eigen::MatrixXd& A,
                              const Eigen::VectorXd& b,
                              const Eigen::VectorXd& lb,
                              const Eigen::VectorXd& ub);
    
public:
    /**
     * @brief Constructor with optimization parameters
     * @param tolerance Convergence tolerance
     * @param max_iterations Maximum number of iterations
     */
    explicit PortfolioOptimizer(double tolerance = 1e-8, int max_iterations = 1000);
    
    /**
     * @brief Find maximum Sharpe ratio portfolio
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param risk_free_rate Risk-free rate
     * @param min_weights Minimum weight constraints
     * @param max_weights Maximum weight constraints
     * @return Optimization result with maximum Sharpe ratio portfolio
     */
    OptimizationResult maximizeSharpeRatio(const Eigen::VectorXd& returns,
                                         const Eigen::MatrixXd& cov_matrix,
                                         double risk_free_rate = 0.02,
                                         const Eigen::VectorXd& min_weights = Eigen::VectorXd(),
                                         const Eigen::VectorXd& max_weights = Eigen::VectorXd());
    
    /**
     * @brief Find minimum variance portfolio
     * @param cov_matrix Covariance matrix
     * @param min_weights Minimum weight constraints
     * @param max_weights Maximum weight constraints
     * @return Optimization result with minimum variance portfolio
     */
    OptimizationResult minimizeVariance(const Eigen::MatrixXd& cov_matrix,
                                       const Eigen::VectorXd& min_weights = Eigen::VectorXd(),
                                       const Eigen::VectorXd& max_weights = Eigen::VectorXd());
    
    /**
     * @brief Optimize portfolio for target return
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param target_return Target portfolio return
     * @param min_weights Minimum weight constraints
     * @param max_weights Maximum weight constraints
     * @return Optimization result for target return
     */
    OptimizationResult optimizeForReturn(const Eigen::VectorXd& returns,
                                        const Eigen::MatrixXd& cov_matrix,
                                        double target_return,
                                        const Eigen::VectorXd& min_weights = Eigen::VectorXd(),
                                        const Eigen::VectorXd& max_weights = Eigen::VectorXd());
    
    /**
     * @brief Generate efficient frontier points
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param num_points Number of frontier points to generate
     * @param min_weights Minimum weight constraints
     * @param max_weights Maximum weight constraints
     * @return Vector of efficient frontier points
     */
    std::vector<FrontierPoint> generateEfficientFrontier(const Eigen::VectorXd& returns,
                                                        const Eigen::MatrixXd& cov_matrix,
                                                        int num_points = 50,
                                                        const Eigen::VectorXd& min_weights = Eigen::VectorXd(),
                                                        const Eigen::VectorXd& max_weights = Eigen::VectorXd());
    
    /**
     * @brief Calculate portfolio performance metrics
     * @param returns Expected returns vector
     * @param cov_matrix Covariance matrix
     * @param weights Portfolio weights
     * @param risk_free_rate Risk-free rate
     * @return Tuple of (expected_return, volatility, sharpe_ratio)
     */
    std::tuple<double, double, double> calculatePortfolioMetrics(const Eigen::VectorXd& returns,
                                                               const Eigen::MatrixXd& cov_matrix,
                                                               const Eigen::VectorXd& weights,
                                                               double risk_free_rate = 0.02);
    
    /**
     * @brief Validate portfolio weights (sum to 1, within bounds)
     * @param weights Portfolio weights
     * @param min_weights Minimum weight constraints
     * @param max_weights Maximum weight constraints
     * @return True if weights are valid
     */
    bool validateWeights(const Eigen::VectorXd& weights,
                        const Eigen::VectorXd& min_weights = Eigen::VectorXd(),
                        const Eigen::VectorXd& max_weights = Eigen::VectorXd());
};

} // namespace QuantRisk

#endif // OPTIMIZE_H
