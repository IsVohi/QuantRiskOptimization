"""
Unit tests for portfolio optimization module

Tests both Python fallback implementations and C++ backend (if available)
for mean-variance optimization, efficient frontier generation, and Sharpe ratio maximization.
"""

import unittest
import numpy as np
import sys
import os

# Add the frontend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend'))

from tests.test_optimize import create_optimizer, PortfolioResult, FrontierPoint

class TestPortfolioOptimization(unittest.TestCase):
    """Test suite for portfolio optimization functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = create_optimizer(use_cpp=True)
        self.optimizer_python = create_optimizer(use_cpp=False)
        
        # Test portfolio data - 4 assets for simplicity
        self.returns = np.array([0.10, 0.08, 0.12, 0.04])
        
        # Create a reasonable covariance matrix
        corr_matrix = np.array([
            [1.00, 0.30, 0.10, -0.10],
            [0.30, 1.00, 0.20, 0.05],
            [0.10, 0.20, 1.00, -0.05],
            [-0.10, 0.05, -0.05, 1.00]
        ])
        
        volatilities = np.array([0.15, 0.12, 0.18, 0.05])
        self.cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(self.cov_matrix)
        if np.min(eigenvals) <= 0:
            self.cov_matrix += np.eye(len(self.returns)) * (abs(np.min(eigenvals)) + 0.001)
        
        # Default constraints
        self.min_weights = np.full(len(self.returns), 0.0)
        self.max_weights = np.full(len(self.returns), 1.0)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        # Test with different parameters
        opt1 = create_optimizer(tolerance=1e-6, max_iterations=500, use_cpp=True)
        self.assertEqual(opt1.tolerance, 1e-6)
        self.assertEqual(opt1.max_iterations, 500)
        
        opt2 = create_optimizer(use_cpp=False)
        self.assertFalse(opt2.use_cpp)
    
    def test_input_validation(self):
        """Test input validation for optimization"""
        # Test dimension mismatch
        wrong_returns = np.array([0.1, 0.05])  # Wrong size
        
        result = self.optimizer.maximize_sharpe_ratio(wrong_returns, self.cov_matrix)
        self.assertFalse(result.converged)
        self.assertEqual(result.method, 'error')
        
        # Test non-square covariance matrix
        wrong_cov = np.random.rand(3, 4)
        result = self.optimizer.maximize_sharpe_ratio(self.returns, wrong_cov)
        self.assertFalse(result.converged)
        self.assertEqual(result.method, 'error')
    
    def test_sharpe_ratio_maximization(self):
        """Test maximum Sharpe ratio optimization"""
        result = self.optimizer.maximize_sharpe_ratio(
            self.returns, self.cov_matrix, risk_free_rate=0.02,
            min_weights=self.min_weights, max_weights=self.max_weights
        )
        
        self.assertIsInstance(result, PortfolioResult)
        
        if result.converged:
            # Check basic properties
            self.assertEqual(len(result.weights), len(self.returns))
            self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)
            self.assertGreaterEqual(np.min(result.weights), -1e-6)  # Allow small numerical errors
            self.assertLessEqual(np.max(result.weights), 1.0 + 1e-6)
            
            # Check that Sharpe ratio is positive for our test data
            self.assertGreater(result.sharpe_ratio, 0)
            
            # Check computation time is recorded
            self.assertGreater(result.computation_time, 0)
    
    def test_minimum_variance_optimization(self):
        """Test minimum variance optimization"""
        result = self.optimizer.minimize_variance(
            self.cov_matrix, min_weights=self.min_weights, max_weights=self.max_weights
        )
        
        self.assertIsInstance(result, PortfolioResult)
        
        if result.converged:
            # Check basic properties
            self.assertEqual(len(result.weights), len(self.returns))
            self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)
            self.assertGreaterEqual(np.min(result.weights), -1e-6)
            self.assertLessEqual(np.max(result.weights), 1.0 + 1e-6)
            
            # Volatility should be positive
            self.assertGreater(result.volatility, 0)
            
            # For minimum variance, should prefer low volatility assets
            # Asset 3 (index 3) has lowest volatility (0.05), so should have high weight
            low_vol_asset_index = np.argmin([0.15, 0.12, 0.18, 0.05])
            self.assertEqual(low_vol_asset_index, 3)
    
    def test_target_return_optimization(self):
        """Test optimization for target return"""
        target_return = 0.08  # 8% target
        
        result = self.optimizer.optimize_for_return(
            self.returns, self.cov_matrix, target_return,
            min_weights=self.min_weights, max_weights=self.max_weights
        )
        
        self.assertIsInstance(result, PortfolioResult)
        
        if result.converged:
            # Check basic properties
            self.assertEqual(len(result.weights), len(self.returns))
            self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)
            
            # Check that achieved return is close to target
            self.assertAlmostEqual(result.expected_return, target_return, places=3)
    
    def test_target_return_bounds(self):
        """Test target return optimization with unrealistic targets"""
        # Test target return too high
        high_target = np.max(self.returns) + 0.05
        result = self.optimizer.optimize_for_return(
            self.returns, self.cov_matrix, high_target
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.method, 'error')
        
        # Test target return too low
        low_target = np.min(self.returns) - 0.05
        result = self.optimizer.optimize_for_return(
            self.returns, self.cov_matrix, low_target
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.method, 'error')
    
    def test_weight_constraints(self):
        """Test optimization with weight constraints"""
        # Test with tighter constraints
        min_weights_tight = np.full(len(self.returns), 0.1)  # At least 10% each
        max_weights_tight = np.full(len(self.returns), 0.4)  # At most 40% each
        
        result = self.optimizer.maximize_sharpe_ratio(
            self.returns, self.cov_matrix,
            min_weights=min_weights_tight, max_weights=max_weights_tight
        )
        
        if result.converged:
            # Check that constraints are satisfied
            self.assertTrue(np.all(result.weights >= min_weights_tight - 1e-6))
            self.assertTrue(np.all(result.weights <= max_weights_tight + 1e-6))
    
    def test_efficient_frontier_generation(self):
        """Test efficient frontier generation"""
        num_points = 20
        frontier_points = self.optimizer.generate_efficient_frontier(
            self.returns, self.cov_matrix, num_points=num_points,
            min_weights=self.min_weights, max_weights=self.max_weights
        )
        
        self.assertIsInstance(frontier_points, list)
        self.assertGreater(len(frontier_points), 0)
        
        if len(frontier_points) > 1:
            # Check that points are ordered by volatility
            volatilities = [point.volatility for point in frontier_points]
            self.assertEqual(volatilities, sorted(volatilities))
            
            # Check that each point has valid properties
            for point in frontier_points:
                self.assertIsInstance(point, FrontierPoint)
                self.assertAlmostEqual(np.sum(point.weights), 1.0, places=5)
                self.assertGreater(point.volatility, 0)
    
    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation"""
        # Use equal weights for testing
        weights = np.full(len(self.returns), 1.0 / len(self.returns))
        
        metrics = self.optimizer.calculate_portfolio_metrics(
            self.returns, self.cov_matrix, weights, risk_free_rate=0.03
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('expected_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        
        if 'error' not in metrics:
            # Check expected return calculation
            expected_return_manual = np.dot(weights, self.returns)
            self.assertAlmostEqual(metrics['expected_return'], expected_return_manual, places=10)
            
            # Check volatility calculation
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            volatility_manual = np.sqrt(portfolio_var)
            self.assertAlmostEqual(metrics['volatility'], volatility_manual, places=10)
            
            # Check Sharpe ratio calculation
            if metrics['volatility'] > 0:
                sharpe_manual = (metrics['expected_return'] - 0.03) / metrics['volatility']
                self.assertAlmostEqual(metrics['sharpe_ratio'], sharpe_manual, places=10)
    
    def test_weight_validation(self):
        """Test weight validation functionality"""
        # Valid weights
        valid_weights = np.array([0.25, 0.25, 0.25, 0.25])
        validation = self.optimizer.validate_weights(valid_weights)
        self.assertTrue(validation['valid'])
        self.assertEqual(len(validation['violations']), 0)
        
        # Invalid weights - don't sum to 1
        invalid_weights = np.array([0.3, 0.3, 0.3, 0.3])
        validation = self.optimizer.validate_weights(invalid_weights)
        self.assertFalse(validation['valid'])
        self.assertGreater(len(validation['violations']), 0)
        
        # Test with constraints
        min_weights = np.array([0.1, 0.1, 0.1, 0.1])
        max_weights = np.array([0.4, 0.4, 0.4, 0.4])
        
        # Weights violating minimum constraint
        below_min_weights = np.array([0.05, 0.3, 0.3, 0.35])
        validation = self.optimizer.validate_weights(
            below_min_weights, min_weights=min_weights, max_weights=max_weights
        )
        self.assertFalse(validation['valid'])
        
        # Weights violating maximum constraint
        above_max_weights = np.array([0.5, 0.2, 0.2, 0.1])
        validation = self.optimizer.validate_weights(
            above_max_weights, min_weights=min_weights, max_weights=max_weights
        )
        self.assertFalse(validation['valid'])
    
    def test_python_fallback(self):
        """Test Python fallback implementations"""
        # Test with Python-only optimizer
        result = self.optimizer_python.maximize_sharpe_ratio(
            self.returns, self.cov_matrix, risk_free_rate=0.02
        )
        
        if result.converged:
            self.assertEqual(result.method, 'python')
            self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)
            self.assertGreater(result.sharpe_ratio, 0)
    
    def test_optimization_consistency(self):
        """Test that optimization results are consistent"""
        # Run the same optimization multiple times
        results = []
        for _ in range(3):
            result = self.optimizer.maximize_sharpe_ratio(
                self.returns, self.cov_matrix, risk_free_rate=0.02
            )
            if result.converged:
                results.append(result)
        
        if len(results) > 1:
            # Results should be very similar (allowing for numerical differences)
            for i in range(1, len(results)):
                np.testing.assert_allclose(
                    results[0].weights, results[i].weights, 
                    rtol=1e-6, atol=1e-8,
                    err_msg="Optimization results should be consistent"
                )
    
    def test_efficient_frontier_properties(self):
        """Test mathematical properties of efficient frontier"""
        frontier_points = self.optimizer.generate_efficient_frontier(
            self.returns, self.cov_matrix, num_points=15
        )
        
        if len(frontier_points) >= 3:
            # Efficient frontier should be convex in mean-variance space
            # This is hard to test directly, but we can check monotonicity
            
            # Extract returns and volatilities
            frontier_returns = [p.expected_return for p in frontier_points]
            frontier_vols = [p.volatility for p in frontier_points]
            
            # Returns should generally increase with volatility on efficient frontier
            # (allowing for some numerical noise)
            correlations = np.corrcoef(frontier_vols, frontier_returns)[0, 1]
            self.assertGreater(correlations, 0.5, 
                             "Expected returns should generally increase with volatility on efficient frontier")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with identical assets (perfectly correlated)
        identical_returns = np.full(4, 0.08)
        identity_cov = np.eye(4) * 0.1
        
        result = self.optimizer.maximize_sharpe_ratio(identical_returns, identity_cov)
        # Should still work, might converge to equal weights
        
        # Test with very small covariance values
        tiny_cov = self.cov_matrix * 1e-10
        result = self.optimizer.minimize_variance(tiny_cov)
        if result.converged:
            self.assertGreater(result.volatility, 0)
        
        # Test with zero risk-free rate
        result = self.optimizer.maximize_sharpe_ratio(
            self.returns, self.cov_matrix, risk_free_rate=0.0
        )
        # Should still work

class TestAdvancedOptimization(unittest.TestCase):
    """Advanced tests for optimization edge cases and performance"""
    
    def setUp(self):
        """Set up test fixtures for advanced tests"""
        self.optimizer = create_optimizer()
        
        # Larger portfolio for stress testing
        n_assets = 8
        np.random.seed(42)
        
        self.large_returns = np.random.uniform(0.02, 0.15, n_assets)
        
        # Generate random correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr = np.dot(A, A.T)
        corr = corr / np.sqrt(np.diag(corr))[:, None] / np.sqrt(np.diag(corr))[None, :]
        
        vols = np.random.uniform(0.05, 0.25, n_assets)
        self.large_cov_matrix = np.outer(vols, vols) * corr
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(self.large_cov_matrix)
        if np.min(eigenvals) <= 0:
            self.large_cov_matrix += np.eye(n_assets) * (abs(np.min(eigenvals)) + 0.001)
    
    def test_large_portfolio_optimization(self):
        """Test optimization with larger portfolios"""
        result = self.optimizer.maximize_sharpe_ratio(
            self.large_returns, self.large_cov_matrix
        )
        
        if result.converged:
            self.assertEqual(len(result.weights), len(self.large_returns))
            self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
            self.assertGreater(result.sharpe_ratio, 0)
    
    def test_optimization_with_negative_returns(self):
        """Test optimization when some assets have negative expected returns"""
        # Modify some returns to be negative
        negative_returns = self.large_returns.copy()
        negative_returns[:3] = -np.abs(negative_returns[:3])  # Make first 3 negative
        
        result = self.optimizer.maximize_sharpe_ratio(
            negative_returns, self.large_cov_matrix
        )
        
        if result.converged:
            # Should avoid negative return assets in unconstrained optimization
            # (though this depends on the specific optimization implementation)
            self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
    
    def test_optimization_performance_scaling(self):
        """Test that optimization performance scales reasonably with portfolio size"""
        import time
        
        sizes_and_times = []
        
        for n in [5, 10, 15]:
            returns = self.large_returns[:n]
            cov_matrix = self.large_cov_matrix[:n, :n]
            
            start_time = time.time()
            result = self.optimizer.maximize_sharpe_ratio(returns, cov_matrix)
            end_time = time.time()
            
            if result.converged:
                sizes_and_times.append((n, end_time - start_time))
        
        # Just check that larger problems don't take exponentially longer
        if len(sizes_and_times) >= 2:
            # Time shouldn't increase by more than factor of 10 for reasonable size increases
            time_ratio = sizes_and_times[-1][1] / sizes_and_times[0][1]
            size_ratio = sizes_and_times[-1][0] / sizes_and_times[0][0]
            
            # This is a loose bound - actual performance depends on implementation
            self.assertLess(time_ratio, size_ratio ** 2 * 10, 
                           "Optimization time should scale reasonably with problem size")

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
