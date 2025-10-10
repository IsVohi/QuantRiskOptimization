"""
Unit tests for risk calculation module

Tests both Python fallback implementations and C++ backend (if available)
for Monte Carlo VaR, Expected Shortfall, and other risk metrics.
"""

import unittest
import numpy as np
import sys
import os

# Add the frontend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend'))

from frontend.risk import create_risk_manager, check_cpp_availability

class TestRiskCalculations(unittest.TestCase):
    """Test suite for risk calculation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = create_risk_manager(seed=42, use_cpp=True)
        self.risk_manager_python = create_risk_manager(seed=42, use_cpp=False)
        
        # Test portfolio data
        self.returns = np.array([0.10, 0.08, 0.12, 0.04, 0.09])
        self.weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        
        # Create a simple covariance matrix
        np.random.seed(42)
        corr_matrix = np.array([
            [1.00, 0.30, 0.20, -0.10, 0.15],
            [0.30, 1.00, 0.25, -0.05, 0.10],
            [0.20, 0.25, 1.00, 0.00, 0.20],
            [-0.10, -0.05, 0.00, 1.00, -0.15],
            [0.15, 0.10, 0.20, -0.15, 1.00]
        ])
        
        volatilities = np.array([0.15, 0.18, 0.25, 0.05, 0.20])
        self.cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(self.cov_matrix)
        if np.min(eigenvals) <= 0:
            self.cov_matrix += np.eye(len(self.returns)) * (abs(np.min(eigenvals)) + 0.001)
    
    def test_cpp_availability(self):
        """Test C++ backend availability checking"""
        status = check_cpp_availability()
        self.assertIsInstance(status, dict)
        self.assertIn('available', status)
        self.assertIn('version', status)
        self.assertIn('error', status)
        
        if status['available']:
            self.assertIsNotNone(status['version'])
            self.assertIsNone(status['error'])
        else:
            self.assertIsNotNone(status['error'])
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        # Test with C++ backend
        rm_cpp = create_risk_manager(seed=123, use_cpp=True)
        self.assertIsNotNone(rm_cpp)
        
        # Test with Python backend
        rm_python = create_risk_manager(seed=123, use_cpp=False)
        self.assertIsNotNone(rm_python)
        self.assertFalse(rm_python.use_cpp)
    
    def test_input_validation(self):
        """Test input validation for risk calculations"""
        # Test dimension mismatch
        wrong_weights = np.array([0.5, 0.5])  # Wrong size
        
        var_result = self.risk_manager.calculate_var(
            self.returns, self.cov_matrix, wrong_weights
        )
        self.assertFalse(var_result['success'])
        self.assertIn('error', var_result)
        
        # Test weights not summing to 1
        bad_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Sum > 1
        
        var_result = self.risk_manager.calculate_var(
            self.returns, self.cov_matrix, bad_weights
        )
        self.assertFalse(var_result['success'])
    
    def test_var_calculation(self):
        """Test VaR calculation"""
        var_result = self.risk_manager.calculate_var(
            self.returns, self.cov_matrix, self.weights,
            confidence_level=0.95, num_simulations=1000
        )
        
        self.assertTrue(var_result['success'])
        self.assertIsInstance(var_result['var'], float)
        self.assertGreater(var_result['var'], 0)  # VaR should be positive (loss)
        self.assertGreater(var_result['computation_time'], 0)
        self.assertEqual(var_result['confidence_level'], 0.95)
        self.assertEqual(var_result['num_simulations'], 1000)
    
    def test_es_calculation(self):
        """Test Expected Shortfall calculation"""
        es_result = self.risk_manager.calculate_es(
            self.returns, self.cov_matrix, self.weights,
            confidence_level=0.95, num_simulations=1000
        )
        
        self.assertTrue(es_result['success'])
        self.assertIsInstance(es_result['es'], float)
        self.assertGreater(es_result['es'], 0)  # ES should be positive (loss)
        self.assertGreater(es_result['computation_time'], 0)
    
    def test_es_greater_than_var(self):
        """Test that Expected Shortfall is greater than VaR"""
        confidence_level = 0.95
        num_sims = 5000
        
        var_result = self.risk_manager.calculate_var(
            self.returns, self.cov_matrix, self.weights,
            confidence_level=confidence_level, num_simulations=num_sims
        )
        
        es_result = self.risk_manager.calculate_es(
            self.returns, self.cov_matrix, self.weights,
            confidence_level=confidence_level, num_simulations=num_sims
        )
        
        if var_result['success'] and es_result['success']:
            self.assertGreaterEqual(es_result['es'], var_result['var'],
                                   "Expected Shortfall should be >= VaR")
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        metrics = self.risk_manager.calculate_portfolio_metrics(
            self.returns, self.cov_matrix, self.weights, risk_free_rate=0.02
        )
        
        self.assertTrue(metrics['success'])
        self.assertIsInstance(metrics['expected_return'], float)
        self.assertIsInstance(metrics['volatility'], float)
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertIsInstance(metrics['risk_contributions'], list)
        
        # Check that risk contributions sum to portfolio volatility
        total_contrib = sum(metrics['risk_contributions'])
        self.assertAlmostEqual(total_contrib, metrics['volatility'], places=6)
    
    def test_simulation_generation(self):
        """Test Monte Carlo simulation path generation"""
        num_sims = 1000
        time_horizon = 5
        
        sim_paths = self.risk_manager.generate_simulation_paths(
            self.returns, self.cov_matrix, self.weights, num_sims, time_horizon
        )
        
        self.assertEqual(len(sim_paths), num_sims)
        self.assertIsInstance(sim_paths, np.ndarray)
        
        # Check that simulations have reasonable range
        self.assertGreater(np.std(sim_paths), 0)  # Should have some variance
        self.assertLess(np.abs(np.mean(sim_paths)), 1.0)  # Mean should be reasonable
    
    def test_different_confidence_levels(self):
        """Test VaR calculation with different confidence levels"""
        confidence_levels = [0.90, 0.95, 0.99]
        var_values = []
        
        for conf in confidence_levels:
            var_result = self.risk_manager.calculate_var(
                self.returns, self.cov_matrix, self.weights,
                confidence_level=conf, num_simulations=2000
            )
            if var_result['success']:
                var_values.append(var_result['var'])
        
        # VaR should increase with confidence level
        if len(var_values) == len(confidence_levels):
            for i in range(1, len(var_values)):
                self.assertGreaterEqual(var_values[i], var_values[i-1],
                                       f"VaR should increase with confidence level")
    
    def test_python_fallback(self):
        """Test Python fallback implementations"""
        # Force Python backend
        var_result_python = self.risk_manager_python.calculate_var(
            self.returns, self.cov_matrix, self.weights,
            confidence_level=0.95, num_simulations=1000
        )
        
        self.assertTrue(var_result_python['success'])
        self.assertEqual(var_result_python['method'], 'python')
        self.assertGreater(var_result_python['var'], 0)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        rm1 = create_risk_manager(seed=123, use_cpp=False)
        rm2 = create_risk_manager(seed=123, use_cpp=False)
        
        var1 = rm1.calculate_var(
            self.returns, self.cov_matrix, self.weights, num_simulations=1000
        )
        
        var2 = rm2.calculate_var(
            self.returns, self.cov_matrix, self.weights, num_simulations=1000
        )
        
        if var1['success'] and var2['success']:
            self.assertAlmostEqual(var1['var'], var2['var'], places=10,
                                  msg="Results should be reproducible with same seed")
    
    def test_performance_benchmark(self):
        """Test performance benchmarking functionality"""
        benchmark_results = self.risk_manager.benchmark_performance(
            matrix_size=50, num_iterations=10
        )
        
        self.assertIsInstance(benchmark_results, dict)
        self.assertIn('python_time_ms', benchmark_results)
        self.assertGreater(benchmark_results['python_time_ms'], 0)
        
        # If C++ is available, check speedup
        if check_cpp_availability()['available'] and 'cpp_time_ms' in benchmark_results:
            self.assertIn('speedup', benchmark_results)
            self.assertGreater(benchmark_results['speedup'], 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with zero weights (should fail)
        zero_weights = np.zeros(len(self.returns))
        var_result = self.risk_manager.calculate_var(
            self.returns, self.cov_matrix, zero_weights
        )
        self.assertFalse(var_result['success'])
        
        # Test with negative returns
        negative_returns = -np.abs(self.returns)
        var_result = self.risk_manager.calculate_var(
            negative_returns, self.cov_matrix, self.weights, num_simulations=500
        )
        self.assertTrue(var_result['success'])  # Should still work
        
        # Test with very small simulation count
        var_result = self.risk_manager.calculate_var(
            self.returns, self.cov_matrix, self.weights, num_simulations=10
        )
        self.assertTrue(var_result['success'])  # Should work but be less accurate

class TestRiskMetricsValidation(unittest.TestCase):
    """Additional tests for risk metrics validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = create_risk_manager(seed=42)
    
    def test_nan_handling(self):
        """Test handling of NaN values in inputs"""
        returns = np.array([0.1, np.nan, 0.05])
        weights = np.array([0.5, 0.3, 0.2])
        cov_matrix = np.eye(3) * 0.1
        
        var_result = self.risk_manager.calculate_var(returns, cov_matrix, weights)
        self.assertFalse(var_result['success'])
        self.assertIn('NaN', var_result['error'])
    
    def test_infinite_values(self):
        """Test handling of infinite values in inputs"""
        returns = np.array([0.1, 0.05, np.inf])
        weights = np.array([0.5, 0.3, 0.2])
        cov_matrix = np.eye(3) * 0.1
        
        var_result = self.risk_manager.calculate_var(returns, cov_matrix, weights)
        self.assertFalse(var_result['success'])
        self.assertIn('infinite', var_result['error'])
    
    def test_singular_covariance_matrix(self):
        """Test handling of singular (non-positive definite) covariance matrix"""
        returns = np.array([0.1, 0.05, 0.08])
        weights = np.array([0.5, 0.3, 0.2])
        
        # Create singular matrix
        cov_matrix = np.array([
            [0.1, 0.05, 0.1],
            [0.05, 0.025, 0.05],  # This row is 0.5 * first row
            [0.1, 0.05, 0.1]      # This row equals first row
        ])
        
        var_result = self.risk_manager.calculate_var(returns, cov_matrix, weights)
        self.assertFalse(var_result['success'])

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

