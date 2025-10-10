"""
Risk calculation module for Quant Risk Optimizer

This module provides Python interfaces to high-performance C++ risk calculations
including Monte Carlo VaR, Expected Shortfall, and portfolio risk metrics.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Union
import warnings

try:
    import quant_risk_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    warnings.warn("C++ backend not available. Using Python fallback implementations.", UserWarning)

from .config import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_TIME_HORIZON,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES
)

class RiskManager:
    """
    High-level risk management interface

    This class provides a unified interface to portfolio risk calculations,
    automatically choosing between C++ and Python implementations based on
    availability and problem size.
    """

    def __init__(self, seed: int = 12345, use_cpp: bool = True):
        """
        Initialize Risk Manager

        Args:
            seed: Random seed for reproducible results
            use_cpp: Whether to use C++ backend when available
        """
        self.seed = seed
        self.use_cpp = use_cpp and CPP_AVAILABLE

        if self.use_cpp:
            self.cpp_calculator = quant_risk_core.RiskCalculator(seed)

        # Initialize random number generator for Python fallback
        np.random.seed(seed)

    def calculate_var(self,
                     returns: np.ndarray,
                     cov_matrix: np.ndarray,
                     weights: np.ndarray,
                     confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
                     num_simulations: int = DEFAULT_NUM_SIMULATIONS,
                     time_horizon: int = DEFAULT_TIME_HORIZON) -> Dict[str, Union[float, bool]]:
        """
        Calculate Value at Risk using Monte Carlo simulation

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days

        Returns:
            Dictionary with VaR result and metadata
        """
        start_time = time.time()

        # Validate inputs
        validation_result = self._validate_inputs(returns, cov_matrix, weights)
        if not validation_result['valid']:
            return {
                'var': np.nan,
                'success': False,
                'error': validation_result['error'],
                'computation_time': 0.0,
                'method': 'validation_failed'
            }

        try:
            if self.use_cpp:
                var_value = self.cpp_calculator.calculate_var(
                    returns, cov_matrix, weights, confidence_level,
                    num_simulations, time_horizon
                )
                method = 'cpp'
            else:
                var_value = self._calculate_var_python(
                    returns, cov_matrix, weights, confidence_level,
                    num_simulations, time_horizon
                )
                method = 'python'

            computation_time = time.time() - start_time

            return {
                'var': float(var_value),
                'success': True,
                'error': None,
                'computation_time': computation_time,
                'method': method,
                'confidence_level': confidence_level,
                'num_simulations': num_simulations,
                'time_horizon': time_horizon
            }

        except Exception as e:
            return {
                'var': np.nan,
                'success': False,
                'error': str(e),
                'computation_time': time.time() - start_time,
                'method': 'error'
            }

    def calculate_es(self,
                    returns: np.ndarray,
                    cov_matrix: np.ndarray,
                    weights: np.ndarray,
                    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
                    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
                    time_horizon: int = DEFAULT_TIME_HORIZON) -> Dict[str, Union[float, bool]]:
        """
        Calculate Expected Shortfall (Conditional VaR)

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            confidence_level: Confidence level
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days

        Returns:
            Dictionary with ES result and metadata
        """
        start_time = time.time()

        # Validate inputs
        validation_result = self._validate_inputs(returns, cov_matrix, weights)
        if not validation_result['valid']:
            return {
                'es': np.nan,
                'success': False,
                'error': validation_result['error'],
                'computation_time': 0.0,
                'method': 'validation_failed'
            }

        try:
            if self.use_cpp:
                es_value = self.cpp_calculator.calculate_es(
                    returns, cov_matrix, weights, confidence_level,
                    num_simulations, time_horizon
                )
                method = 'cpp'
            else:
                es_value = self._calculate_es_python(
                    returns, cov_matrix, weights, confidence_level,
                    num_simulations, time_horizon
                )
                method = 'python'

            computation_time = time.time() - start_time

            return {
                'es': float(es_value),
                'success': True,
                'error': None,
                'computation_time': computation_time,
                'method': method,
                'confidence_level': confidence_level,
                'num_simulations': num_simulations,
                'time_horizon': time_horizon
            }

        except Exception as e:
            return {
                'es': np.nan,
                'success': False,
                'error': str(e),
                'computation_time': time.time() - start_time,
                'method': 'error'
            }

    def calculate_portfolio_metrics(self,
                                   returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   weights: np.ndarray,
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            risk_free_rate: Risk-free rate for Sharpe ratio calculation

        Returns:
            Dictionary with portfolio metrics
        """
        try:
            # Portfolio return and volatility
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))

            # Sharpe ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0

            # Risk contributions
            if self.use_cpp:
                risk_contributions = self.cpp_calculator.calculate_risk_contributions(cov_matrix, weights)
            else:
                risk_contributions = self._calculate_risk_contributions_python(cov_matrix, weights)

            return {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'risk_contributions': risk_contributions.tolist() if hasattr(risk_contributions, 'tolist') else risk_contributions,
                'success': True
            }

        except Exception as e:
            return {
                'expected_return': np.nan,
                'volatility': np.nan,
                'sharpe_ratio': np.nan,
                'risk_contributions': [],
                'success': False,
                'error': str(e)
            }

    def generate_simulation_paths(self,
                                 returns: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 weights: np.ndarray,
                                 num_simulations: int = 1000,
                                 time_horizon: int = DEFAULT_TIME_HORIZON) -> np.ndarray:
        """
        Generate Monte Carlo simulation paths

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            num_simulations: Number of simulation paths
            time_horizon: Time horizon in days

        Returns:
            Array of simulated portfolio returns
        """
        if self.use_cpp:
            return np.array(self.cpp_calculator.generate_simulations(
                returns, cov_matrix, weights, num_simulations, time_horizon
            ))
        else:
            return self._generate_simulations_python(
                returns, cov_matrix, weights, num_simulations, time_horizon
            )

    def benchmark_performance(self,
                             matrix_size: int = 100,
                             num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark C++ vs Python performance

        Args:
            matrix_size: Size of test matrices
            num_iterations: Number of benchmark iterations

        Returns:
            Dictionary with timing results
        """
        results = {}

        # Benchmark C++ if available
        if self.use_cpp:
            cpp_time = quant_risk_core.benchmark_cpp_vs_python(matrix_size, num_iterations)
            results['cpp_time_ms'] = cpp_time

        # Benchmark Python
        start_time = time.time()
        A = np.random.randn(matrix_size, matrix_size)
        B = np.random.randn(matrix_size, matrix_size)
        for _ in range(num_iterations):
            C = np.dot(A, B)
            A = np.abs(C)

        python_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        results['python_time_ms'] = python_time

        if 'cpp_time_ms' in results:
            results['speedup'] = python_time / results['cpp_time_ms']

        return results

    # Private methods for Python fallback implementations
    def _validate_inputs(self,
                        returns: np.ndarray,
                        cov_matrix: np.ndarray,
                        weights: np.ndarray) -> Dict[str, Union[bool, str]]:
        """Validate input arrays for risk calculations - ENHANCED"""
        try:
            # Convert to numpy arrays if needed
            returns = np.asarray(returns, dtype=np.float64)
            cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)

            # Check shapes
            if len(returns.shape) != 1:
                return {'valid': False, 'error': 'Returns must be 1-dimensional'}

            if len(cov_matrix.shape) != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
                return {'valid': False, 'error': 'Covariance matrix must be square'}

            if len(weights.shape) != 1:
                return {'valid': False, 'error': 'Weights must be 1-dimensional'}

            # Check dimensions match
            if not (len(returns) == cov_matrix.shape[0] == len(weights)):
                return {'valid': False, 'error': 'Dimension mismatch between returns, covariance matrix, and weights'}

            # Check weights sum to approximately 1
            weight_sum = weights.sum()
            if abs(weight_sum - 1.0) > 1e-4:  # More tolerant threshold
                # Try to normalize weights
                if weight_sum > 0:
                    weights = weights / weight_sum
                else:
                    return {'valid': False, 'error': 'Weights must sum to a positive value'}

            # Check for NaN or infinite values
            if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
                return {'valid': False, 'error': 'Returns contain NaN or infinite values'}

            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                return {'valid': False, 'error': 'Covariance matrix contains NaN or infinite values'}

            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                return {'valid': False, 'error': 'Weights contain NaN or infinite values'}

            # Check positive definiteness of covariance matrix
            try:
                np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # Try to fix the matrix
                cov_matrix_fixed = self._fix_covariance_matrix(cov_matrix)
                try:
                    np.linalg.cholesky(cov_matrix_fixed)
                    # Replace original matrix with fixed version
                    # Note: This is a side effect but necessary for calculations
                except np.linalg.LinAlgError:
                    return {'valid': False, 'error': 'Covariance matrix is not positive definite and cannot be fixed'}

            return {'valid': True, 'error': None}

        except Exception as e:
            return {'valid': False, 'error': f'Input validation error: {str(e)}'}

    def _fix_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Fix a covariance matrix to be positive definite"""
        try:
            # Method: eigenvalue regularization
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Set minimum eigenvalue to small positive number
            min_eigenval = 1e-8
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            cov_matrix_fixed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Ensure symmetry
            cov_matrix_fixed = (cov_matrix_fixed + cov_matrix_fixed.T) / 2
            
            return cov_matrix_fixed
        except:
            # Fallback: diagonal matrix with variances
            n = cov_matrix.shape[0]
            diag_vals = np.diag(cov_matrix)
            diag_vals = np.maximum(diag_vals, 1e-8)  # Ensure positive
            return np.diag(diag_vals)

    def _calculate_var_python(self,
                             returns: np.ndarray,
                             cov_matrix: np.ndarray,
                             weights: np.ndarray,
                             confidence_level: float,
                             num_simulations: int,
                             time_horizon: int) -> float:
        """Python fallback for VaR calculation - IMPROVED"""
        # Generate simulations
        simulated_returns = self._generate_simulations_python(
            returns, cov_matrix, weights, num_simulations, time_horizon
        )

        # Calculate VaR
        var_index = int((1.0 - confidence_level) * num_simulations)
        var_index = max(0, min(var_index, num_simulations - 1))
        
        sorted_returns = np.sort(simulated_returns)
        return -sorted_returns[var_index]  # Return as positive loss

    def _calculate_es_python(self,
                            returns: np.ndarray,
                            cov_matrix: np.ndarray,
                            weights: np.ndarray,
                            confidence_level: float,
                            num_simulations: int,
                            time_horizon: int) -> float:
        """Python fallback for Expected Shortfall calculation - IMPROVED"""
        # Generate simulations
        simulated_returns = self._generate_simulations_python(
            returns, cov_matrix, weights, num_simulations, time_horizon
        )

        # Calculate ES
        var_index = int((1.0 - confidence_level) * num_simulations)
        var_index = max(0, min(var_index, num_simulations - 1))
        
        sorted_returns = np.sort(simulated_returns)
        
        if var_index == 0:
            return -sorted_returns[0]
        
        tail_losses = sorted_returns[:var_index + 1]
        return -np.mean(tail_losses)  # Return as positive expected loss

    def _generate_simulations_python(self,
                                    returns: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    weights: np.ndarray,
                                    num_simulations: int,
                                    time_horizon: int) -> np.ndarray:
        """Python fallback for Monte Carlo simulation - IMPROVED"""
        try:
            # Ensure covariance matrix is positive definite
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # Use eigendecomposition if Cholesky fails
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
                L = eigenvecs @ np.diag(np.sqrt(eigenvals))

            # Portfolio expected return
            portfolio_return = np.dot(weights, returns)
            time_scale = np.sqrt(time_horizon)

            # Generate simulations - vectorized for efficiency
            random_normals = np.random.normal(0, 1, (len(returns), num_simulations))
            
            # Transform to correlated shocks
            correlated_shocks = L @ random_normals * time_scale
            
            # Calculate portfolio returns
            portfolio_shocks = weights @ correlated_shocks
            simulated_returns = portfolio_return * time_horizon + portfolio_shocks

            return simulated_returns

        except Exception as e:
            # Fallback: simple uncorrelated simulation
            warnings.warn(f"Using simplified simulation due to error: {e}")
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            random_shocks = np.random.normal(0, portfolio_vol * np.sqrt(time_horizon), num_simulations)
            simulated_returns = portfolio_return * time_horizon + random_shocks
            
            return simulated_returns

    def _calculate_risk_contributions_python(self,
                                           cov_matrix: np.ndarray,
                                           weights: np.ndarray) -> np.ndarray:
        """Python fallback for risk contribution calculation"""
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))

        if portfolio_volatility == 0.0:
            return np.zeros_like(weights)

        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = (weights * marginal_contrib) / portfolio_volatility

        return risk_contrib

def create_risk_manager(seed: int = 12345, use_cpp: bool = True) -> RiskManager:
    """
    Factory function to create a RiskManager instance

    Args:
        seed: Random seed for reproducible results
        use_cpp: Whether to use C++ backend when available

    Returns:
        Configured RiskManager instance
    """
    return RiskManager(seed=seed, use_cpp=use_cpp)

def check_cpp_availability() -> Dict[str, Union[bool, str]]:
    """
    Check if C++ backend is available and functional

    Returns:
        Dictionary with availability information
    """
    if not CPP_AVAILABLE:
        return {
            'available': False,
            'version': None,
            'error': 'C++ backend not compiled or installed'
        }

    try:
        version = quant_risk_core.version()
        # Quick functionality test
        test_calc = quant_risk_core.RiskCalculator(42)
        return {
            'available': True,
            'version': version,
            'error': None
        }
    except Exception as e:
        return {
            'available': False,
            'version': None,
            'error': str(e)
        }