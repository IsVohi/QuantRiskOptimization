"""
Portfolio optimization module for Quant Risk Optimizer

This module provides Python interfaces to high-performance C++ portfolio
optimization algorithms including mean-variance optimization, efficient
frontier generation, and Sharpe ratio maximization.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import warnings

try:
    import quant_risk_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    warnings.warn("C++ backend not available. Using Python fallback implementations.", UserWarning)

from .config import (
    DEFAULT_RISK_FREE_RATE,
    OPTIMIZATION_TOLERANCE,
    MAX_OPTIMIZATION_ITERATIONS,
    MIN_WEIGHT,
    MAX_WEIGHT,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES
)

class PortfolioResult(NamedTuple):
    """Portfolio optimization result structure"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    converged: bool
    iterations: int
    objective_value: float
    computation_time: float
    method: str

class FrontierPoint(NamedTuple):
    """Efficient frontier point structure"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: np.ndarray

class EfficientFrontierResult(NamedTuple):
    """Efficient frontier result structure"""
    returns: List[float]
    volatilities: List[float]
    sharpe_ratios: List[float]
    weights_list: List[np.ndarray]
    converged: bool
    computation_time: float
    method: str

class PortfolioOptimizer:
    """
    High-level portfolio optimization interface

    This class provides a unified interface to portfolio optimization algorithms,
    automatically choosing between C++ and Python implementations based on
    availability and performance requirements.
    """

    def __init__(self,
                 tolerance: float = OPTIMIZATION_TOLERANCE,
                 max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
                 use_cpp: bool = True):
        """
        Initialize Portfolio Optimizer

        Args:
            tolerance: Convergence tolerance for optimization
            max_iterations: Maximum number of optimization iterations
            use_cpp: Whether to use C++ backend when available
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_cpp = use_cpp and CPP_AVAILABLE

        if self.use_cpp:
            self.cpp_optimizer = quant_risk_core.PortfolioOptimizer(self.tolerance, self.max_iterations)

    def maximize_sharpe_ratio(self,
                             returns: np.ndarray,
                             cov_matrix: np.ndarray,
                             risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                             min_weights: Optional[np.ndarray] = None,
                             max_weights: Optional[np.ndarray] = None) -> PortfolioResult:
        """
        Find the portfolio with maximum Sharpe ratio

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            min_weights: Minimum weight constraints (default: no short selling)
            max_weights: Maximum weight constraints (default: no limits)

        Returns:
            PortfolioResult with optimization details
        """
        start_time = time.time()

        # Validate inputs
        validation_result = self._validate_inputs(returns, cov_matrix)
        if not validation_result['valid']:
            return self._create_error_result(validation_result['error'], time.time() - start_time)

        # Set default constraints
        n_assets = len(returns)
        if min_weights is None:
            min_weights = np.full(n_assets, MIN_WEIGHT)
        if max_weights is None:
            max_weights = np.full(n_assets, MAX_WEIGHT)
        
        # Validate constraints
        if not self._validate_constraints(min_weights, max_weights, n_assets):
            return self._create_error_result("Invalid weight constraints", time.time() - start_time)

        try:
            if self.use_cpp:
                result = self.cpp_optimizer.maximize_sharpe_ratio(
                    returns, cov_matrix, risk_free_rate, min_weights, max_weights
                )
                method = 'cpp'
            else:
                result = self._maximize_sharpe_ratio_python(
                    returns, cov_matrix, risk_free_rate, min_weights, max_weights
                )
                method = 'python'

            computation_time = time.time() - start_time

            # Convert C++ result to PortfolioResult
            if self.use_cpp and hasattr(result, 'weights'):
                return PortfolioResult(
                    weights=np.array(result.weights),
                    expected_return=result.expected_return,
                    volatility=result.volatility,
                    sharpe_ratio=result.sharpe_ratio,
                    converged=result.converged,
                    iterations=result.iterations,
                    objective_value=result.objective_value,
                    computation_time=computation_time,
                    method=method
                )
            else:
                # Python result is already properly formatted
                return result._replace(
                    computation_time=computation_time,
                    method=method
                )

        except Exception as e:
            return self._create_error_result(str(e), time.time() - start_time)

    def minimize_variance(self,
                         cov_matrix: np.ndarray,
                         min_weights: Optional[np.ndarray] = None,
                         max_weights: Optional[np.ndarray] = None) -> PortfolioResult:
        """
        Find the minimum variance portfolio

        Args:
            cov_matrix: Covariance matrix
            min_weights: Minimum weight constraints
            max_weights: Maximum weight constraints

        Returns:
            PortfolioResult with optimization details
        """
        start_time = time.time()

        # Validate covariance matrix
        if not self._is_positive_definite(cov_matrix):
            return self._create_error_result("Covariance matrix is not positive definite",
                                           time.time() - start_time)

        # Set default constraints
        n_assets = cov_matrix.shape[0]
        if min_weights is None:
            min_weights = np.full(n_assets, MIN_WEIGHT)
        if max_weights is None:
            max_weights = np.full(n_assets, MAX_WEIGHT)
        
        # Validate constraints
        if not self._validate_constraints(min_weights, max_weights, n_assets):
            return self._create_error_result("Invalid weight constraints", time.time() - start_time)

        try:
            if self.use_cpp:
                result = self.cpp_optimizer.minimize_variance(cov_matrix, min_weights, max_weights)
                method = 'cpp'
            else:
                result = self._minimize_variance_python(cov_matrix, min_weights, max_weights)
                method = 'python'

            computation_time = time.time() - start_time

            # Convert C++ result to PortfolioResult
            if self.use_cpp and hasattr(result, 'weights'):
                return PortfolioResult(
                    weights=np.array(result.weights),
                    expected_return=result.expected_return,
                    volatility=result.volatility,
                    sharpe_ratio=result.sharpe_ratio,
                    converged=result.converged,
                    iterations=result.iterations,
                    objective_value=result.objective_value,
                    computation_time=computation_time,
                    method=method
                )
            else:
                return result._replace(
                    computation_time=computation_time,
                    method=method
                )

        except Exception as e:
            return self._create_error_result(str(e), time.time() - start_time)

    def optimize_for_return(self,
                           returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           target_return: float,
                           min_weights: Optional[np.ndarray] = None,
                           max_weights: Optional[np.ndarray] = None) -> PortfolioResult:
        """
        Find the minimum variance portfolio for a target return

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            target_return: Target portfolio return
            min_weights: Minimum weight constraints
            max_weights: Maximum weight constraints

        Returns:
            PortfolioResult with optimization details
        """
        start_time = time.time()

        # Validate inputs
        validation_result = self._validate_inputs(returns, cov_matrix)
        if not validation_result['valid']:
            return self._create_error_result(validation_result['error'], time.time() - start_time)

        # Check if target return is achievable - IMPROVED BOUNDS CHECKING
        min_possible_return = np.min(returns) * 0.95  # Add 5% buffer
        max_possible_return = np.max(returns) * 1.05  # Add 5% buffer
        
        if target_return < min_possible_return or target_return > max_possible_return:
            return self._create_error_result(
                f"Target return {target_return:.4f} is outside achievable range "
                f"[{min_possible_return:.4f}, {max_possible_return:.4f}]",
                time.time() - start_time
            )

        # Set default constraints
        n_assets = len(returns)
        if min_weights is None:
            min_weights = np.full(n_assets, MIN_WEIGHT)
        if max_weights is None:
            max_weights = np.full(n_assets, MAX_WEIGHT)
        
        # Validate constraints
        if not self._validate_constraints(min_weights, max_weights, n_assets):
            return self._create_error_result("Invalid weight constraints", time.time() - start_time)

        try:
            if self.use_cpp:
                result = self.cpp_optimizer.optimize_for_return(
                    returns, cov_matrix, target_return, min_weights, max_weights
                )
                method = 'cpp'
            else:
                result = self._optimize_for_return_python(
                    returns, cov_matrix, target_return, min_weights, max_weights
                )
                method = 'python'

            computation_time = time.time() - start_time

            # Convert C++ result to PortfolioResult
            if self.use_cpp and hasattr(result, 'weights'):
                return PortfolioResult(
                    weights=np.array(result.weights),
                    expected_return=result.expected_return,
                    volatility=result.volatility,
                    sharpe_ratio=result.sharpe_ratio,
                    converged=result.converged,
                    iterations=result.iterations,
                    objective_value=result.objective_value,
                    computation_time=computation_time,
                    method=method
                )
            else:
                return result._replace(
                    computation_time=computation_time,
                    method=method
                )

        except Exception as e:
            return self._create_error_result(str(e), time.time() - start_time)

    def generate_efficient_frontier(self,
                                  returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  num_points: int = 50,
                                  min_weights: Optional[np.ndarray] = None,
                                  max_weights: Optional[np.ndarray] = None) -> EfficientFrontierResult:
        """
        Generate efficient frontier points

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            num_points: Number of frontier points to generate
            min_weights: Minimum weight constraints
            max_weights: Maximum weight constraints

        Returns:
            EfficientFrontierResult with frontier data
        """
        start_time = time.time()
        
        # Validate inputs
        validation_result = self._validate_inputs(returns, cov_matrix)
        if not validation_result['valid']:
            return EfficientFrontierResult(
                returns=[], volatilities=[], sharpe_ratios=[], weights_list=[],
                converged=False, computation_time=0.0, method='error'
            )

        # Set default constraints
        n_assets = len(returns)
        if min_weights is None:
            min_weights = np.full(n_assets, MIN_WEIGHT)
        if max_weights is None:
            max_weights = np.full(n_assets, MAX_WEIGHT)
        
        # Validate constraints
        if not self._validate_constraints(min_weights, max_weights, n_assets):
            return EfficientFrontierResult(
                returns=[], volatilities=[], sharpe_ratios=[], weights_list=[],
                converged=False, computation_time=0.0, method='error'
            )

        try:
            if self.use_cpp:
                cpp_points = self.cpp_optimizer.generate_efficient_frontier(
                    returns, cov_matrix, num_points, min_weights, max_weights
                )
                # Convert C++ frontier points to Python objects
                frontier_returns = [point.expected_return for point in cpp_points]
                frontier_vols = [point.volatility for point in cpp_points]
                frontier_sharpes = [point.sharpe_ratio for point in cpp_points]
                frontier_weights = [np.array(point.weights) for point in cpp_points]
                method = 'cpp'
            else:
                frontier_data = self._generate_efficient_frontier_python(
                    returns, cov_matrix, num_points, min_weights, max_weights
                )
                frontier_returns = frontier_data['returns']
                frontier_vols = frontier_data['volatilities']
                frontier_sharpes = frontier_data['sharpe_ratios']
                frontier_weights = frontier_data['weights_list']
                method = 'python'

            computation_time = time.time() - start_time

            return EfficientFrontierResult(
                returns=frontier_returns,
                volatilities=frontier_vols,
                sharpe_ratios=frontier_sharpes,
                weights_list=frontier_weights,
                converged=True,
                computation_time=computation_time,
                method=method
            )

        except Exception as e:
            warnings.warn(f"Error generating efficient frontier: {str(e)}")
            return EfficientFrontierResult(
                returns=[], volatilities=[], sharpe_ratios=[], weights_list=[],
                converged=False, computation_time=time.time() - start_time, method='error'
            )

    def calculate_portfolio_metrics(self,
                                   returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   weights: np.ndarray,
                                   risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics

        Args:
            returns: Expected returns array
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with portfolio metrics
        """
        try:
            if self.use_cpp:
                metrics = self.cpp_optimizer.calculate_portfolio_metrics(
                    returns, cov_matrix, weights, risk_free_rate
                )
                return {
                    'expected_return': metrics[0],
                    'volatility': metrics[1],
                    'sharpe_ratio': metrics[2]
                }
            else:
                return self._calculate_portfolio_metrics_python(
                    returns, cov_matrix, weights, risk_free_rate
                )

        except Exception as e:
            return {
                'expected_return': np.nan,
                'volatility': np.nan,
                'sharpe_ratio': np.nan,
                'error': str(e)
            }

    def validate_weights(self,
                        weights: np.ndarray,
                        min_weights: Optional[np.ndarray] = None,
                        max_weights: Optional[np.ndarray] = None,
                        tolerance: float = 1e-6) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate portfolio weights

        Args:
            weights: Portfolio weights to validate
            min_weights: Minimum weight constraints
            max_weights: Maximum weight constraints
            tolerance: Numerical tolerance for validation

        Returns:
            Dictionary with validation results
        """
        violations = []

        # Check if weights sum to 1
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > tolerance:
            violations.append(f"Weights sum to {weight_sum:.6f}, should be 1.0")

        # Check minimum weight constraints
        if min_weights is not None:
            below_min = weights < (min_weights - tolerance)
            if np.any(below_min):
                violations.append(f"{np.sum(below_min)} weights below minimum constraints")

        # Check maximum weight constraints
        if max_weights is not None:
            above_max = weights > (max_weights + tolerance)
            if np.any(above_max):
                violations.append(f"{np.sum(above_max)} weights above maximum constraints")

        # Check for NaN or infinite values
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            violations.append("Weights contain NaN or infinite values")

        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'weight_sum': weight_sum,
            'num_assets': len(weights)
        }

    # Private methods for validation and error handling
    def _validate_inputs(self,
                        returns: np.ndarray,
                        cov_matrix: np.ndarray) -> Dict[str, Union[bool, str]]:
        """Validate input arrays for optimization"""
        try:
            # Convert to numpy arrays if needed
            returns = np.asarray(returns, dtype=np.float64)
            cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
            
            # Check shapes
            if len(returns.shape) != 1:
                return {'valid': False, 'error': 'Returns must be 1-dimensional'}

            if len(cov_matrix.shape) != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
                return {'valid': False, 'error': 'Covariance matrix must be square'}

            # Check dimensions match
            if len(returns) != cov_matrix.shape[0]:
                return {'valid': False, 'error': 'Returns and covariance matrix dimensions do not match'}

            # Check for NaN or infinite values
            if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
                return {'valid': False, 'error': 'Returns contain NaN or infinite values'}

            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                return {'valid': False, 'error': 'Covariance matrix contains NaN or infinite values'}

            # Check positive definiteness with better error handling
            if not self._is_positive_definite(cov_matrix):
                # Try to make matrix positive definite
                cov_matrix_fixed = self._fix_covariance_matrix(cov_matrix)
                if not self._is_positive_definite(cov_matrix_fixed):
                    return {'valid': False, 'error': 'Covariance matrix is not positive definite and cannot be fixed'}

            return {'valid': True, 'error': None}

        except Exception as e:
            return {'valid': False, 'error': f'Input validation error: {str(e)}'}

    def _validate_constraints(self, min_weights: np.ndarray, max_weights: np.ndarray, n_assets: int) -> bool:
        """Validate weight constraints"""
        try:
            # Check array shapes
            if len(min_weights) != n_assets or len(max_weights) != n_assets:
                return False
            
            # Check that min <= max
            if np.any(min_weights > max_weights):
                return False
            
            # Check feasibility: sum of min weights <= 1 <= sum of max weights
            if np.sum(min_weights) > 1.0 + 1e-6:
                return False
            if np.sum(max_weights) < 1.0 - 1e-6:
                return False
            
            return True
        except:
            return False

    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite"""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def _fix_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Fix a covariance matrix to be positive definite"""
        try:
            # Method 1: Add small diagonal term
            eigenvals = np.linalg.eigvals(cov_matrix)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval <= 0:
                # Add small positive value to diagonal
                regularization = abs(min_eigenval) + 1e-8
                cov_matrix_fixed = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization
                return cov_matrix_fixed
            
            return cov_matrix
        except:
            # Fallback: return identity matrix scaled by mean variance
            n = cov_matrix.shape[0]
            mean_var = np.mean(np.diag(cov_matrix))
            return np.eye(n) * mean_var

    def _create_error_result(self, error: str, computation_time: float) -> PortfolioResult:
        """Create an error result"""
        return PortfolioResult(
            weights=np.array([]),
            expected_return=np.nan,
            volatility=np.nan,
            sharpe_ratio=np.nan,
            converged=False,
            iterations=0,
            objective_value=np.nan,
            computation_time=computation_time,
            method='error'
        )

    # Python fallback implementations
    def _maximize_sharpe_ratio_python(self,
                                     returns: np.ndarray,
                                     cov_matrix: np.ndarray,
                                     risk_free_rate: float,
                                     min_weights: np.ndarray,
                                     max_weights: np.ndarray) -> PortfolioResult:
        """Python fallback for Sharpe ratio maximization"""
        # Simplified implementation using scipy.optimize
        from scipy.optimize import minimize

        n_assets = len(returns)

        def objective(w):
            portfolio_return = np.dot(w, returns)
            portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - risk_free_rate) / portfolio_vol

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        bounds = [(min_weights[i], max_weights[i]) for i in range(n_assets)]

        # Better initial guess - try multiple starting points
        best_result = None
        best_objective = np.inf
        
        for _ in range(5):  # Try 5 different starting points
            x0 = np.random.dirichlet(np.ones(n_assets))  # Random valid weights
            x0 = np.clip(x0, min_weights, max_weights)
            x0 = x0 / np.sum(x0)  # Normalize
            
            try:
                result = minimize(
                    objective, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
                
                if result.success and result.fun < best_objective:
                    best_result = result
                    best_objective = result.fun
            except:
                continue

        if best_result is not None and best_result.success:
            weights = best_result.x
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0

            return PortfolioResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                converged=True,
                iterations=best_result.nit,
                objective_value=-best_result.fun,
                computation_time=0.0,  # Will be set by caller
                method='python'
            )
        else:
            error_msg = "Optimization failed to converge" if best_result is None else best_result.message
            return self._create_error_result(f"Optimization failed: {error_msg}", 0.0)

    def _minimize_variance_python(self,
                                 cov_matrix: np.ndarray,
                                 min_weights: np.ndarray,
                                 max_weights: np.ndarray) -> PortfolioResult:
        """Python fallback for minimum variance optimization"""
        from scipy.optimize import minimize

        n_assets = cov_matrix.shape[0]

        def objective(w):
            return 0.5 * np.dot(w.T, np.dot(cov_matrix, w))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        bounds = [(min_weights[i], max_weights[i]) for i in range(n_assets)]

        # Better initial guess
        x0 = np.full(n_assets, 1.0 / n_assets)
        x0 = np.clip(x0, min_weights, max_weights)
        x0 = x0 / np.sum(x0)

        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            portfolio_vol = np.sqrt(2 * result.fun)  # Since objective is 0.5 * variance

            return PortfolioResult(
                weights=weights,
                expected_return=0.0,  # Not specified for minimum variance
                volatility=portfolio_vol,
                sharpe_ratio=0.0,
                converged=True,
                iterations=result.nit,
                objective_value=result.fun,
                computation_time=0.0,  # Will be set by caller
                method='python'
            )
        else:
            return self._create_error_result(f"Optimization failed: {result.message}", 0.0)

    def _optimize_for_return_python(self,
                                   returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   target_return: float,
                                   min_weights: np.ndarray,
                                   max_weights: np.ndarray) -> PortfolioResult:
        """Python fallback for target return optimization"""
        from scipy.optimize import minimize

        n_assets = len(returns)

        def objective(w):
            return 0.5 * np.dot(w.T, np.dot(cov_matrix, w))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, returns) - target_return}
        ]

        bounds = [(min_weights[i], max_weights[i]) for i in range(n_assets)]

        # Better initial guess
        x0 = np.full(n_assets, 1.0 / n_assets)
        x0 = np.clip(x0, min_weights, max_weights)
        x0 = x0 / np.sum(x0)

        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(2 * result.fun)
            sharpe_ratio = (portfolio_return - DEFAULT_RISK_FREE_RATE) / portfolio_vol if portfolio_vol > 0 else 0.0

            return PortfolioResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                converged=True,
                iterations=result.nit,
                objective_value=result.fun,
                computation_time=0.0,  # Will be set by caller
                method='python'
            )
        else:
            return self._create_error_result(f"Optimization failed: {result.message}", 0.0)

    def _generate_efficient_frontier_python(self,
                                           returns: np.ndarray,
                                           cov_matrix: np.ndarray,
                                           num_points: int,
                                           min_weights: np.ndarray,
                                           max_weights: np.ndarray) -> Dict:
        """Python fallback for efficient frontier generation"""
        frontier_returns = []
        frontier_vols = []
        frontier_sharpes = []
        frontier_weights = []

        # Generate return targets - IMPROVED RANGE
        min_return = np.min(returns) * 0.8  # More conservative lower bound
        max_return = np.max(returns) * 1.2  # More aggressive upper bound
        return_targets = np.linspace(min_return, max_return, num_points)

        successful_points = 0
        for target_return in return_targets:
            try:
                result = self._optimize_for_return_python(
                    returns, cov_matrix, target_return, min_weights, max_weights
                )

                if result.converged:
                    frontier_returns.append(result.expected_return)
                    frontier_vols.append(result.volatility)
                    frontier_sharpes.append(result.sharpe_ratio)
                    frontier_weights.append(result.weights)
                    successful_points += 1
            except Exception:
                continue  # Skip this point if optimization fails

        # Sort by volatility
        if frontier_vols:
            sorted_indices = np.argsort(frontier_vols)
            frontier_returns = [frontier_returns[i] for i in sorted_indices]
            frontier_vols = [frontier_vols[i] for i in sorted_indices]
            frontier_sharpes = [frontier_sharpes[i] for i in sorted_indices]
            frontier_weights = [frontier_weights[i] for i in sorted_indices]

        return {
            'returns': frontier_returns,
            'volatilities': frontier_vols,
            'sharpe_ratios': frontier_sharpes,
            'weights_list': frontier_weights
        }

    def _calculate_portfolio_metrics_python(self,
                                           returns: np.ndarray,
                                           cov_matrix: np.ndarray,
                                           weights: np.ndarray,
                                           risk_free_rate: float) -> Dict[str, float]:
        """Python fallback for portfolio metrics calculation"""
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(max(0.0, portfolio_variance))

        sharpe_ratio = ((portfolio_return - risk_free_rate) / portfolio_volatility
                       if portfolio_volatility > 0 else 0.0)

        return {
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio)
        }

def create_optimizer(tolerance: float = OPTIMIZATION_TOLERANCE,
                    max_iterations: int = MAX_OPTIMIZATION_ITERATIONS,
                    use_cpp: bool = True) -> PortfolioOptimizer:
    """
    Factory function to create a PortfolioOptimizer instance

    Args:
        tolerance: Convergence tolerance for optimization
        max_iterations: Maximum number of optimization iterations
        use_cpp: Whether to use C++ backend when available

    Returns:
        Configured PortfolioOptimizer instance
    """
    return PortfolioOptimizer(
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_cpp=use_cpp
    )