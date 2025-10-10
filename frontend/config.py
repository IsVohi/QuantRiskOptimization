"""
Configuration file for Quant Risk Optimizer

This module contains all configuration parameters, constants, and settings
for the quantitative risk management and portfolio optimization dashboard.
"""

import os
from typing import Dict, List, Any

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

# Application metadata
APP_TITLE = "Quant Risk Optimizer"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "High-Performance Portfolio Risk Management & Optimization Dashboard"
APP_AUTHOR = "Quantitative Finance Team"

# Server configuration
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 8050))

# =============================================================================
# FINANCIAL PARAMETERS
# =============================================================================

# Default risk parameters
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_TIME_HORIZON = 1  # days
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual

# Portfolio constraints
MIN_WEIGHT = 0.0  # No short selling by default
MAX_WEIGHT = 1.0  # Maximum position size
MIN_PORTFOLIO_WEIGHT = 0.01  # Minimum 1% allocation
MAX_PORTFOLIO_WEIGHT = 0.50  # Maximum 50% allocation

# Optimization parameters
OPTIMIZATION_TOLERANCE = 1e-1
MAX_OPTIMIZATION_ITERATIONS = 100000

# =============================================================================
# DASHBOARD STYLING AND LAYOUT
# =============================================================================

# Color scheme (professional blue theme)
COLORS = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#ff7f0e',    # Orange for highlights
    'success': '#2ca02c',      # Green for positive returns
    'danger': '#d62728',       # Red for losses/risk
    'warning': '#ff7f0e',      # Orange for warnings
    'info': '#17a2b8',         # Cyan for information
    'light': '#f8f9fa',        # Light gray background
    'dark': '#343a40',         # Dark text
    'background': '#ffffff',    # White background
    'grid': '#e6e6e6',         # Light gray for grid lines
    'text': '#333333',         # Dark gray text
    'muted': '#6c757d'         # Muted text
}

# Chart configuration
CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
        'hoverClosestCartesian', 'hoverCompareCartesian'
    ],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'quant_risk_chart',
        'height': 600,
        'width': 800,
        'scale': 2
    }
}

# Default chart layout
DEFAULT_CHART_LAYOUT = {
    'template': 'plotly_white',
    'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': COLORS['text']},
    'paper_bgcolor': COLORS['background'],
    'plot_bgcolor': COLORS['background'],
    'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1
    },
    'hovermode': 'closest'
}

# =============================================================================
# DATA VALIDATION AND LIMITS
# =============================================================================

# File upload limits
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_EXTENSIONS = ['.csv', '.xlsx', '.xls']

# Portfolio size limits
MIN_ASSETS = 2
MAX_ASSETS = 50

# Simulation limits
MIN_SIMULATIONS = 1000
MAX_SIMULATIONS = 100000

# Data quality thresholds
MIN_HISTORICAL_PERIODS = 30
MAX_MISSING_DATA_RATIO = 0.1  # 10% maximum missing data

# =============================================================================
# UI COMPONENT STYLES
# =============================================================================

# Sidebar styling
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20rem',
    'padding': '2rem 1rem',
    'background-color': COLORS['light'],
    'border-right': f'1px solid {COLORS["grid"]}',
    'overflow-y': 'auto'
}

# Content styling
CONTENT_STYLE = {
    'margin-left': '22rem',
    'padding': '2rem 1rem',
    'background-color': COLORS['background']
}

# Card styling
CARD_STYLE = {
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
    'border-radius': '0.5rem',
    'margin-bottom': '1rem',
    'background-color': COLORS['background']
}

# Button styling
BUTTON_STYLE = {
    'margin': '0.5rem 0.25rem',
    'border-radius': '0.375rem',
    'font-weight': '500'
}

# =============================================================================
# ASSET CLASSES AND BENCHMARKS
# =============================================================================

# Sample asset classes for demonstration
SAMPLE_ASSETS = [
    {'name': 'US Large Cap Equities', 'symbol': 'VTI', 'type': 'Equity'},
    {'name': 'International Developed Equities', 'symbol': 'VEA', 'type': 'Equity'},
    {'name': 'Emerging Markets Equities', 'symbol': 'VWO', 'type': 'Equity'},
    {'name': 'US Aggregate Bonds', 'symbol': 'BND', 'type': 'Fixed Income'},
    {'name': 'Treasury Inflation-Protected Securities', 'symbol': 'VTIP', 'type': 'Fixed Income'},
    {'name': 'Real Estate Investment Trusts', 'symbol': 'VNQ', 'type': 'Real Estate'},
    {'name': 'Commodities', 'symbol': 'VDE', 'type': 'Commodity'},
    {'name': 'Gold', 'symbol': 'IAU', 'type': 'Commodity'}
]

# Benchmark indices
BENCHMARKS = {
    'SP500': 'S&P 500 Index',
    'MSCI_WORLD': 'MSCI World Index',
    'AGG': 'Bloomberg Aggregate Bond Index',
    'GOLD': 'Gold Spot Price'
}

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

# Risk metrics to display
RISK_METRICS = [
    'Value at Risk (VaR)',
    'Expected Shortfall (ES)',
    'Maximum Drawdown',
    'Volatility',
    'Sharpe Ratio',
    'Sortino Ratio',
    'Beta',
    'Alpha'
]

# Time periods for analysis
TIME_PERIODS = {
    '1M': 21,     # 1 month (trading days)
    '3M': 63,     # 3 months
    '6M': 126,    # 6 months
    '1Y': 252,    # 1 year
    '2Y': 504,    # 2 years
    '3Y': 756,    # 3 years
    '5Y': 1260    # 5 years
}

# =============================================================================
# ERROR MESSAGES AND NOTIFICATIONS
# =============================================================================

ERROR_MESSAGES = {
    'file_upload': 'Error uploading file. Please check the format and try again.',
    'portfolio_empty': 'Please upload a portfolio or select assets.',
    'insufficient_data': 'Insufficient data for reliable calculations.',
    'optimization_failed': 'Portfolio optimization failed to converge.',
    'invalid_weights': 'Portfolio weights are invalid or do not sum to 1.',
    'simulation_error': 'Monte Carlo simulation failed. Please check parameters.',
    'matrix_error': 'Covariance matrix is not positive definite.'
}

SUCCESS_MESSAGES = {
    'optimization_complete': 'Portfolio optimization completed successfully.',
    'simulation_complete': 'Risk simulation completed successfully.',
    'data_loaded': 'Portfolio data loaded successfully.',
    'weights_updated': 'Portfolio weights updated successfully.'
}

# =============================================================================
# CACHING AND PERFORMANCE
# =============================================================================

# Cache configuration
CACHE_TYPE = 'filesystem'  # 'redis' for production
CACHE_DIR = '.cache'
CACHE_TIMEOUT = 3600  # 1 hour in seconds

# Performance monitoring
ENABLE_PERFORMANCE_LOGGING = DEBUG_MODE
PERFORMANCE_LOG_FILE = 'performance.log'

# =============================================================================
# EXPORT AND REPORTING
# =============================================================================

# Report configuration
REPORT_FORMATS = ['PDF', 'Excel', 'CSV']
DEFAULT_REPORT_FORMAT = 'PDF'

# Export file naming
EXPORT_FILENAME_TEMPLATE = "portfolio_analysis_{timestamp}"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """
    Validate configuration parameters
    
    Returns:
        Dict containing validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check critical parameters
    if MIN_SIMULATIONS > MAX_SIMULATIONS:
        validation_results['valid'] = False
        validation_results['errors'].append("MIN_SIMULATIONS cannot be greater than MAX_SIMULATIONS")
    
    if MIN_WEIGHT > MAX_WEIGHT:
        validation_results['valid'] = False
        validation_results['errors'].append("MIN_WEIGHT cannot be greater than MAX_WEIGHT")
    
    if DEFAULT_CONFIDENCE_LEVEL <= 0 or DEFAULT_CONFIDENCE_LEVEL >= 1:
        validation_results['valid'] = False
        validation_results['errors'].append("DEFAULT_CONFIDENCE_LEVEL must be between 0 and 1")
    
    # Check for reasonable values
    if MAX_ASSETS > 100:
        validation_results['warnings'].append("MAX_ASSETS is very high and may impact performance")
    
    return validation_results

def get_app_info() -> Dict[str, str]:
    """
    Get application information
    
    Returns:
        Dictionary with app metadata
    """
    return {
        'title': APP_TITLE,
        'version': APP_VERSION,
        'description': APP_DESCRIPTION,
        'author': APP_AUTHOR
    }

# Validate configuration on import
_validation = validate_config()
if not _validation['valid']:
    raise ValueError(f"Configuration validation failed: {_validation['errors']}")

if _validation['warnings'] and DEBUG_MODE:
    print(f"Configuration warnings: {_validation['warnings']}")
