"""
Main Dash application for Quant Risk Optimizer

This module contains the complete dashboard interface with interactive
components for portfolio risk analysis and optimization.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import io
import base64
import time
from typing import Dict, List, Optional, Tuple

# Import our custom modules
from .config import *
from .risk import create_risk_manager, check_cpp_availability
from .optimize import create_optimizer

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = APP_TITLE

# Global variables for state management
risk_manager = create_risk_manager()
optimizer = create_optimizer()

# Check C++ backend availability
cpp_status = check_cpp_availability()

def create_sidebar():
    """Create the sidebar with controls"""
    return dbc.Card([
        dbc.CardHeader(html.H4("Portfolio Controls", className="text-center")),
        dbc.CardBody([
            # File Upload Section
            html.Div([
                html.H5("Upload Portfolio", className="mb-3"),
                dcc.Upload(
                    id='upload-portfolio',
                    children=dbc.Button(
                        "Upload CSV/Excel File",
                        color="primary",
                        className="w-100"
                    ),
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                dbc.Button(
                    "Use Sample Portfolio",
                    id="load-sample-btn",
                    color="secondary",
                    className="w-100 mb-3"
                ),
            ]),
            html.Hr(),

            # Risk Parameters Section
            html.Div([
                html.H5("Risk Parameters", className="mb-3"),
                html.Label("Confidence Level (%):"),
                dcc.Slider(
                    id='confidence-slider',
                    min=90, max=99, step=0.5, value=95,
                    marks={i: f'{i}%' for i in range(90, 100, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),

                html.Label("Number of Simulations:", className="mt-3"),
                dcc.Dropdown(
                    id='simulations-dropdown',
                    options=[
                        {'label': '1,000', 'value': 1000},
                        {'label': '5,000', 'value': 5000},
                        {'label': '10,000', 'value': 10000},
                        {'label': '25,000', 'value': 25000},
                        {'label': '50,000', 'value': 50000}
                    ],
                    value=10000
                ),

                html.Label("Time Horizon (days):", className="mt-3"),
                dcc.Input(
                    id='time-horizon-input',
                    type='number',
                    value=1,
                    min=1,
                    max=252,
                    step=1,
                    style={'width': '100%'}
                ),

                html.Label("Risk-Free Rate (%):", className="mt-3"),
                dcc.Input(
                    id='risk-free-rate-input',
                    type='number',
                    value=2.0,
                    min=0,
                    max=10,
                    step=0.1,
                    style={'width': '100%'}
                ),
            ]),
            html.Hr(),

            # Optimization Parameters Section
            html.Div([
                html.H5("Optimization Settings", className="mb-3"),
                html.Label("Objective:"),
                dcc.RadioItems(
                    id='optimization-objective',
                    options=[
                        {'label': 'Maximize Sharpe Ratio', 'value': 'sharpe'},
                        {'label': 'Minimize Variance', 'value': 'variance'},
                        {'label': 'Target Return', 'value': 'target'}
                    ],
                    value='sharpe',
                    className="mb-3"
                ),

                html.Div(id='target-return-input-div', style={'display': 'none'}, children=[
                    html.Label("Target Return (%):"),
                    dcc.Input(
                        id='target-return-input',
                        type='number',
                        value=8.0,
                        min=0,
                        max=20,
                        step=0.1,
                        style={'width': '100%'}
                    )
                ]),

                html.Label("Weight Constraints:", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Min %:", style={'fontSize': '0.9em'}),
                        dcc.Input(
                            id='min-weight-input',
                            type='number',
                            value=0.0,  # Relaxed from previous setting
                            min=0,
                            max=50,
                            step=0.1,
                            style={'width': '100%'}
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Max %:", style={'fontSize': '0.9em'}),
                        dcc.Input(
                            id='max-weight-input',
                            type='number',
                            value=80.0,  # Relaxed from previous setting
                            min=1,
                            max=100,
                            step=0.1,
                            style={'width': '100%'}
                        )
                    ], width=6)
                ])
            ]),
            html.Hr(),

            # Action Buttons
            html.Div([
                dbc.Button(
                    "Calculate Risk Metrics",
                    id="calculate-risk-btn",
                    color="success",
                    className="w-100 mb-2"
                ),
                dbc.Button(
                    "Optimize Portfolio",
                    id="optimize-btn",
                    color="warning",
                    className="w-100 mb-2"
                ),
                dbc.Button(
                    "Generate Efficient Frontier",
                    id="frontier-btn",
                    color="info",
                    className="w-100 mb-3"
                ),
            ]),

            # Backend Status
            html.Hr(),
            html.Div([
                html.H6("System Status", className="mb-2"),
                html.P([
                    html.Strong("Backend: "),
                    html.Span("C++ Available" if cpp_status['available'] else "Python Only",
                             className="text-success" if cpp_status['available'] else "text-warning")
                ], className="mb-1 small"),
                html.P([
                    html.Strong("Version: "),
                    html.Span(cpp_status.get('version', 'N/A'))
                ], className="mb-0 small")
            ])
        ])
    ], style=SIDEBAR_STYLE)

def create_main_content():
    """Create the main content area"""
    return html.Div([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1(APP_TITLE, className="mb-2"),
                html.P(APP_DESCRIPTION, className="lead mb-4")
            ])
        ]),

        # Alert for messages
        dbc.Row([
            dbc.Col([
                dbc.Alert(id="alert-message", is_open=False, dismissable=True)
            ])
        ]),

        # Portfolio Overview
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Portfolio Overview")),
                    dbc.CardBody([
                        html.Div(id="portfolio-overview", children=[
                            html.P("Upload a portfolio to begin analysis.", className="text-muted")
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),

        # Risk Metrics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Value at Risk", className="card-title"),
                        html.H4(id="var-value", children="--", className="text-danger"),
                        html.P("95% confidence", className="small text-muted mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Expected Shortfall", className="card-title"),
                        html.H4(id="es-value", children="--", className="text-danger"),
                        html.P("Conditional VaR", className="small text-muted mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Portfolio Volatility", className="card-title"),
                        html.H4(id="volatility-value", children="--", className="text-warning"),
                        html.P("Annualized", className="small text-muted mb-0")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Sharpe Ratio", className="card-title"),
                        html.H4(id="sharpe-value", children="--", className="text-success"),
                        html.P("Risk-adjusted return", className="small text-muted mb-0")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),

        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Monte Carlo Simulation Results", className="mb-0"),
                        dbc.Badge("Histogram of Simulated Returns", color="secondary", className="float-end")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="simulation-histogram")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Efficient Frontier", className="mb-0"),
                        dbc.Badge("Risk vs Return", color="secondary", className="float-end")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="efficient-frontier")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),

        # Portfolio Composition and Performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Optimal Portfolio Weights", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="portfolio-weights-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Performance Metrics", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="performance-table")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),

        # Additional Analysis Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Risk Contribution Analysis", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="risk-contribution-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Correlation Heatmap", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="correlation-heatmap")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),

        # Latency Benchmark
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Performance Benchmark", className="mb-0"),
                        dbc.Button(
                            "Run Benchmark",
                            id="benchmark-btn",
                            color="outline-primary",
                            size="sm",
                            className="float-end"
                        )
                    ]),
                    dbc.CardBody([
                        html.Div(id="benchmark-results")
                    ])
                ])
            ], width=12)
        ])

    ], style=CONTENT_STYLE)

# App Layout
app.layout = dbc.Container([
    dcc.Store(id='portfolio-data-store'),
    dcc.Store(id='optimization-results-store'),
    dcc.Store(id='frontier-results-store'),
    create_sidebar(),
    create_main_content()
], fluid=True)

# Callbacks
@app.callback(
    Output('target-return-input-div', 'style'),
    Input('optimization-objective', 'value')
)
def toggle_target_return_input(objective):
    """Show/hide target return input based on optimization objective"""
    if objective == 'target':
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    [Output('portfolio-data-store', 'data'),
     Output('alert-message', 'children'),
     Output('alert-message', 'color'),
     Output('alert-message', 'is_open')],
    [Input('upload-portfolio', 'contents'),
     Input('load-sample-btn', 'n_clicks')],
    State('upload-portfolio', 'filename')
)
def handle_portfolio_data(contents, sample_clicks, filename):
    """Handle portfolio data upload or sample loading - ENHANCED"""
    ctx = callback_context
    if not ctx.triggered:
        return None, "", "primary", False

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if trigger_id == 'load-sample-btn' and sample_clicks:
            # Create sample portfolio data with REALISTIC values
            sample_data = {
                'assets': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
                'weights': [0.125]*8,
                'returns': [0.08, 0.10, 0.07, 0.09, 0.12, 0.13, 0.11, 0.12],
                'volatilities': [0.18, 0.20, 0.16, 0.22, 0.28, 0.33, 0.19, 0.23]
            }

            # Generate realistic covariance matrix - IMPROVED
            n_assets = len(sample_data['assets'])
            
            # Create correlation matrix with realistic values
            np.random.seed(42)  # For reproducibility
            base_corr = 0.3  # Base correlation
            corr_matrix = np.full((n_assets, n_assets), base_corr)
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Add some random variation to correlations
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    corr_val = base_corr + np.random.normal(0, 0.1)
                    corr_val = np.clip(corr_val, 0.1, 0.8)  # Keep correlations reasonable
                    corr_matrix[i, j] = corr_val
                    corr_matrix[j, i] = corr_val
            
            # Ensure positive definiteness
            eigenvals = np.linalg.eigvals(corr_matrix)
            if np.min(eigenvals) <= 0:
                corr_matrix += np.eye(n_assets) * (abs(np.min(eigenvals)) + 0.01)
            
            vols = np.array(sample_data['volatilities'])
            cov_matrix = np.outer(vols, vols) * corr_matrix

            sample_data['cov_matrix'] = cov_matrix.tolist()

            return (sample_data, "Sample portfolio loaded successfully!", "success", True)

        elif trigger_id == 'upload-portfolio' and contents:
            # Parse uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            if filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                return None, "Unsupported file format. Please use CSV or Excel.", "danger", True

            # Validate required columns
            required_cols = ['Asset', 'Weight', 'Expected_Return', 'Volatility']
            if not all(col in df.columns for col in required_cols):
                return (None, f"Missing required columns. Need: {required_cols}", "danger", True)

            # Basic data validation - IMPROVED
            weight_sum = df['Weight'].sum()
            if abs(weight_sum - 1.0) > 0.05:  # More tolerant
                # Try to normalize weights
                df['Weight'] = df['Weight'] / weight_sum
                
            # Check for negative or zero values
            if (df['Weight'] < 0).any():
                return (None, "Negative weights not allowed in this version", "danger", True)
            
            if (df['Expected_Return'] <= 0).any() or (df['Volatility'] <= 0).any():
                return (None, "Expected returns and volatilities must be positive", "danger", True)

            # Generate realistic covariance matrix
            n_assets = len(df)
            np.random.seed(42)
            
            # Create reasonable correlation structure
            corr_matrix = np.eye(n_assets) * 0.4 + np.ones((n_assets, n_assets)) * 0.2
            
            # Ensure positive definiteness
            eigenvals = np.linalg.eigvals(corr_matrix)
            if np.min(eigenvals) <= 0:
                corr_matrix += np.eye(n_assets) * (abs(np.min(eigenvals)) + 0.01)
            
            vols = df['Volatility'].values
            cov_matrix = np.outer(vols, vols) * corr_matrix

            portfolio_data = {
                'assets': df['Asset'].tolist(),
                'weights': df['Weight'].tolist(),
                'returns': df['Expected_Return'].tolist(),
                'volatilities': df['Volatility'].tolist(),
                'cov_matrix': cov_matrix.tolist()
            }

            return (portfolio_data, f"Portfolio '{filename}' uploaded successfully!", "success", True)

    except Exception as e:
        error_msg = f"Error processing portfolio data: {str(e)}"
        return None, error_msg, "danger", True

    return None, "", "primary", False

@app.callback(
    Output('portfolio-overview', 'children'),
    Input('portfolio-data-store', 'data')
)
def update_portfolio_overview(portfolio_data):
    """Update portfolio overview display"""
    if not portfolio_data:
        return html.P("Upload a portfolio to begin analysis.", className="text-muted")

    df = pd.DataFrame({
        'Asset': portfolio_data['assets'],
        'Weight': [f"{w:.1%}" for w in portfolio_data['weights']],
        'Expected Return': [f"{r:.1%}" for r in portfolio_data['returns']],
        'Volatility': [f"{v:.1%}" for v in portfolio_data['volatilities']]
    })

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'},
        style_data={'backgroundColor': COLORS['light']},
        page_size=10
    )

@app.callback(
    [Output('var-value', 'children'),
     Output('es-value', 'children'),
     Output('volatility-value', 'children'),
     Output('sharpe-value', 'children'),
     Output('simulation-histogram', 'figure'),
     Output('risk-contribution-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('alert-message', 'children', allow_duplicate=True),
     Output('alert-message', 'color', allow_duplicate=True),
     Output('alert-message', 'is_open', allow_duplicate=True)],
    Input('calculate-risk-btn', 'n_clicks'),
    [State('portfolio-data-store', 'data'),
     State('confidence-slider', 'value'),
     State('simulations-dropdown', 'value'),
     State('time-horizon-input', 'value'),
     State('risk-free-rate-input', 'value')],
    prevent_initial_call=True
)
def calculate_risk_metrics(n_clicks, portfolio_data, confidence, num_sims, time_horizon, risk_free_rate):
    """Calculate and display risk metrics - ENHANCED ERROR HANDLING"""
    if not n_clicks or not portfolio_data:
        return "--", "--", "--", "--", {}, {}, {}, "", "primary", False

    try:
        # Convert data to numpy arrays with validation
        returns = np.asarray(portfolio_data['returns'], dtype=np.float64)
        weights = np.asarray(portfolio_data['weights'], dtype=np.float64)
        cov_matrix = np.asarray(portfolio_data['cov_matrix'], dtype=np.float64)
        
        # Validate inputs
        if np.any(np.isnan(returns)) or np.any(np.isnan(weights)) or np.any(np.isnan(cov_matrix)):
            return "--", "--", "--", "--", {}, {}, {}, "Invalid data: contains NaN values", "danger", True
        
        confidence_level = confidence / 100
        rf_rate = risk_free_rate / 100

        # Calculate risk metrics
        start_time = time.time()
        
        var_result = risk_manager.calculate_var(
            returns, cov_matrix, weights, confidence_level, num_sims, time_horizon
        )

        es_result = risk_manager.calculate_es(
            returns, cov_matrix, weights, confidence_level, num_sims, time_horizon
        )

        portfolio_metrics = risk_manager.calculate_portfolio_metrics(
            returns, cov_matrix, weights, rf_rate
        )

        # Generate simulation paths
        try:
            sim_paths = risk_manager.generate_simulation_paths(
                returns, cov_matrix, weights, min(num_sims, 5000), time_horizon
            )
        except Exception as e:
            # Fallback simulation
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sim_paths = np.random.normal(
                portfolio_return * time_horizon,
                portfolio_vol * np.sqrt(time_horizon),
                min(num_sims, 5000)
            )

        calc_time = time.time() - start_time

        # Create histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=sim_paths,
            nbinsx=50,
            name='Simulated Returns',
            marker_color=COLORS['primary'],
            opacity=0.7
        ))

        # Add VaR and ES lines
        if var_result['success']:
            hist_fig.add_vline(
                x=-var_result['var'],
                line_dash="dash",
                line_color=COLORS['danger'],
                annotation_text=f"VaR ({confidence}%)"
            )

        if es_result['success']:
            hist_fig.add_vline(
                x=-es_result['es'],
                line_dash="dot",
                line_color=COLORS['danger'],
                annotation_text=f"ES ({confidence}%)"
            )

        hist_fig.update_layout(
            **DEFAULT_CHART_LAYOUT,
            title="Monte Carlo Simulation Results",
            xaxis_title="Portfolio Return",
            yaxis_title="Frequency"
        )

        # Create risk contribution chart
        risk_contrib_fig = go.Figure()
        if portfolio_metrics['success']:
            # Calculate individual asset risk contributions
            portfolio_vol = portfolio_metrics['volatility']
            if portfolio_vol > 0:
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib / portfolio_vol
                
                risk_contrib_fig.add_trace(go.Bar(
                    x=portfolio_data['assets'],
                    y=risk_contrib * 100,
                    marker_color=COLORS['warning'],
                    name='Risk Contribution'
                ))
            
        risk_contrib_fig.update_layout(
            **DEFAULT_CHART_LAYOUT,
            title="Risk Contribution by Asset",
            xaxis_title="Asset",
            yaxis_title="Risk Contribution (%)"
        )

        # Create correlation heatmap
        if cov_matrix.size > 0:
            diag_sqrt = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(diag_sqrt, diag_sqrt)
        else:
            corr_matrix = np.eye(len(portfolio_data['assets']))
            
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=portfolio_data['assets'],
            y=portfolio_data['assets'],
            colorscale='RdYlBu_r',
            zmid=0,
            text=np.round(corr_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True
        ))

        heatmap_fig.update_layout(
            **DEFAULT_CHART_LAYOUT,
            title="Asset Correlation Matrix"
        )

        # Format display values
        var_display = f"{var_result['var']:.2%}" if var_result['success'] else "Error"
        es_display = f"{es_result['es']:.2%}" if es_result['success'] else "Error"
        vol_display = f"{portfolio_metrics['volatility']:.2%}" if portfolio_metrics['success'] else "Error"
        sharpe_display = f"{portfolio_metrics['sharpe_ratio']:.3f}" if portfolio_metrics['success'] else "Error"

        success_msg = f"Risk calculation completed in {calc_time:.3f}s using {var_result.get('method', 'unknown')} backend"

        return (var_display, es_display, vol_display, sharpe_display,
                hist_fig, risk_contrib_fig, heatmap_fig,
                success_msg, "success", True)

    except Exception as e:
        error_msg = f"Error calculating risk metrics: {str(e)}"
        return "--", "--", "--", "--", {}, {}, {}, error_msg, "danger", True

@app.callback(
    [Output('optimization-results-store', 'data'),
     Output('portfolio-weights-chart', 'figure'),
     Output('performance-table', 'children'),
     Output('alert-message', 'children', allow_duplicate=True),
     Output('alert-message', 'color', allow_duplicate=True),
     Output('alert-message', 'is_open', allow_duplicate=True)],
    Input('optimize-btn', 'n_clicks'),
    [State('portfolio-data-store', 'data'),
     State('optimization-objective', 'value'),
     State('target-return-input', 'value'),
     State('min-weight-input', 'value'),
     State('max-weight-input', 'value'),
     State('risk-free-rate-input', 'value')],
    prevent_initial_call=True
)
def optimize_portfolio(n_clicks, portfolio_data, objective, target_return,
                      min_weight, max_weight, risk_free_rate):
    """Optimize portfolio weights - ENHANCED ERROR HANDLING"""
    if not n_clicks or not portfolio_data:
        return None, {}, [], "", "primary", False

    try:
        # Convert data to numpy arrays with validation
        returns = np.asarray(portfolio_data['returns'], dtype=np.float64)
        cov_matrix = np.asarray(portfolio_data['cov_matrix'], dtype=np.float64)
        rf_rate = risk_free_rate / 100

        # Set up weight constraints as numpy arrays
        n_assets = len(returns)
        min_weights = np.full(n_assets, max(0.0, min_weight / 100))  # Ensure non-negative
        max_weights = np.full(n_assets, min(1.0, max_weight / 100))  # Ensure <= 1

        # Validate constraints feasibility
        if np.sum(min_weights) > 1.0:
            return None, {}, [], "Minimum weight constraints are not feasible (sum > 100%)", "danger", True
        
        if np.sum(max_weights) < 1.0:
            return None, {}, [], "Maximum weight constraints are not feasible (sum < 100%)", "danger", True

        start_time = time.time()

        if objective == 'sharpe':
            opt_result = optimizer.maximize_sharpe_ratio(returns, cov_matrix, rf_rate, min_weights, max_weights)
        elif objective == 'variance':
            opt_result = optimizer.minimize_variance(cov_matrix, min_weights, max_weights)
        elif objective == 'target':
            if target_return is None or target_return <= 0:
                return None, {}, [], "Target return must be positive", "danger", True
            opt_result = optimizer.optimize_for_return(returns, cov_matrix, target_return/100, min_weights, max_weights)
        else:
            return None, {}, [], "Unknown optimization objective", "danger", True

        opt_time = time.time() - start_time

        # Check if optimization converged
        if not opt_result.converged:
            error_msg = f"Optimization failed to converge. Try relaxing constraints or using different parameters."
            return None, {}, [], error_msg, "warning", True

        # Create weights chart
        weights_fig = go.Figure()

        # Current weights
        weights_fig.add_trace(go.Bar(
            x=portfolio_data['assets'],
            y=np.array(portfolio_data['weights']) * 100,
            name='Current Weights',
            marker_color=COLORS['primary'],
            opacity=0.7
        ))

        # Optimal weights
        weights_fig.add_trace(go.Bar(
            x=portfolio_data['assets'],
            y=np.array(opt_result.weights) * 100,
            name='Optimal Weights',
            marker_color=COLORS['success'],
            opacity=0.9
        ))

        weights_fig.update_layout(
            **DEFAULT_CHART_LAYOUT,
            title="Current vs Optimal Portfolio Weights",
            xaxis_title="Asset",
            yaxis_title="Weight (%)",
            barmode='group'
        )

        # Calculate current portfolio metrics for comparison
        current_return = np.dot(returns, portfolio_data['weights'])
        current_vol = np.sqrt(np.dot(portfolio_data['weights'], np.dot(cov_matrix, portfolio_data['weights'])))
        current_sharpe = (current_return - rf_rate) / current_vol if current_vol > 0 else 0.0

        # Create performance metrics table
        metrics_data = {
            'Metric': [
                'Expected Return (%)',
                'Volatility (%)',
                'Sharpe Ratio',
                'Optimization Time (s)'
            ],
            'Current Portfolio': [
                f"{current_return:.2%}",
                f"{current_vol:.2%}",
                f"{current_sharpe:.3f}",
                "N/A"
            ],
            'Optimal Portfolio': [
                f"{opt_result.expected_return:.2%}",
                f"{opt_result.volatility:.2%}",
                f"{opt_result.sharpe_ratio:.3f}",
                f"{opt_time:.3f}"
            ]
        }

        performance_table = dash_table.DataTable(
            data=pd.DataFrame(metrics_data).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in metrics_data.keys()],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': COLORS['primary'], 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': COLORS['light']}
        )

        success_msg = f"Portfolio optimization completed in {opt_time:.3f}s using {opt_result.method} solver"

        # Convert namedtuple to dict for storage
        opt_result_dict = {
            'weights': opt_result.weights.tolist(),
            'expected_return': opt_result.expected_return,
            'volatility': opt_result.volatility,
            'sharpe_ratio': opt_result.sharpe_ratio,
            'converged': opt_result.converged,
            'method': opt_result.method
        }

        return (opt_result_dict, weights_fig, performance_table, success_msg, "success", True)

    except Exception as e:
        error_msg = f"Error optimizing portfolio: {str(e)}"
        return None, {}, [], error_msg, "danger", True

@app.callback(
    [Output('frontier-results-store', 'data'),
     Output('efficient-frontier', 'figure'),
     Output('alert-message', 'children', allow_duplicate=True),
     Output('alert-message', 'color', allow_duplicate=True),
     Output('alert-message', 'is_open', allow_duplicate=True)],
    Input('frontier-btn', 'n_clicks'),
    [State('portfolio-data-store', 'data'),
     State('optimization-results-store', 'data'),
     State('min-weight-input', 'value'),
     State('max-weight-input', 'value'),
     State('risk-free-rate-input', 'value')],
    prevent_initial_call=True
)
def generate_efficient_frontier(n_clicks, portfolio_data, opt_results, min_weight, max_weight, risk_free_rate):
    """Generate and display efficient frontier - ENHANCED ERROR HANDLING"""
    if not n_clicks or not portfolio_data:
        return None, {}, "", "primary", False

    try:
        returns = np.asarray(portfolio_data['returns'], dtype=np.float64)
        cov_matrix = np.asarray(portfolio_data['cov_matrix'], dtype=np.float64)
        rf_rate = risk_free_rate / 100

        # Set up constraints
        n_assets = len(returns)
        min_weights = np.full(n_assets, max(0.0, min_weight / 100))
        max_weights = np.full(n_assets, min(1.0, max_weight / 100))

        # Generate efficient frontier
        start_time = time.time()
        frontier_result = optimizer.generate_efficient_frontier(
            returns, cov_matrix, 30, min_weights, max_weights  # Reduced points for reliability
        )
        frontier_time = time.time() - start_time

        # Check if frontier generation was successful
        if not frontier_result.converged or len(frontier_result.returns) == 0:
            return None, {}, "Efficient frontier generation failed. Try relaxing constraints.", "warning", True

        # Create efficient frontier plot
        frontier_fig = go.Figure()

        # Plot efficient frontier
        if len(frontier_result.volatilities) > 0:
            frontier_fig.add_trace(go.Scatter(
                x=frontier_result.volatilities,
                y=frontier_result.returns,
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=6)
            ))

        # Add current portfolio
        current_return = np.dot(returns, portfolio_data['weights'])
        current_vol = np.sqrt(np.dot(portfolio_data['weights'], np.dot(cov_matrix, portfolio_data['weights'])))
        
        frontier_fig.add_trace(go.Scatter(
            x=[current_vol],
            y=[current_return],
            mode='markers',
            name='Current Portfolio',
            marker=dict(size=12, color=COLORS['secondary'], symbol='star')
        ))

        # Add optimal portfolio if available
        if opt_results and opt_results.get('converged', False):
            frontier_fig.add_trace(go.Scatter(
                x=[opt_results['volatility']],
                y=[opt_results['expected_return']],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=12, color=COLORS['success'], symbol='diamond')
            ))

        frontier_fig.update_layout(
            **DEFAULT_CHART_LAYOUT,
            title="Efficient Frontier",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return"
        )

        success_msg = f"Efficient frontier generated in {frontier_time:.3f}s with {len(frontier_result.returns)} points"

        # Convert to serializable format
        frontier_data = {
            'returns': frontier_result.returns,
            'volatilities': frontier_result.volatilities,
            'sharpe_ratios': frontier_result.sharpe_ratios,
            'converged': frontier_result.converged
        }

        return (frontier_data, frontier_fig, success_msg, "success", True)

    except Exception as e:
        error_msg = f"Error generating efficient frontier: {str(e)}"
        return None, {}, error_msg, "danger", True

@app.callback(
    Output('benchmark-results', 'children'),
    Input('benchmark-btn', 'n_clicks'),
    prevent_initial_call=True
)
def run_benchmark(n_clicks):
    """Run performance benchmark"""
    if not n_clicks:
        return html.P("Click 'Run Benchmark' to test performance.")

    try:
        results = risk_manager.benchmark_performance(matrix_size=50, num_iterations=100)
        
        benchmark_data = []
        if 'python_time_ms' in results:
            benchmark_data.append({
                'Backend': 'Python',
                'Time (ms)': f"{results['python_time_ms']:.2f}",
                'Relative Speed': '1.0x'
            })
        
        if 'cpp_time_ms' in results:
            benchmark_data.append({
                'Backend': 'C++',
                'Time (ms)': f"{results['cpp_time_ms']:.2f}",
                'Relative Speed': f"{results.get('speedup', 1.0):.1f}x faster"
            })

        return dash_table.DataTable(
            data=benchmark_data,
            columns=[{'name': col, 'id': col} for col in ['Backend', 'Time (ms)', 'Relative Speed']],
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': COLORS['primary'], 'color': 'white'}
        )
    except Exception as e:
        return html.P(f"Benchmark error: {str(e)}", className="text-danger")

if __name__ == '__main__':
    app.run(debug=DEBUG_MODE, host=HOST, port=PORT)