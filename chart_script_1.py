import plotly.express as px
import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {
    "operations": ["Monte Carlo VaR (10K sims)", "Mean-Variance Optimization", "Efficient Frontier (50 pts)", "Risk Decomposition"],
    "speedup_factors": [12.0, 11.9, 10.0, 9.0],
    "cpp_times": [15, 8, 120, 5],
    "python_times": [180, 95, 1200, 45]
}

# Abbreviate operation names to fit 15 character limit
operations_abbrev = [
    "MC VaR (10K)",
    "Mean-Var Opt", 
    "Eff Frontier",
    "Risk Decomp"
]

# Create horizontal bar chart
fig = go.Figure()

# Add bars with gradient colors from blue to green
fig.add_trace(go.Bar(
    y=operations_abbrev,
    x=data["speedup_factors"],
    orientation='h',
    marker=dict(
        color=data["speedup_factors"],
        colorscale=[[0, '#5D878F'], [1, '#2E8B57']],  # Blue to green gradient
        showscale=False
    ),
    text=[f"{x}x" for x in data["speedup_factors"]],
    textposition='inside',
    textfont=dict(color='white', size=12)
))

# Update layout
fig.update_layout(
    title="C++ vs Python Performance Speedup",
    xaxis_title="Speedup Factor",
    yaxis_title="Operation"
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(showgrid=False)

# Update traces for better appearance
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("speedup_chart.png")
fig.write_image("speedup_chart.svg", format="svg")

print("Chart saved successfully!")