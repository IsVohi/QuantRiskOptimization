# Create a system architecture diagram for Quant Risk Optimizer
diagram_code = """
graph TB
    subgraph "Frontend Layer"
        WD["Web Dashboard<br/>Dash/Plotly<br/>━━━━━━━━━━━━━━━━<br/>• Interactive UI<br/>• File Upload<br/>• Visualizations<br/>• Export Tools"]
    end
    
    subgraph "Middleware Layer"
        API["Python API<br/>NumPy/SciPy<br/>━━━━━━━━━━━━━━━━<br/>• Risk Management<br/>• Portfolio Optimization<br/>• Data Validation<br/>• Error Handling"]
    end
    
    subgraph "Integration Layer"
        PB["pybind11<br/>C++/Python Bridge<br/>━━━━━━━━━━━━━━━━<br/>• GIL Release<br/>• Type Conversion<br/>• Error Propagation"]
    end
    
    subgraph "Backend Layer"
        CORE["C++ Core<br/>Eigen/Modern C++17<br/>━━━━━━━━━━━━━━━━<br/>• Monte Carlo<br/>• Linear Algebra<br/>• Matrix Operations<br/>• Optimization Algorithms"]
    end
    
    %% Data flow arrows
    WD -->|User Requests| API
    API -->|Results & Data| WD
    API -->|Function Calls| PB
    PB -->|Native Data| API
    PB -->|Optimized Calls| CORE
    CORE -->|Computed Results| PB
    
    %% Styling
    classDef frontend fill:#B3E5EC,stroke:#1FB8CD,stroke-width:2px,color:#000
    classDef middleware fill:#A5D6A7,stroke:#2E8B57,stroke-width:2px,color:#000
    classDef integration fill:#FFEB8A,stroke:#D2BA4C,stroke-width:2px,color:#000
    classDef backend fill:#FFCDD2,stroke:#DB4545,stroke-width:2px,color:#000
    
    class WD frontend
    class API middleware
    class PB integration
    class CORE backend
"""

# Create the mermaid diagram and save as both PNG and SVG
png_path, svg_path = create_mermaid_diagram(
    diagram_code, 
    png_filepath='architecture_diagram.png',
    svg_filepath='architecture_diagram.svg',
    width=1200,
    height=900
)

print(f"Architecture diagram saved as PNG: {png_path}")
print(f"Architecture diagram saved as SVG: {svg_path}")