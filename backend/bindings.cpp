#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "risk.h"
#include "optimize.h"

namespace py = pybind11;

PYBIND11_MODULE(quant_risk_core, m) {
    m.doc() = "High-performance quantitative risk management and portfolio optimization";
    
    // =====================================================
    // Risk Calculation Classes and Functions
    // =====================================================
    
    py::class_<QuantRisk::RiskCalculator>(m, "RiskCalculator")
        .def(py::init<unsigned int>(), 
             "Constructor with random seed for reproducible results",
             py::arg("seed") = 12345)
        
        .def("calculate_var", &QuantRisk::RiskCalculator::calculateVaR,
             "Calculate portfolio Value at Risk using Monte Carlo simulation",
             py::arg("returns"), py::arg("cov_matrix"), py::arg("weights"),
             py::arg("confidence_level") = 0.95,
             py::arg("num_simulations") = 10000,
             py::arg("time_horizon") = 1,
             py::call_guard<py::gil_scoped_release>())  // Release GIL during computation
        
        .def("calculate_es", &QuantRisk::RiskCalculator::calculateES,
             "Calculate Expected Shortfall (Conditional VaR)",
             py::arg("returns"), py::arg("cov_matrix"), py::arg("weights"),
             py::arg("confidence_level") = 0.95,
             py::arg("num_simulations") = 10000,
             py::arg("time_horizon") = 1,
             py::call_guard<py::gil_scoped_release>())
        
        .def("calculate_volatility", &QuantRisk::RiskCalculator::calculateVolatility,
             "Calculate portfolio volatility (annualized)",
             py::arg("cov_matrix"), py::arg("weights"),
             py::call_guard<py::gil_scoped_release>())
        
        .def("generate_simulations", &QuantRisk::RiskCalculator::generateSimulations,
             "Generate Monte Carlo simulation paths",
             py::arg("returns"), py::arg("cov_matrix"), py::arg("weights"),
             py::arg("num_simulations"), py::arg("time_horizon"),
             py::call_guard<py::gil_scoped_release>())
        
        .def("calculate_max_drawdown", &QuantRisk::RiskCalculator::calculateMaxDrawdown,
             "Calculate maximum drawdown from return series",
             py::arg("returns"))
        
        .def("calculate_risk_contributions", &QuantRisk::RiskCalculator::calculateRiskContributions,
             "Decompose portfolio risk by asset contribution",
             py::arg("cov_matrix"), py::arg("weights"),
             py::call_guard<py::gil_scoped_release>());
    
    // =====================================================
    // Portfolio Optimization Classes and Functions
    // =====================================================
    
    py::class_<QuantRisk::OptimizationResult>(m, "OptimizationResult")
        .def(py::init<>())
        .def_readwrite("weights", &QuantRisk::OptimizationResult::weights)
        .def_readwrite("expected_return", &QuantRisk::OptimizationResult::expected_return)
        .def_readwrite("volatility", &QuantRisk::OptimizationResult::volatility)
        .def_readwrite("sharpe_ratio", &QuantRisk::OptimizationResult::sharpe_ratio)
        .def_readwrite("converged", &QuantRisk::OptimizationResult::converged)
        .def_readwrite("iterations", &QuantRisk::OptimizationResult::iterations)
        .def_readwrite("objective_value", &QuantRisk::OptimizationResult::objective_value);
    
    py::class_<QuantRisk::FrontierPoint>(m, "FrontierPoint")
        .def(py::init<>())
        .def_readwrite("expected_return", &QuantRisk::FrontierPoint::expected_return)
        .def_readwrite("volatility", &QuantRisk::FrontierPoint::volatility)
        .def_readwrite("sharpe_ratio", &QuantRisk::FrontierPoint::sharpe_ratio)
        .def_readwrite("weights", &QuantRisk::FrontierPoint::weights);
    
    py::class_<QuantRisk::PortfolioOptimizer>(m, "PortfolioOptimizer")
        .def(py::init<double, int>(), 
             "Constructor with optimization parameters",
             py::arg("tolerance") = 1e-8, py::arg("max_iterations") = 1000)
        
        .def("maximize_sharpe_ratio", &QuantRisk::PortfolioOptimizer::maximizeSharpeRatio,
             "Find maximum Sharpe ratio portfolio",
             py::arg("returns"), py::arg("cov_matrix"),
             py::arg("risk_free_rate") = 0.02,
             py::arg("min_weights") = Eigen::VectorXd(),
             py::arg("max_weights") = Eigen::VectorXd(),
             py::call_guard<py::gil_scoped_release>())
        
        .def("minimize_variance", &QuantRisk::PortfolioOptimizer::minimizeVariance,
             "Find minimum variance portfolio",
             py::arg("cov_matrix"),
             py::arg("min_weights") = Eigen::VectorXd(),
             py::arg("max_weights") = Eigen::VectorXd(),
             py::call_guard<py::gil_scoped_release>())
        
        .def("optimize_for_return", &QuantRisk::PortfolioOptimizer::optimizeForReturn,
             "Optimize portfolio for target return",
             py::arg("returns"), py::arg("cov_matrix"), py::arg("target_return"),
             py::arg("min_weights") = Eigen::VectorXd(),
             py::arg("max_weights") = Eigen::VectorXd(),
             py::call_guard<py::gil_scoped_release>())
        
        .def("generate_efficient_frontier", &QuantRisk::PortfolioOptimizer::generateEfficientFrontier,
             "Generate efficient frontier points",
             py::arg("returns"), py::arg("cov_matrix"),
             py::arg("num_points") = 50,
             py::arg("min_weights") = Eigen::VectorXd(),
             py::arg("max_weights") = Eigen::VectorXd(),
             py::call_guard<py::gil_scoped_release>())
        
        .def("calculate_portfolio_metrics", &QuantRisk::PortfolioOptimizer::calculatePortfolioMetrics,
             "Calculate portfolio performance metrics",
             py::arg("returns"), py::arg("cov_matrix"), py::arg("weights"),
             py::arg("risk_free_rate") = 0.02)
        
        .def("validate_weights", &QuantRisk::PortfolioOptimizer::validateWeights,
             "Validate portfolio weights",
             py::arg("weights"),
             py::arg("min_weights") = Eigen::VectorXd(),
             py::arg("max_weights") = Eigen::VectorXd());
    
    // =====================================================
    // Utility Functions
    // =====================================================
    
    m.def("version", []() {
        return "1.0.0";
    }, "Get library version");
    
    m.def("benchmark_cpp_vs_python", [](int matrix_size, int num_iterations) {
        // Simple benchmark function for comparing C++ vs Python performance
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create random matrices and perform operations
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(matrix_size, matrix_size);
        Eigen::MatrixXd B = Eigen::MatrixXd::Random(matrix_size, matrix_size);
        
        for (int i = 0; i < num_iterations; ++i) {
            Eigen::MatrixXd C = A * B;
            A = C.cwiseAbs();  // Element-wise absolute value
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / 1000.0;  // Return milliseconds
    }, "Benchmark C++ matrix operations",
    py::arg("matrix_size") = 100, py::arg("num_iterations") = 100,
    py::call_guard<py::gil_scoped_release>());
    
    // Add necessary includes for timing
    m.attr("__version__") = "1.0.0";
}

// Additional helper functions for Python interoperability
namespace {

// Convert Python lists to Eigen vectors
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> py_list_to_eigen_vector(const py::list& py_list) {
    std::vector<T> std_vec = py_list.cast<std::vector<T>>();
    Eigen::Matrix<T, Eigen::Dynamic, 1> eigen_vec(std_vec.size());
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec(i) = std_vec[i];
    }
    return eigen_vec;
}

// Convert Python nested lists to Eigen matrices
template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> py_nested_list_to_eigen_matrix(
    const py::list& py_nested_list) {
    
    if (py_nested_list.empty()) {
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();
    }
    
    size_t rows = py_nested_list.size();
    size_t cols = py_nested_list[0].cast<py::list>().size();
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        py::list row = py_nested_list[i].cast<py::list>();
        for (size_t j = 0; j < cols; ++j) {
            eigen_mat(i, j) = row[j].cast<T>();
        }
    }
    
    return eigen_mat;
}

}  // anonymous namespace
