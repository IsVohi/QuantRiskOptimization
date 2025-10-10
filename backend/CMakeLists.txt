cmake_minimum_required(VERSION 3.15)
project(quant_risk_optimiser VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Check if we're building with MSVC
if(MSVC)
    # MSVC specific flags
    add_compile_options(/W4 /wd4267 /wd4244)
    add_definitions(-D_USE_MATH_DEFINES)
else()
    # GCC/Clang flags
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    
    # Enable OpenMP if available (for potential parallelization)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(SOURCES
    risk.cpp
    optimize.cpp
    bindings.cpp
)

# Create the Python module
pybind11_add_module(quant_risk_core ${SOURCES})

# Compiler-specific properties for pybind11 module
target_compile_definitions(quant_risk_core PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Link Eigen3 (header-only library)
target_link_libraries(quant_risk_core PRIVATE Eigen3::Eigen)

# Set properties for the module
set_target_properties(quant_risk_core PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

# Strip binary in release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        pybind11_strip(quant_risk_core)
    endif()
endif()

# Platform-specific optimizations
if(APPLE)
    # macOS specific optimizations
    set_target_properties(quant_risk_core PROPERTIES
        MACOSX_RPATH TRUE
        INSTALL_NAME_DIR "@rpath"
    )
elseif(WIN32)
    # Windows specific settings
    if(MSVC)
        target_compile_options(quant_risk_core PRIVATE /bigobj)
    endif()
endif()

# Optional: Enable IPO/LTO if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET quant_risk_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled for Release build")
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Python: ${Python_VERSION}")
message(STATUS "pybind11: ${pybind11_VERSION}")

# Installation (optional)
install(TARGETS quant_risk_core 
        COMPONENT python
        LIBRARY DESTINATION ${Python_SITEARCH})

# Testing support
enable_testing()

# Custom target for development
add_custom_target(dev_build
    COMMAND ${CMAKE_COMMAND} --build . --config Release
    COMMENT "Building for development"
)

# Custom target for benchmarking
add_custom_target(benchmark
    COMMAND python -c "import quant_risk_core; print('Benchmarking...'); print(f'Time: {quant_risk_core.benchmark_cpp_vs_python(200, 50)} ms')"
    DEPENDS quant_risk_core
    COMMENT "Running performance benchmark"
)

# Add alias for easier referencing
add_library(QuantRisk::Core ALIAS quant_risk_core)
