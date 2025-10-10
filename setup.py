"""
Setup configuration for Quant Risk Optimizer

This file configures the package for installation with pip, including
C++ extension compilation using pybind11 and CMake.
"""

import os
import sys
import subprocess
from pathlib import Path
import platform

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, find_packages

# Package metadata
PACKAGE_NAME = "quant-risk-optimiser"
VERSION = "1.0.0"
AUTHOR = "Quantitative Finance Team"
AUTHOR_EMAIL = "quant@example.com"
DESCRIPTION = "High-Performance Portfolio Risk Management & Optimization"
LONG_DESCRIPTION = """
Quant Risk Optimizer

A professional-grade portfolio risk management and optimization platform that combines
Python's ease of use with C++'s computational performance. Built for quantitative analysts, portfolio managers, and financial engineers.
"""

URL = "https://github.com/quantrisk/quant-risk-optimiser"

def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            requirements = [
                line.strip() for line in f 
                if line.strip() and not line.startswith('#')
            ]
        return requirements
    return []

# Development, docs, and benchmark requirements
dev_requirements = [
    'pytest>=6.0',
    'pytest-cov>=2.0',
    'black>=21.0',
    'flake8>=3.8',
    'mypy>=0.800',
    'pre-commit>=2.0',
]
doc_requirements = [
    'sphinx>=4.0',
    'sphinx-rtd-theme>=1.0',
    'myst-parser>=0.15',
]
benchmark_requirements = [
    'memory-profiler>=0.60',
    'psutil>=5.8',
    'py-spy>=0.3',
]

# Platform-specific compiler options
def get_compiler_options():
    opts = ["-O3", "-fvisibility=hidden", "-g0", "-DNDEBUG"]
    link_opts = []

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        # macOS
        if machine == "arm64":
            opts.append("-mcpu=apple-m1")
        # No -march=native for clang on macOS
    elif machine in ["x86_64", "amd64"]:
        opts.append("-march=native")
        opts.append("-mtune=native")
    return opts, link_opts

extra_compile_args, extra_link_args = get_compiler_options()

ext_modules = [
    Pybind11Extension(
        "quant_risk_core",
        sources=[
            "backend/risk.cpp",
            "backend/optimize.cpp", 
            "backend/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "backend",
            "/usr/include/eigen3",
            "/usr/local/include/eigen3",
            "/opt/homebrew/include/eigen3",  # Homebrew (Apple Silicon)
        ],
        language='c++',
        cxx_std=17,
        define_macros=[("VERSION_INFO", f'"{VERSION}"')],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

class CustomBuildExt(build_ext):
    """Custom build extension to handle dependencies and compilation flags"""

    def build_extensions(self):
        # No special handling; all handled above
        super().build_extensions()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(where="frontend"),
    package_dir={"": "frontend"},
    package_data={"": ["*.csv", "*.json", "*.yaml", "*.yml"]},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": dev_requirements,
        "docs": doc_requirements,
        "benchmark": benchmark_requirements,
        "all": dev_requirements + doc_requirements + benchmark_requirements,
    },
    entry_points={
        "console_scripts": [
            "quant-risk-optimizer=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="finance portfolio optimization risk management quantitative analysis",
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}/wiki",
    },
)
