from setuptools import setup, find_packages
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")

setup(
    name="pytorch-rocm-check",
    version="1.0.0",
    author="indrarg8899",
    description="PyTorch ROCm compatibility checker with auto-migration",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/indrarg8899/pytorch-rocm-check",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "jinja2>=3.1.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "ruff", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "rocm-check=src.checker:main",
            "rocm-migrate=src.migrator:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
