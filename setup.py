[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pytorch-rocm-check"
version = "1.0.0"
description = "PyTorch ROCm compatibility checker and CUDA→ROCm migration tool"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
authors = [{name = "indrarg8899"}]
keywords = ["pytorch", "rocm", "amd", "hip", "migration", "cuda"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.scripts]
rocm-check = "src.checker:main"
rocm-migrate = "src.migrator:main"
rocm-env-validate = "src.validator:main"
rocm-matrix = "src.matrix:main"

[project.optional-dependencies]
dev = ["pytest", "flake8", "mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
