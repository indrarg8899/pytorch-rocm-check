# ⚡ PyTorch ROCm Check

![PyPI](https://img.shields.io/badge/PyTorch-2.x+-ee4c2c?logo=pytorch&logoColor=white)
![ROCm](https://img.shields.io/badge/ROCm-5.x+-ed1c24?logo=amd&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://img.shields.io/github/actions/workflow/status/indrarg8899/pytorch-rocm-check/ci.yml?branch=main&label=CI)
![Stars](https://img.shields.io/github/stars/indrarg8899/pytorch-rocm-check?style=social)

> Automated PyTorch ↔ ROCm compatibility checker and CUDA-to-ROCm migration toolkit for AMD Instinct GPUs.

## 🚀 Features

- **Compatibility Matrix Lookup** — Instantly check if your PyTorch version supports your ROCm version
- **CUDA→ROCm Migration Scanner** — Scan existing projects for CUDA-specific code and generate ROCm equivalents
- **Environment Validator** — Verify GPU drivers, ROCm toolkit, and PyTorch build are correctly configured
- **Migration Report Generator** — Produce detailed PDF/HTML migration reports with actionable recommendations
- **CI/CD Integration** — Drop-in GitHub Actions and Jenkins pipeline stages
- **Auto-Fix Suggestions** — Concrete code patches for common CUDA→HIP porting patterns
- **Multi-GPU Support** — Validates configurations across AMD MI50, MI100, MI210, MI250X, MI300X

## 📦 Installation

```bash
pip install pytorch-rocm-check
```

Or from source:

```bash
git clone https://github.com/indrarg8899/pytorch-rocm-check.git
cd pytorch-rocm-check
pip install -e ".[dev]"
```

## 🔧 Usage

### Quick Compatibility Check

```bash
# Check current environment
rocm-check

# Check specific versions
rocm-check --pytorch 2.1.0 --rocm 5.7

# Verbose output with full matrix
rocm-check --verbose --matrix
```

### CUDA→ROCm Migration

```bash
# Scan a project directory
rocm-migrate scan /path/to/project

# Generate migration plan
rocm-migrate plan /path/to/project --output migration-plan.json

# Apply automated fixes
rocm-migrate apply /path/to/project --dry-run
rocm-migrate apply /path/to/project --confirm
```

### Environment Validation

```bash
# Full system validation
rocm-env-validate

# Export report
rocm-env-validate --export report.html
```

### Python API

```python
from pytorch_rocm_check import ROCmChecker, MigrationScanner

# Check compatibility
checker = ROCmChecker()
result = checker.check(pytorch_version="2.1.0", rocm_version="5.7")
print(result.is_compatible)  # True/False
print(result.recommendations)

# Scan for CUDA code
scanner = MigrationScanner()
report = scanner.scan("/path/to/project")
for issue in report.issues:
    print(f"{issue.file}:{issue.line} — {issue.description}")
```

## 📊 Compatibility Matrix

| PyTorch | ROCm 5.6 | ROCm 5.7 | ROCm 6.0 | ROCm 6.1 |
|---------|----------|----------|----------|----------|
| 2.0.x   | ✅       | ✅       | ⚠️       | ❌       |
| 2.1.x   | ✅       | ✅       | ✅       | ⚠️       |
| 2.2.x   | ❌       | ✅       | ✅       | ✅       |
| 2.3.x   | ❌       | ⚠️       | ✅       | ✅       |
| 2.4.x   | ❌       | ❌       | ✅       | ✅       |

*✅ = Full support, ⚠️ = Partial/supportable, ❌ = Not supported*

## 🏗️ Architecture

```
pytorch-rocm-check/
├── src/
│   ├── checker.py          # Core compatibility checking engine
│   ├── migrator.py         # CUDA→ROCm migration scanner
│   ├── validator.py        # Environment validation
│   ├── matrix.py           # Version compatibility matrix
│   └── report.py           # Report generation
├── tests/
│   ├── test_checker.py
│   ├── test_migrator.py
│   └── test_validator.py
├── docs/
│   └── MIGRATION_GUIDE.md
├── .github/workflows/ci.yml
├── LICENSE
└── setup.py
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [AMD ROCm](https://rocm.docs.amd.com/) documentation
- [PyTorch](https://pytorch.org/) community
- AMD Developer Cloud program
