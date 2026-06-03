# PyTorch ROCm Compatibility Checker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ROCm](https://img.shields.io/badge/ROCm-5.0+-d4232a.svg)](https://rocm.docs.amd.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Comprehensive tool for checking PyTorch model and operator compatibility with AMD ROCm. Includes auto-migration from CUDA to ROCm, detailed reporting, and known issue tracking.

## Features

- **Compatibility Scanning** — Check any PyTorch model for ROCm compatibility
- **Operator Matrix** — Full coverage of 200+ PyTorch ops with ROCm status
- **Auto-Migration** — Convert CUDA-specific code to ROCm-compatible equivalents
- **HTML/JSON Reports** — Rich reports with compatibility scores and recommendations
- **Model Database** — Pre-mapped compatibility for 50+ popular architectures
- **Environment Detection** — Auto-detect ROCm version, GPU, and driver info
- **Batch Processing** — Check entire projects or model zoos at once
- **Known Issues Tracker** — Curated list of ROCm-specific bugs and workarounds

## Quick Start

```bash
pip install pytorch-rocm-check
```

### Check a Model

```python
from src.checker import ROCmChecker
from src.models import load_model

model = load_model("resnet50")
checker = ROCmChecker()
result = checker.check_model(model)
print(result.summary())
```

### CLI Usage

```bash
# Check a single model
python scripts/check_model.py --model resnet50 --rocm-version 5.7

# Batch check a directory
python scripts/batch_check.py --dir ./models/ --output report.html

# Generate migration suggestions
python -m src.migrator --input model.py --output model_rocm.py

# Scan environment
python -m src.environment
```

### Auto-Migration

```python
from src.migrator import CUDAToROCmMigrator

migrator = CUDAToROCmMigrator()
migrator.migrate_file("my_model.py", "my_model_rocm.py")
print(migrator.get_report())
```

### Generate Reports

```python
from src.report import ReportGenerator

gen = ReportGenerator()
gen.add_result(checker_result)
gen.generate_html("compatibility_report.html")
gen.generate_json("compatibility_report.json")
```

## Compatibility Matrix

| Category | Total Ops | Compatible | Partial | Incompatible |
|----------|-----------|------------|---------|-------------|
| Tensor Ops | 45 | 42 | 2 | 1 |
| NN Ops | 38 | 35 | 2 | 1 |
| CUDA-specific | 22 | 12 | 6 | 4 |
| Autograd | 15 | 14 | 1 | 0 |
| Distributed | 18 | 14 | 3 | 1 |

See [docs/compatibility_matrix.md](docs/compatibility_matrix.md) for full details.

## Supported ROCm Versions

- ROCm 5.7+ (recommended)
- ROCm 5.6 (partial)
- ROCm 5.5 (limited, known issues)
- ROCm 6.0+ (latest, best support)

## Project Structure

```
pytorch-rocm-check/
├── src/
│   ├── checker.py          # Main compatibility checker
│   ├── ops.py              # Operator compatibility matrix
│   ├── models.py           # Model compatibility database
│   ├── models_database.py  # Known compatible/incompatible models
│   ├── migrator.py         # CUDA→ROCm auto-migration
│   ├── report.py           # HTML/JSON report generator
│   ├── environment.py      # ROCm environment detection
│   └── utils.py            # Shared utilities
├── configs/
│   └── default.yml         # Default configuration
├── tests/
│   └── test_checker.py     # Test suite
├── docs/
│   ├── migration_guide.md  # CUDA→ROCm migration guide
│   ├── compatibility_matrix.md
│   └── known_issues.md     # Known ROCm issues
├── scripts/
│   ├── check_model.py      # CLI: check single model
│   └── batch_check.py      # CLI: batch check
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
└── setup.py
```

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/xyz`)
3. Commit changes (`git commit -m 'Add xyz'`)
4. Push (`git push origin feature/xyz`)
5. Open PR

## License

MIT License — see [LICENSE](LICENSE)

## Acknowledgments

- AMD ROCm team for documentation
- PyTorch community for operator specifications
- Contributors who tested on real AMD hardware
