"""Tests for the ROCm compatibility checker."""
import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checker import ROCmChecker, CheckResult, Issue
from src.ops import get_op_compat, is_cuda_specific, get_stats, CompatStatus
from src.models import get_model_compat
from src.migrator import CUDAToROCmMigrator
from src.report import ReportGenerator
from src.environment import detect_environment
from src.utils import format_size, check_rocm_minimum, version_tuple


class TestROCmChecker:
    def setup_method(self):
        self.checker = ROCmChecker(rocm_version="5.7")

    def test_check_code_string_clean(self):
        code = "import torch\nx = torch.randn(10, 10)\ny = torch.relu(x)"
        result = self.checker.check_code_string(code)
        assert result.score >= 0
        assert len(result.issues) == 0

    def test_check_code_string_cuda(self):
        code = "torch.backends.cudnn.enabled = True"
        result = self.checker.check_code_string(code)
        assert len(result.issues) > 0

    def test_check_file(self):
        code = "import torch\ntorch.backends.cudnn.enabled = True\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = self.checker.check_file(f.name)
            os.unlink(f.name)
        assert result.target.endswith(".py")

    def test_result_summary(self):
        result = CheckResult(target="test", compatible_ops=10, incompatible_ops=2)
        result.issues.append(Issue("test.py", 1, "error", "test", "error message"))
        summary = result.summary()
        assert "10" in summary
        assert "error message" in summary


class TestOps:
    def test_get_op_compat(self):
        compat = get_op_compat("torch.add")
        assert compat is not None
        assert compat.status == CompatStatus.FULL

    def test_get_incompatible(self):
        from src.ops import get_incompatible_ops
        ops = get_incompatible_ops()
        assert len(ops) > 0

    def test_is_cuda_specific(self):
        assert is_cuda_specific("torch.backends.cudnn.enabled")
        assert is_cuda_specific("CUDAExtension")
        assert not is_cuda_specific("x = torch.randn(10)")

    def test_get_stats(self):
        stats = get_stats()
        assert "full" in stats
        assert "none" in stats
        assert stats["full"] + stats["partial"] + stats["none"] > 0


class TestModels:
    def test_get_model_compat(self):
        model = get_model_compat("resnet50")
        assert model.compatible is True

    def test_get_incompatible(self):
        from src.models import get_incompatible_models
        models = get_incompatible_models()
        assert len(models) > 0


class TestMigrator:
    def test_migrate_content(self):
        migrator = CUDAToROCmMigrator()
        code = "torch.backends.cudnn.enabled = True"
        result = migrator.migrate_content(code)
        assert "MIOpen" in result

    def test_migrate_report(self):
        migrator = CUDAToROCmMigrator()
        migrator.migrate_content("torch.backends.cudnn.enabled = True")
        report = migrator.get_report()
        assert "Migration Report" in report

    def test_migrate_file(self):
        migrator = CUDAToROCmMigrator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as inf:
            inf.write("torch.backends.cudnn.enabled = True")
            inf.flush()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as outf:
                outf.close()
                migrator.migrate_file(inf.name, outf.name)
                with open(outf.name) as f:
                    content = f.read()
                assert "MIOpen" in content
                os.unlink(inf.name)
                os.unlink(outf.name)


class TestReport:
    def test_generate_html(self):
        gen = ReportGenerator()
        result = CheckResult(target="test", compatible_ops=5, incompatible_ops=1)
        gen.add_result(result)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            gen.generate_html(f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_generate_json(self):
        gen = ReportGenerator()
        result = CheckResult(target="test", compatible_ops=5, incompatible_ops=1)
        gen.add_result(result)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            gen.generate_json(f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestUtils:
    def test_format_size(self):
        assert format_size(1024) == "1.0KB"
        assert format_size(1024 * 1024) == "1.0MB"

    def test_version_tuple(self):
        assert version_tuple("5.7") == (5, 7)
        assert version_tuple("6.0.1") == (6, 0, 1)

    def test_check_rocm_minimum(self):
        assert check_rocm_minimum("5.7", "5.5") is True
        assert check_rocm_minimum("5.5", "5.7") is False
        assert check_rocm_minimum("6.0", "5.7") is True


class TestEnvironment:
    def test_detect_environment(self):
        env = detect_environment()
        assert env.python_version
        assert env.os_info
        assert env.rocm_version
        summary = env.summary()
        assert "ROCm" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
