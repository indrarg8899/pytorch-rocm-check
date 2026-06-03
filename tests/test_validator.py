"""Tests for environment validator."""
import unittest
from src.validator import EnvironmentValidator


class TestEnvironmentValidator(unittest.TestCase):
    """Test the environment validator."""

    def test_validator_runs(self):
        validator = EnvironmentValidator(verbose=False)
        report = validator.validate()
        self.assertGreater(len(report.checks), 0)

    def test_report_counts(self):
        validator = EnvironmentValidator()
        report = validator.validate()
        total = report.passed_count + report.failed_count
        self.assertEqual(total, len(report.checks))

    def test_html_export(self):
        import tempfile, os
        validator = EnvironmentValidator()
        report = validator.validate()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            validator.export_html(f.name)
            content = open(f.name).read()
            self.assertIn("ROCm", content)
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
