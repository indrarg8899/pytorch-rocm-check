"""Tests for migration scanner."""
import unittest
from src.migrator import MigrationScanner, MigrationReport


class TestMigrationScannerDetailed(unittest.TestCase):
    """Detailed tests for the migration scanner."""

    def test_scan_empty_project(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = MigrationScanner(tmpdir)
            report = scanner.scan()
            self.assertEqual(len(report.issues), 0)
            self.assertEqual(report.files_scanned, 0)

    def test_skip_git_directory(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = os.path.join(tmpdir, ".git")
            os.makedirs(git_dir)
            with open(os.path.join(git_dir, "config"), "w") as f:
                f.write("cudaMalloc test")

            scanner = MigrationScanner(tmpdir)
            report = scanner.scan()
            self.assertEqual(len(report.issues), 0)

    def test_multiple_files(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.cu", "b.cpp", "c.py"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("cudaMalloc(&ptr, size);\n")

            scanner = MigrationScanner(tmpdir)
            report = scanner.scan()
            self.assertGreaterEqual(report.files_scanned, 3)

    def test_report_summary(self):
        from src.migrator import MigrationReport
        report = MigrationReport(project_path="/tmp/test")
        report.files_scanned = 10
        self.assertEqual(report.error_count, 0)


if __name__ == "__main__":
    unittest.main()
