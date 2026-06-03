"""Tests for PyTorch ROCm Checker."""
import unittest
from src.checker import ROCmChecker, CompatLevel


class TestROCmChecker(unittest.TestCase):
    """Test the compatibility checker."""

    def test_known_compatible(self):
        checker = ROCmChecker()
        result = checker.check(pytorch_version="2.1.0", rocm_version="5.7")
        self.assertTrue(result.is_compatible)
        self.assertEqual(result.level, CompatLevel.FULL)

    def test_known_incompatible(self):
        checker = ROCmChecker()
        result = checker.check(pytorch_version="2.0.0", rocm_version="6.0")
        self.assertEqual(result.level, CompatLevel.PARTIAL)

    def test_unsupported_combo(self):
        checker = ROCmChecker()
        result = checker.check(pytorch_version="1.13.0", rocm_version="6.1")
        self.assertEqual(result.level, CompatLevel.UNSUPPORTED)

    def test_short_version(self):
        checker = ROCmChecker()
        self.assertEqual(checker._short_version("2.1.0"), "2.1")
        self.assertEqual(checker._short_version("5.7.1"), "5.7")

    def test_recommendations_on_unsupported(self):
        checker = ROCmChecker()
        result = checker.check(pytorch_version="1.12.0", rocm_version="6.1")
        self.assertTrue(len(result.recommendations) > 0)

    def test_mi300x_detection(self):
        checker = ROCmChecker()
        gpu = checker._match_gpu("Instinct MI300X")
        self.assertEqual(gpu, "MI300X")


class TestMigrationScanner(unittest.TestCase):
    """Test the migration scanner."""

    def test_cpp_cuda_api_detection(self):
        from src.migrator import MigrationScanner
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_file = os.path.join(tmpdir, "kernel.cu")
            with open(cuda_file, "w") as f:
                f.write('#include <cuda_runtime.h>\n')
                f.write('cudaMalloc(&ptr, size);\n')
                f.write('cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);\n')

            scanner = MigrationScanner(tmpdir)
            report = scanner.scan()
            self.assertGreater(len(report.issues), 0)
            self.assertGreater(report.summary["errors"], 0)

    def test_python_cuda_detection(self):
        from src.migrator import MigrationScanner
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "train.py")
            with open(py_file, "w") as f:
                f.write('model = model.cuda()\n')
                f.write('if torch.cuda.is_available():\n')
                f.write('    torch.cuda.amp.autocast()\n')

            scanner = MigrationScanner(tmpdir)
            report = scanner.scan()
            self.assertGreater(len(report.issues), 0)


class TestCompatibilityMatrix(unittest.TestCase):
    """Test the compatibility matrix."""

    def test_query_by_gpu(self):
        from src.matrix import CompatibilityMatrix
        matrix = CompatibilityMatrix()
        results = matrix.query(gpu="MI300X")
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.gpu, "MI300X")

    def test_best_combo(self):
        from src.matrix import CompatibilityMatrix
        matrix = CompatibilityMatrix()
        best = matrix.get_best_combo("MI300X")
        self.assertIsNotNone(best)
        self.assertEqual(best.gpu, "MI300X")


if __name__ == "__main__":
    unittest.main()
