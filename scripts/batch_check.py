#!/usr/bin/env python3
"""CLI: Batch check multiple models or a project directory."""
import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checker import ROCmChecker, CheckResult
from src.report import ReportGenerator
from src.environment import detect_environment


def check_directory(checker: ROCmChecker, dir_path: str) -> list:
    results = []
    py_files = glob.glob(os.path.join(dir_path, "**/*.py"), recursive=True)

    for filepath in py_files:
        if "__pycache__" in filepath or "test_" in filepath:
            continue
        try:
            result = checker.check_file(filepath)
            results.append(result)
            status = "✅" if result.error_count == 0 else "⚠️" if result.error_count < 3 else "❌"
            print(f"  {status} {os.path.relpath(filepath, dir_path)} - Score: {result.calculate_score():.0f}")
        except Exception as e:
            print(f"  ❌ {os.path.relpath(filepath, dir_path)} - Error: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch check models/project for ROCm compatibility")
    parser.add_argument("--dir", "-d", required=True, help="Directory to scan")
    parser.add_argument("--models", nargs="+", help="List of model names to check")
    parser.add_argument("--rocm-version", default="5.7", help="Target ROCm version")
    parser.add_argument("--output", "-o", default="batch_report.html", help="Output report file")
    args = parser.parse_args()

    env = detect_environment()
    print(env.summary())
    print()

    checker = ROCmChecker(rocm_version=args.rocm_version)
    all_results = []

    if args.models:
        print(f"Checking {len(args.models)} models...")
        try:
            import torch
            for model_name in args.models:
                try:
                    model = torch.hub.load("pytorch/vision", model_name, pretrained=False)
                    result = checker.check_model(model)
                    all_results.append(result)
                    status = "✅" if result.error_count == 0 else "⚠️"
                    print(f"  {status} {model_name} - Score: {result.calculate_score():.0f}")
                except Exception as e:
                    print(f"  ❌ {model_name} - {e}")
        except ImportError:
            print("⚠️  PyTorch not installed — model loading skipped")

    if os.path.isdir(args.dir):
        print(f"\nScanning {args.dir}...")
        dir_results = check_directory(checker, args.dir)
        all_results.extend(dir_results)

    if all_results:
        gen = ReportGenerator()
        for r in all_results:
            gen.add_result(r)

        if args.output.endswith(".json"):
            gen.generate_json(args.output)
        else:
            gen.generate_html(args.output)
        print(f"\n{'='*50}")
        print(f"Report saved to: {args.output}")
        print(f"Total targets: {len(all_results)}")
        avg_score = sum(r.calculate_score() for r in all_results) / len(all_results)
        print(f"Average score: {avg_score:.1f}/100")
    else:
        print("No targets to check.")


if __name__ == "__main__":
    main()
