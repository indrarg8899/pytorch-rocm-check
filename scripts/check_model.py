#!/usr/bin/env python3
"""CLI: Check a single model for ROCm compatibility."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checker import ROCmChecker
from src.models import get_model_compat
from src.environment import detect_environment


def main():
    parser = argparse.ArgumentParser(description="Check PyTorch model ROCm compatibility")
    parser.add_argument("--model", required=True, help="Model name (e.g., resnet50)")
    parser.add_argument("--rocm-version", default="5.7", help="Target ROCm version")
    parser.add_argument("--output", "-o", help="Save report to file (HTML/JSON)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Checking model: {args.model}")
    print(f"Target ROCm: {args.rocm_version}")
    print()

    env = detect_environment()
    if args.verbose:
        print(env.summary())
        print()

    compat = get_model_compat(args.model)
    if compat.compatible:
        print(f"✅ {compat.name} is compatible with ROCm >= {compat.min_rocm}")
        if compat.workarounds:
            print(f"   Workarounds needed: {', '.join(compat.workarounds)}")
        if compat.notes:
            print(f"   Notes: {compat.notes}")
    else:
        print(f"❌ {compat.name} is NOT compatible with ROCm")
        if compat.notes:
            print(f"   Reason: {compat.notes}")

    try:
        import torch
        print(f"\nLoading model...")
        if hasattr(torch.hub, "load"):
            model = torch.hub.load("pytorch/vision", args.model, pretrained=False)
            checker = ROCmChecker(rocm_version=args.rocm_version)
            result = checker.check_model(model)
            print(result.summary())

            if args.output:
                from src.report import ReportGenerator
                gen = ReportGenerator()
                gen.add_result(result)
                if args.output.endswith(".json"):
                    gen.generate_json(args.output)
                else:
                    gen.generate_html(args.output)
                print(f"\nReport saved to: {args.output}")
    except ImportError:
        print("\n⚠️  PyTorch not installed — model loading skipped")
    except Exception as e:
        print(f"\n⚠️  Could not load model: {e}")


if __name__ == "__main__":
    main()
