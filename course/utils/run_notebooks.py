"""
Utility script to execute Jupyter notebooks top-to-bottom and verify zero errors.
Usage: python course/utils/run_notebooks.py [notebook_path_or_pattern]
"""

import sys
import time
import subprocess
from pathlib import Path


def run_notebook(nb_path: str | Path, timeout: int = 300) -> bool:
    path = Path(nb_path).resolve()
    if not path.exists():
        print(f"[ERROR] Notebook not found: {path}")
        return False

    print(f"\n{'='*70}")
    print(f"RUNNING: {path.name}")
    print(f"{'='*70}")

    start_time = time.time()
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(path.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.time() - start_time
        if proc.returncode == 0:
            print(f"[SUCCESS] {path.name} finished in {elapsed:.2f}s")
            return True
        else:
            print(f"[FAILED] {path.name} failed after {elapsed:.2f}s (code {proc.returncode})")
            if proc.stdout:
                print("STDOUT:\n", proc.stdout[-1500:])
            if proc.stderr:
                print("STDERR:\n", proc.stderr[-1500:])
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[FAILED] {path.name} execution raised exception after {elapsed:.2f}s: {e}")
        return False



def run_all(pattern: str = "course/notebooks/*.ipynb") -> list[tuple[str, bool]]:
    root = Path.cwd()
    nb_files = sorted(root.glob(pattern))
    if not nb_files:
        print(f"[WARN] No notebooks found matching: {pattern}")
        return []

    results = []
    for f in nb_files:
        success = run_notebook(f)
        results.append((f.name, success))

    print(f"\n{'='*70}")
    print("SUMMARY RESULTS")
    print(f"{'='*70}")
    all_ok = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name:40} : {status}")
        if not success:
            all_ok = False
    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "course/notebooks/*.ipynb"
    results = run_all(target)
    if any(not s for _, s in results):
        sys.exit(1)
