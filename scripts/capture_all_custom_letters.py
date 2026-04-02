import argparse
import string
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    python_exe = Path(sys.executable)
    capture_script = PROJECT_ROOT / "scripts" / "capture_custom_letters.py"
    letters = list(string.ascii_uppercase) + ["space", "del", "nothing"]

    for letter in letters:
        print(f"\nStarting capture for {letter}. Close the window or press Q when done.\n")
        command = [
            str(python_exe),
            str(capture_script),
            "--config",
            args.config,
            "--letter",
            letter,
            "--count",
            str(args.count),
        ]
        result = subprocess.run(command, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
