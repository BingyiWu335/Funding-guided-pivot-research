import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "figure", nargs="?", default="all", choices=("all", "1", "2", "3", "4", "5")
    )
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent / "plots"
    figures = range(1, 6) if args.figure == "all" else (int(args.figure),)
    for number in figures:
        subprocess.run(
            [sys.executable, str(scripts / f"figure{number}.py")], check=True
        )


if __name__ == "__main__":
    main()
