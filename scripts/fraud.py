import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.cli import main

if __name__ == "__main__":
    main()
