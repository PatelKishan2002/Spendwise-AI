"""Subprocess entry: ``python parse_receipt_worker.py <image_path>`` → JSON on stdout."""

import sys
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: parse_receipt_worker.py <image_path>"}))
        return
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {str(image_path)}"}))
        return
    try:
        from receipt_parser import ReceiptParser
        parser = ReceiptParser()
        result = parser.parse(str(image_path))
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return


if __name__ == "__main__":
    main()
