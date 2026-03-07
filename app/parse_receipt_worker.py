"""
Run receipt parsing in a subprocess to avoid PyTorch/transformers
mutex issues in the main Streamlit process (macOS libc++abi crash).
Usage: python parse_receipt_worker.py <image_path>
Outputs: JSON to stdout, errors to stderr, exit 0 on success.
"""
import sys
import json
from pathlib import Path

# Project root and src on path
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
        # Always return JSON to stdout so the Streamlit UI can show a clean fallback.
        print(json.dumps({"error": str(e)}))
        return


if __name__ == "__main__":
    main()
