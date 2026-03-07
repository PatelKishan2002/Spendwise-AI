#!/bin/bash
# SpendWise AI - Conda environment setup (Python 3.11)
# Use this to avoid macOS mutex crash with HuggingFace tokenizers on Python 3.13.
#
# Usage:
#   ./setup_conda_env.sh
# Then:
#   conda activate spendwise311
#   cd /path/to/spendwise-ai
#   streamlit run app/streamlit_app.py

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Create conda environment spendwise311 (Python 3.11) ==="
conda create -n spendwise311 python=3.11 -y

echo ""
echo "=== Step 2: Install dependencies into spendwise311 ==="
conda run -n spendwise311 pip install --upgrade pip
conda run -n spendwise311 pip install torch torchvision torchaudio
conda run -n spendwise311 pip install "transformers>=4.30" datasets
conda run -n spendwise311 pip install pandas numpy scikit-learn
conda run -n spendwise311 pip install streamlit plotly matplotlib seaborn
conda run -n spendwise311 pip install Pillow tqdm anthropic sentencepiece python-dotenv
conda run -n spendwise311 pip install pytesseract 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Run the app with:"
echo "  conda activate spendwise311"
echo "  cd \"$SCRIPT_DIR\""
echo "  streamlit run app/streamlit_app.py"
