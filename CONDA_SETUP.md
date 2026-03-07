# SpendWise AI – Conda setup (Python 3.11)

Using **Python 3.11** avoids the macOS mutex crash with HuggingFace tokenizers that can occur on Python 3.13.

## Option A: Run the setup script

From the project root:

```bash
cd "/Users/patel/Documents/E/MPS Sem 5/ALY 6030/Final_Project/spendwise-ai"
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

Then:

```bash
conda activate spendwise311
cd "/Users/patel/Documents/E/MPS Sem 5/ALY 6030/Final_Project/spendwise-ai"
streamlit run app/streamlit_app.py
```

## Option B: Manual steps

```bash
# 1. Create environment
conda create -n spendwise311 python=3.11 -y

# 2. Activate
conda activate spendwise311

# 3. Install dependencies
pip install torch torchvision torchaudio
pip install "transformers>=4.30" datasets
pip install pandas numpy scikit-learn
pip install streamlit plotly matplotlib seaborn
pip install Pillow tqdm anthropic sentencepiece python-dotenv
pip install pytesseract   # optional, for Tesseract OCR fallback

# 4. Go to project and run
cd "/Users/patel/Documents/E/MPS Sem 5/ALY 6030/Final_Project/spendwise-ai"
streamlit run app/streamlit_app.py
```

## Optional: Tesseract (receipt fallback)

If you want the Receipt Scanner to have a Tesseract fallback when Donut is slow or unavailable:

```bash
brew install tesseract
```

Then `pytesseract` (already in the pip list above) will work.

## Run the dashboard

Whenever you want to start the app:

```bash
conda activate spendwise311
cd "/Users/patel/Documents/E/MPS Sem 5/ALY 6030/Final_Project/spendwise-ai"
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (e.g. http://localhost:8501).
