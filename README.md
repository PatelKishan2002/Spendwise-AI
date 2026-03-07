# SpendWise AI

AI-Powered Personal Finance Intelligence System

## Features

- Receipt Scanning: Extract data from receipt images (Donut OCR)
- Transaction Classification: Auto-categorize with high accuracy
- Anomaly Detection: Flag unusual spending (VAE)
- Spending Forecast: Predict expenses (Transformer)
- AI Assistant: Natural language queries (Claude)
- Smart Recommendations: Personalized savings suggestions

## Tech Stack

PyTorch, Transformers, Scikit-learn, Streamlit, Anthropic Claude (optional)

## Quick Start

  pip install -r requirements.txt
  streamlit run app/streamlit_app.py

## Project Structure

  spendwise-ai/
  - src/ (receipt_parser, transaction_classifier, anomaly_detector, spending_forecaster, llm_assistant, recommendation_engine)
  - models/
  - data/
  - app/streamlit_app.py
