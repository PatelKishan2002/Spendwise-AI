"""
Diagnostic script - run from spendwise-ai/ root:
    conda activate spendwise
    python diagnose_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"Python: {sys.version}")
print(f"Project root: {PROJECT_ROOT}")
print(f"src path: {PROJECT_ROOT / 'src'}")
print()

# --- Test 1: Classifier ---
print("=" * 50)
print("TEST 1: Transaction Classifier")
print("=" * 50)
try:
    from transaction_classifier import TransactionClassifierInference
    print("  Import: OK")
    model_path = str(PROJECT_ROOT / "models/classifier_model")
    print(f"  Model path: {model_path}")
    print(f"  model.pt exists: {Path(model_path, 'model.pt').exists()}")
    print(f"  tokenizer.json exists: {Path(model_path, 'tokenizer.json').exists()}")
    classifier = TransactionClassifierInference(model_path)
    result = classifier.classify("STARBUCKS #1234 $5.75")
    print(f"  Result: {result}")
    print("  STATUS: OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()

# --- Test 2: Anomaly Detector ---
print("=" * 50)
print("TEST 2: Anomaly Detector")
print("=" * 50)
try:
    from anomaly_detector import AnomalyDetector
    print("  Import: OK")
    model_path = str(PROJECT_ROOT / "models/anomaly_model")
    print(f"  Model path: {model_path}")
    print(f"  model.pt exists: {Path(model_path, 'model.pt').exists()}")
    detector = AnomalyDetector(model_path)
    print("  Load: OK")
    print("  STATUS: OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()

# --- Test 3: Forecaster ---
print("=" * 50)
print("TEST 3: Spending Forecaster")
print("=" * 50)
try:
    from spending_forecaster import SpendingForecasterInference
    print("  Import: OK")
    model_path = str(PROJECT_ROOT / "models/forecaster_model")
    print(f"  Model path: {model_path}")
    print(f"  model.pt exists: {Path(model_path, 'model.pt').exists()}")
    
    # Check what keys the checkpoint actually has
    import torch
    checkpoint = torch.load(f"{model_path}/model.pt", map_location="cpu", weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    if "config" in checkpoint:
        print(f"  Config: {checkpoint['config']}")
    # Show all top-level keys that contain 'scaler'
    scaler_keys = [k for k in checkpoint.keys() if "scaler" in k.lower()]
    print(f"  Scaler keys: {scaler_keys}")
    for k in scaler_keys:
        print(f"    {k}: {checkpoint[k]}")
    
    forecaster = SpendingForecasterInference(model_path)
    result = forecaster.predict([5000, 5500, 4800, 5200, 4900, 5100, 5300, 4700])
    print(f"  Result: {result}")
    print("  STATUS: OK")
except Exception as e:
    import traceback
    print(f"  FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print()

# --- Test 4: Recommendation Engine ---
print("=" * 50)
print("TEST 4: Recommendation Engine")
print("=" * 50)
try:
    from recommendation_engine import RecommendationService
    print("  Import: OK")
    print("  STATUS: OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()

# --- Test 5: LLM Assistant ---
print("=" * 50)
print("TEST 5: LLM Assistant")
print("=" * 50)
try:
    from llm_assistant import FinancialDataManager, FinancialAssistant
    print("  Import: OK")
    print("  STATUS: OK")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()
print("=" * 50)
print("DONE")
print("=" * 50)

