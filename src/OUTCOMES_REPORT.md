# SpendWise AI – Outcomes Report

Single place to see **what each file produces**: outputs, metrics, and artifacts.  
**Update this file** when you add or re-run notebooks (e.g. 05–08).

---

## File 1: `01_data_preparation.ipynb`

**Purpose:** Generate synthetic transactions, define taxonomy, create train/val/test splits for the classifier.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `data/synthetic/transactions_full.csv` | All synthetic transactions (122,754 rows) |
| `data/processed/train.csv` | Training set for classifier (85,927 samples) |
| `data/processed/val.csv` | Validation set (18,413 samples) |
| `data/processed/test.csv` | Test set (18,414 samples) |
| `data/processed/label_mappings.json` | category/subcategory to ID and back |
| `data/raw/categories/taxonomy.json` | Full category taxonomy |
| `configs/data_config.json` | Data generation config |
| `data/processed/eda_category_distribution.png` | EDA plot |
| `data/processed/eda_monthly_trend.png` | EDA plot |
| `data/processed/eda_amount_distribution.png` | EDA plot |

**Actual run results:**
- Total transactions: **122,754**; unique users: **100**; date range **2025-08-25 → 2026-02-21**
- Categories: **12**; subcategories: **54**
- Train: **85,927 (70.0%)**; Val: **18,413 (15.0%)**; Test: **18,414 (15.0%)**
- Category distribution (example): Bills & Utilities 6.0%, Education 4.9%, Entertainment 6.1%, Financial 4.7%, Food & Dining 22.0% (similar across splits)
- All files saved: `transactions_full.csv` 122,754 rows; `train.csv` 85,927; `val.csv` 18,413; `test.csv` 18,414; `label_mappings.json`; `taxonomy.json`; `data_config.json`

---

## File 2: `02_receipt_ocr_model.ipynb`

**Purpose:** Load Donut (CORD), visualize receipts, build and test ReceiptParser, save production module.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `data/processed/receipt_samples.png` | Grid of sample CORD receipt images |
| `data/processed/receipt_extraction_demo.png` | One receipt with extracted items/total |
| `src/receipt_parser.py` | Production ReceiptParser class |

**Actual run results:**
- Device: **cpu**
- CORD dataset: Train **800** receipts; Validation **100**; Test **100**
- Sample keys: `image`, `ground_truth`; image size (864, 1296)
- Donut model: **naver-clova-ix/donut-base-finetuned-cord-v2**; **260,083,832** parameters; loaded on cpu
- Encoder: 1024-dim features; Decoder: vocab **57,580** tokens, hidden size 1024
- Single-receipt run: tensor shape [1, 3, 1280, 960]; generated 54 tokens; parsed JSON (menu items, subtotal)
- Parser ready; testing on 5 sample receipts (items found, total $ per receipt)
- Parser module saved to `src/receipt_parser.py`

---

## File 3: `03_transaction_classifier.ipynb`

**Purpose:** Fine-tune DistilBERT for category/subcategory classification; evaluate; save model and inference module.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `models/classifier_model/model.pt` | Model state, config, label_mappings, metrics |
| `models/classifier_model/tokenizer_config.json` | Tokenizer config |
| `models/classifier_model/tokenizer.json` | Tokenizer vocab |
| `models/classifier_model/training_history.png` | Loss and accuracy curves |
| `models/classifier_model/confusion_matrix.png` | Test-set confusion matrix |
| `src/transaction_classifier.py` | Production classifier + inference class |

**Actual run results:**
- Data loaded: Train **85,927**; Val **18,413**; Test **18,414**; Categories **12**; Subcategories **54**
- Tokenizer: **distilbert-base-uncased**; vocabulary size **30,522**
- Datasets: Train 85,927; Val 18,413; Test 18,414; sample shapes input_ids [64], attention_mask [64]
- Model: **66,978,882** total parameters (all trainable); 12 categories, 54 subcategories
- Training: Batch 32, LR 2e-05, Epochs 3, total steps 8,058, warmup 805, batches/epoch 2,686
- Test results: **Category Accuracy 94.84%**; **Category F1 0.9492**; **Subcategory Accuracy 92.63%**
- Classification report (per category): e.g. Bills & Utilities 1.00/1.00/1.00; Education 0.56/0.65/0.60; Food & Dining 1.00/1.00/1.00; Transportation 1.00/1.00/1.00; etc.
- Inference test: STARBUCKS → Food & Dining 100%, Coffee Shops 100%; UBER → Transportation 100%, Rideshare 100%; AMAZON → Shopping 100%, Amazon 99.9%; PEETS COFFEE → Food & Dining 100%, Coffee Shops 89.1%; etc.
- Model saved to `models/classifier_model`; module to `src/transaction_classifier.py`

---

## File 4: `04_anomaly_detection.ipynb`

**Purpose:** Train VAE on weekly spending vectors; set anomaly threshold; build AnomalyDetector; save model and module.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `models/anomaly_model/model.pt` | VAE state, config, threshold, scaler, category_cols |
| `models/anomaly_model/training_history.png` | VAE loss and reconstruction loss |
| `models/anomaly_model/error_distribution.png` | Histogram/boxplot of reconstruction errors |
| `models/anomaly_model/normal_vs_anomaly.png` | Normal vs synthetic anomaly error distribution |
| `src/anomaly_detector.py` | Production VAE + AnomalyDetector (same logic as notebook 04; AnomalyDetector(model_path), then detect(spending)) |

**Notebook updates:** create_weekly_spending_vectors is defined in its own cell (after loading data), then called in the next cell. The notebook prints "Module saved to src/anomaly_detector.py"; the file src/anomaly_detector.py exists and matches the VAE and AnomalyDetector run in the notebook.

**Actual run results:**
- Loaded **122,754** transactions; **12** categories; weekly spending vectors: **2,600** samples (user-weeks), **11** features (categories)
- Data prepared: Train **2,080** samples, Val **520**; input dimension **11**
- VAE: input_dim 11, hidden 64, latent 16; **12,971** total parameters
- Training: 50 epochs; final Train Loss **0.4451**, Val Loss **0.4508**, Recon **0.1902**
- Reconstruction error: Min **0.0285**, Max **0.6462**, Mean **0.1840**, Std **0.0890**
- Anomaly threshold (95th percentile): **0.3546**; anomalies detected: **130 (5.0%)**
- Synthetic anomalies: detection rate **16%** (16/100)
- Detector test: Normal week vs ANOMALY (shopping spree, overspending, suspiciously quiet) with anomaly score and top categories
- Model saved to `models/anomaly_model`; "Module saved to src/anomaly_detector.py"; summary: "VAE trained; threshold 95th percentile; detection rate on synthetic anomalies: 16%"

---

## File 5: `05_spending_forecaster.ipynb`

**Purpose:** Predict future spending (per category + total, with uncertainty) using ZICATT (Zero-Inflated Cross-Attention Temporal Transformer). The dashboard and app use this model only.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `models/forecaster_model/model.pt` | ZICATT checkpoint (state, config, scalers, categories, metrics) |
| `models/forecaster_model/training_history_zicatt.png` | Train/val loss curves (total, gate, amount) |
| `models/forecaster_model/predictions_zicatt.png` | Per-category actual vs predicted |
| `models/forecaster_model/uncertainty_zicatt.png` | Average uncertainty per category |
| `src/spending_forecaster.py` | ZICATT, ZICATTInference (production) |

**Actual run results:**
- Per-user category-week matrix; sequences **(1,800, 8, 11)** with ~**10.9%** zero targets; split Train **1,260**, Val **270**, Test **270**
- ZICATT: **141,059** parameters; temporal + cross-category attention; early stopping
- Gate accuracy **88.3%**; total MAE ≈ **$1.7K**, RMSE ≈ **$2.2K**, MAPE ≈ **39%**; 95% interval coverage **94.5%**
- Inference: per-category breakdown; one-week total forecast ~**$5.6K**; model and `src/spending_forecaster.py` used by API and Streamlit app

---

## File 6: `06_llm_integration.ipynb`

**Purpose:** Build natural language interface using Claude API; query spending in plain English; connect classifier, anomaly, and forecaster.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `src/llm_assistant.py` | FinancialDataManager, FinancialAssistant (production) |

**Actual run results:**
- Data loaded: **122,754** transactions; date range **2025-08-25** to **2026-02-21**
- Testing Data Functions: (1) Spending by category: total and top 3 categories for user_0001; (2) Subscriptions: count and monthly total; (3) Last month: total spending and transaction count
- Defined **5** tools for Claude; FinancialAssistant (demo mode if no API key: "Running in demo mode")
- Test queries return keyword-based responses (subscriptions, spending by category, trend, summary)
- ChatSession demo (3 messages); EnhancedFinancialAssistant: anomaly detector and forecaster loaded if models exist
- "Module saved to src/llm_assistant.py"

---

## File 7: `07_recommendation_engine.ipynb`

**Purpose:** Generate personalized savings recommendations (overspending, subscriptions, high-frequency, positive trends, budget health).

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `src/recommendation_engine.py` | Priority, RecommendationType, Recommendation, SpendingAnalyzer, RecommendationEngine, RecommendationService |

**Actual run results:**
- Loaded **122,754** transactions; Testing Analyzer: category stats (top 5: last_month, mean, vs_average_pct); Subscriptions: count and total_monthly; high-frequency categories count; savings rate (%)
- Sample recommendation (JSON); engine generates N recommendations for user_0001; total potential savings $X/month ($Y/year)
- Multi-user summary table: 10 users (user_0001–user_0010) with High/Medium/Low/Positive counts and Savings column
- RecommendationService: get_recommendations(limit=5), get_savings_summary; potential_monthly_savings and by_category; "Module saved to src/recommendation_engine.py"

---

## File 8: `08_final_pipeline.ipynb`

**Purpose:** Integrate all components; unified SpendWise API; Streamlit dashboard; requirements, README, project report.

**Key outputs (files created/updated):**

| Path | Description |
|------|-------------|
| `app/streamlit_app.py` | Streamlit ML Showcase (Dashboard, Transactions, Analytics, Insights, AI Assistant, Receipt Scanner) + sidebar **Mode** handoff to `personal_account.py` |
| `app/personal_account.py` | My Account: login, My Dashboard, Add Expense, My Transactions, My Assistant, My Insights |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and quick start |
| `PROJECT_REPORT.txt` | Completion report (components, files) |

**Actual run results:**
- Data loaded: **122,754** transactions; Components Status: **data OK**; classifier, anomaly, forecaster, recommender, assistant **OK** (or "not available" if model/file missing)
- SpendWise API initialized: **100** users, **12** categories
- Test API (user_0001): Spending summary (total_expenses, total_income, net, change_pct); top 5 categories with amount and percentage; monthly trend average; anomaly status and score (/100); forecast predicted and range (or error); top 3 recommendation titles
- Streamlit app path: `app/streamlit_app.py`; requirements.txt created; README.md created; PROJECT_REPORT.txt saved (components list, files, skills)

---

## Production modules (from notebooks)

| Module | Produced by | Use |
|--------|-------------|-----|
| `src/receipt_parser.py` | Notebook 02 | Parse receipt images to structured data |
| `src/transaction_classifier.py` | Notebook 03 | Classify transaction text to category/subcategory |
| `src/anomaly_detector.py` | Notebook 04 | Score spending vectors for anomaly |
| `src/spending_forecaster.py` | Notebook 05 | Predict next week spending from history |
| `src/llm_assistant.py` | Notebook 06 | Financial assistant (data manager + Claude/demo chat) |
| `src/recommendation_engine.py` | Notebook 07 | RecommendationService, SpendingAnalyzer, RecommendationEngine |

---

*Last updated: add date when you add or re-run files. When you add notebooks 05–08, fill in the corresponding sections above.*
