**Single outcomes summary:** For a consolidated list of what each notebook/file produces (outputs, metrics, artifacts), see **OUTCOMES_REPORT.md** in this folder. Update that file when you add notebooks 05–08.

---

Part 1: The Transformer Foundation
Everything in this project (Donut, BERT, Forecaster) is built on one core idea: the Attention Mechanism.
Why Attention Exists
The Problem: Before attention, models read sequences word-by-word and forgot earlier parts. Like reading a book but only remembering the last page.
The Solution: Attention lets the model look at ALL words simultaneously and decide which ones matter for the current task.
Attention in One Sentence

"For each word, compute how much every other word should influence its meaning."

The Math (Simplified)
Input: "I went to the bank to deposit money"

For the word "bank":
- How relevant is "I"?        → Low score (0.02)
- How relevant is "deposit"?  → High score (0.45)
- How relevant is "money"?    → High score (0.40)

Result: Model understands "bank" = financial institution, not river bank
This is computed using three vectors per word:

Query (Q): "What am I looking for?"
Key (K): "What do I contain?"
Value (V): "What information do I provide?"

Attention Score = softmax(Q × K^T) × V
That's it. Every transformer—BERT, GPT, Donut—uses this same mechanism.

Part 2: The Five Components Explained
Component 1: Receipt OCR (Donut)
What it does: Image → Structured JSON
Why it's hard: A receipt isn't just text. The position matters—"$5.99" means nothing unless you know it's next to "Coffee."
How Donut works:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Receipt   │ ──► │   Vision    │ ──► │   Text      │ ──► JSON
│   Image     │     │   Encoder   │     │   Decoder   │
└─────────────┘     │  (Swin)     │     │  (BART)     │
                    └─────────────┘     └─────────────┘

Vision Encoder: Breaks image into patches, learns spatial relationships (what's next to what)
Text Decoder: Generates structured output token by token, using attention to look back at image features

Key insight: Donut skips traditional OCR entirely. No Tesseract. It goes directly from pixels to structured text—this is why it handles messy receipts better.

Component 2: Transaction Classifier (BERT)
What it does: "STARBUCKS #1234 $5.75" → Category: Coffee Shops
Why BERT and not simple rules?
Rules fail on edge cases:

"SQ *BLUE BOTTLE" → Is this a bottle shop or Blue Bottle Coffee?
"AMZN MKTP US" → Amazon sells everything—is it groceries, electronics, clothing?

How BERT classification works:
┌──────────────────────────────────────────────────────┐
│  [CLS] STARBUCKS #1234 $5.75 [SEP]                   │
│    ↓       ↓        ↓     ↓    ↓                     │
│  ┌─────────────────────────────────┐                 │
│  │     BERT (12 attention layers)  │                 │
│  └─────────────────────────────────┘                 │
│    ↓                                                 │
│  [CLS] embedding (768 dimensions)                    │
│    ↓                                                 │
│  Classification Head (Linear → 12 categories)        │
└──────────────────────────────────────────────────────┘
Key insight: The [CLS] token's final embedding captures the "meaning" of the entire transaction. We only use that one vector for classification.
Fine-tuning: BERT already knows English. We're just teaching it: "When you see patterns like 'UBER *TRIP', that's Transportation."

Component 3: Anomaly Detection (VAE)
What it does: Flags unusual spending → "This $800 electronics purchase is abnormal for you"
Why VAE and not rules?
Rules: "Flag anything over $500" → Too rigid. $500 is normal for rent, abnormal for coffee.
VAE learns YOUR normal patterns.
How VAE works:
Normal spending pattern:
[groceries: $200, coffee: $50, transport: $100, ...]
        ↓
   ┌─────────┐
   │ Encoder │ → Compressed representation (latent space)
   └─────────┘
        ↓
   ┌─────────┐
   │ Decoder │ → Reconstructed pattern
   └─────────┘
        ↓
[groceries: $195, coffee: $48, transport: $105, ...]

Reconstruction Error: LOW → Normal spending
Anomalous spending:
[groceries: $200, coffee: $500, transport: $100, ...]
        ↓
   Encoder → Decoder → [groceries: $198, coffee: $55, transport: $102]
        
Reconstruction Error: HIGH (500 vs 55) → ANOMALY DETECTED
Key insight: VAE learns to compress and reconstruct "normal." Anything it can't reconstruct well = abnormal.

Component 4: Spending Forecaster (Transformer)
What it does: Past 6 months → "Next month you'll spend ~$2,400"
Why Transformer for time-series?
Traditional methods (ARIMA) assume patterns are linear and stationary. Real spending isn't—you spend more in December (holidays), less in January (recovery).
How it works:
Input: [Jan: $2100, Feb: $2300, Mar: $2150, Apr: $2400, May: $2250, Jun: ?]
                                    ↓
        ┌───────────────────────────────────────────┐
        │  Transformer with Temporal Attention      │
        │  - Learns: "June follows May patterns"    │
        │  - Learns: "Weekends = higher spending"   │
        │  - Learns: "End of month = bills due"     │
        └───────────────────────────────────────────┘
                                    ↓
Output: Jun: $2,380 (±$150)
Key insight: Attention lets the model look at ALL past months and decide which are most relevant for predicting the next one. December last year matters for predicting December this year.

Component 5: LLM Integration (Claude/GPT)
What it does: "How much did I spend on Uber this month?" → "$156 across 12 trips"
The architecture:
User Query: "What's my biggest expense category?"
                        ↓
              ┌─────────────────┐
              │  Query Parser   │  (Understand intent)
              └─────────────────┘
                        ↓
              ┌─────────────────┐
              │  Data Retrieval │  (Fetch from your transaction DB)
              └─────────────────┘
                        ↓
              ┌─────────────────┐
              │  LLM Response   │  (Generate natural answer)
              └─────────────────┘
                        ↓
"Your biggest expense is Food & Dining at $847, 
 which is 32% of your total spending."
Key insight: The LLM doesn't store your data. It receives your query + relevant transaction data as context, then generates a response.




*** Notebook 01 ***

1. **Imports & Configuration**  
   Loads all libraries (pandas, numpy, PyTorch, sklearn, etc.), sets a fixed random seed so runs are reproducible, and defines CONFIG (e.g. 100 users, 6 months, 70/15/15 train/val/test). Also prints PyTorch version and whether CUDA is available.

2. **Project Directory Setup**  
   Creates the folder structure (e.g. `data/raw/`, `data/processed/`, `models/`, `src/`, `configs/`) so raw data, processed data, and model outputs stay organized.

3. **Category Taxonomy Definition**  
   Defines the 12 main spending categories and their subcategories (e.g. Food & Dining → Restaurants, Groceries, Coffee Shops, …). Saves this as `taxonomy.json` so every other step uses the same labels.

4. **Merchant Name Generator**  
   A class that creates realistic bank-style merchant names (e.g. "AMZN MKTP US*…", "STARBUCKS #6890") instead of plain names like "Amazon", so the data looks like real statements.

5. **Transaction Generator Class**  
   A class that creates full transaction rows: realistic amounts per category (e.g. $4–15 for coffee, $50–200 for insurance), date patterns (daily coffee, monthly bills), and some users spending more than others.

6. **Generate Dataset**  
   Runs the transaction generator for 100 users over 6 months and stores everything in one big DataFrame (`transactions_df`). This is the synthetic training data.

7. **Exploratory Data Analysis (EDA)**  
   Plots the data: counts by category, monthly spending over time, and amount distributions. Helps spot issues (e.g. one category dominating) before training.

8. **Prepare Classifier Dataset & Split**  
   Turns each transaction into a text line like `"MERCHANT $AMOUNT"` and numeric category/subcategory IDs. Splits into train / validation / test (70/15/15) with **stratified** split so each split has similar category proportions.

9. **PyTorch Dataset Class**  
   Wraps the classifier data in a PyTorch `Dataset`: it tokenizes the text with BERT’s tokenizer and returns batches of `input_ids`, `attention_mask`, and labels so the model can be trained.

10. **Save Outputs**  
    Saves the full transactions CSV, train/val/test CSVs, `label_mappings.json` (so model predictions can be turned back into category names), and config. Later notebooks (e.g. the classifier in Notebook 03) load these.

Key outputs from Notebook 01:

train.csv, val.csv, test.csv → Used in Notebook 03 (classifier)
label_mappings.json → Convert model predictions back to category names
taxonomy.json → Reference for all categories

**Outcomes when you run:** Total transactions ~122K (100 users, 6 months); 12 categories, 45 subcategories. Train ~85.9K, val ~18.4K, test ~18.4K (70/15/15 stratified). Console prints row counts per saved file and category distribution across splits. EDA plots saved under data/processed/.


*** Notebook 02 – Receipt OCR (Donut) ***

1. **Imports & Setup**  
   Loads libraries (torch, transformers, DonutProcessor, VisionEncoderDecoderModel, datasets, PIL, matplotlib), sets device (CPU/GPU), and sets PROJECT_ROOT so paths work from any working directory. Creates `data/processed` if needed.

2. **Understanding Donut Architecture**  
   Markdown: explains how Donut skips traditional OCR—image goes through a single encoder–decoder to structured output (CORD format). Encoder = Swin Transformer; decoder = BART.

3. **Load Sample Receipt Dataset (CORD)**  
   Loads the CORD receipt dataset from HuggingFace (`naver-clova-ix/cord-v2` or similar), which provides receipt images and ground-truth JSON for testing the parser.

4. **Visualize Sample Receipts**  
   Plots a grid of sample receipt images from the dataset and saves the figure as `data/processed/receipt_samples.png`.

5. **Load Pre-trained Donut Model**  
   Loads the pre-trained Donut model and processor (`naver-clova-ix/donut-base-finetuned-cord-v2`), moves the model to the chosen device, and sets it to eval mode.

6. **Understanding the Model Components**  
   Inspects the encoder and decoder (e.g. input/output shapes, parameter counts) so you see how the image becomes tokens and then JSON.

7. **Process a Single Receipt**  
   Runs one receipt image through the model: preprocess → generate → decode to text, then parses the raw JSON string so you see the model’s output before post-processing.

8. **Visualize Extraction Result**  
   Draws the receipt image and overlays or prints the extracted menu items and totals, then saves the figure as `data/processed/receipt_extraction_demo.png`.

9. **Build Production-Ready ReceiptParser Class**  
   Defines the `ReceiptParser` class: loads model/processor, implements `parse(image)` (handles PIL/path/array), and `_post_process(raw)` to turn the model’s JSON into a clean dict (items with name/quantity/price, subtotal, tax, total). Handles both dict and string menu items from the model.

10. **Test the Parser on Multiple Receipts**  
    Instantiates `ReceiptParser`, then runs it on several CORD test samples (e.g. indices 0, 5, 10, 15, 20) and prints the number of items and total for each receipt.

11. **Measure Performance**  
    Defines `evaluate_parser()` to time parsing and count items per receipt over a subset (e.g. 30 samples), then prints average time per receipt and average items detected.

12. **Save Parser Module for Production**  
    Writes the `ReceiptParser` class (and helpers) to a Python module (e.g. `src/receipt_parser.py` or under PROJECT_ROOT) so other notebooks or the app can import and use it.

Key outputs from Notebook 02:

receipt_samples.png, receipt_extraction_demo.png → EDA/visualization of Donut on CORD
ReceiptParser class (in notebook and saved .py) → Used to parse receipt images in the pipeline

**Outcomes when you run:** CORD dataset sizes (train/val/test); Donut model loaded on device with parameter count. Single-receipt extraction shows raw JSON and parsed items/total. Parser test on 5 receipts: items found and total per receipt. Evaluation: average time per receipt and average items detected. src/receipt_parser.py written at end.


*** Notebook 03 – Transaction Classifier (BERT) ***

1. **Imports & Setup**  
   Loads libraries (torch, transformers, sklearn, pandas, etc.), sets seed and device, and sets PROJECT_ROOT (works from spendwise-ai/, spendwise-ai/src/, or parent). Creates data/processed and models/classifier_model if needed.

2. **Load Prepared Data from Notebook 01**  
   Loads train/val/test CSVs and label_mappings.json; prints sample counts and a quick look at sample transactions.

3. **Understanding BERT Tokenization**  
   Markdown explains token IDs, [CLS]/[SEP], attention mask. Code: load DistilBERT tokenizer, demonstrate tokenization on a sample string, show full encoding with padding/truncation.

4. **PyTorch Dataset Class**  
   Defines TransactionDataset (text → tokenized input_ids, attention_mask, category_label, subcategory_label). Creates train/val/test datasets and verifies one sample.

5. **Build the Classifier Model**  
   Markdown describes architecture (BERT → [CLS] → two heads). Defines TransactionClassifier (category + subcategory heads). Initializes model and prints parameter counts.

6. **Training Configuration**  
   Sets BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO; creates DataLoaders, AdamW optimizer, linear schedule with warmup, CrossEntropyLoss.

7. **Training Functions**  
   Defines train_epoch() (forward, loss 0.6*cat + 0.4*subcat, backward, clip grad, step) and evaluate() (no grad, collect preds/labels, return loss, acc, F1).

8. **Train the Model**  
   Runs NUM_EPOCHS training loop, keeps history and best_model_state by validation category accuracy.

9. **Visualize Training Progress**  
   Plots train/val loss, category accuracy, subcategory accuracy; saves training_history.png.

10. **Final Evaluation on Test Set**  
    Loads best weights, evaluates on test set, prints classification report and plots normalized confusion matrix (confusion_matrix.png).

11. **Save Model for Production**  
    Saves model.pt (state_dict, label_mappings, config, metrics) and tokenizer to models/classifier_model/.

12. **Build Inference Class for Production**  
    Defines TransactionClassifierInference: load checkpoint, tokenizer, model; classify(text) returns category, subcategory, and confidences.

13. **Test the Inference Class**  
    Instantiates classifier and runs it on a list of sample transaction strings (including new patterns), printing category and subcategory for each.

14. **Save Inference Module**  
    Writes the classifier and inference class as Python code to src/transaction_classifier.py for use in the app.

Key outputs from Notebook 03:

models/classifier_model/model.pt, tokenizer → Saved model and tokenizer
training_history.png, confusion_matrix.png → Training and evaluation visuals
src/transaction_classifier.py → Production inference module

**Outcomes when you run:** Train/val/test sample counts and category/subcategory counts. Model parameter counts. Each epoch: train/val loss, category and subcategory accuracy; best model saved by val category accuracy. Test set: Category Accuracy and F1, Subcategory Accuracy; classification report and confusion matrix. Inference test on sample transactions prints category and subcategory with confidence. Model and tokenizer saved to models/classifier_model/; module to src/transaction_classifier.py.


*** Notebook 04 – Anomaly Detection (VAE) ***

1. **Imports & Setup**  
   Loads libraries (torch, sklearn, pandas, etc.), sets seed and device, and sets PROJECT_ROOT (same pattern as 03). Creates models/anomaly_model if needed.

2. **Prepare Spending Pattern Data**  
   Loads transactions_full.csv and label_mappings.json. A dedicated cell defines create_weekly_spending_vectors(): aggregate expenses by (user_id, week, category) into a pivot; each row is one user-week, columns are category totals. The next cell calls it and prints sample and category list.

3. **Prepare Training Data**  
   Builds feature matrix X from category columns, scales with StandardScaler, train/val split, TensorDataset and DataLoaders (BATCH_SIZE 64).

4. **Understanding VAE Architecture**  
   Markdown: encoder → latent (mean, log-var) → reparameterize → decoder; VAE loss = reconstruction + KL.

5. **Build VAE Model**  
   Defines VAE class: encoder (Linear→ReLU×2), fc_mu/fc_logvar, decoder (Linear→ReLU×2→Linear). encode(), reparameterize(), decode(), forward(), get_reconstruction_error(). Initializes model (input_dim, hidden 64, latent 16) and moves to device.

6. **VAE Loss Function**  
   vae_loss(x_recon, x, mu, logvar, beta=0.5): MSE reconstruction + beta * KL (Gaussian closed form). Returns total, recon, kl.

7. **Training Loop**  
   Sets LEARNING_RATE, NUM_EPOCHS, BETA; Adam optimizer. Defines train_epoch() and evaluate() (forward, vae_loss, backward/step or no_grad).

8. **Train the VAE**  
   Runs NUM_EPOCHS, records train/val loss and recon loss in history, prints every 10 epochs.

9. **Visualize Training Progress**  
   Plots total loss and reconstruction loss (train vs val); saves training_history.png.

10. **Calculate Anomaly Scores**  
    Gets reconstruction error for all X_scaled with get_reconstruction_error(); prints min/max/mean/std. Plots histogram and boxplot of errors; saves error_distribution.png.

11. **Set Anomaly Threshold**  
    Threshold = 95th percentile of recon errors; flags is_anomaly; prints count and percentage.

12. **Create Synthetic Anomalies**  
    create_synthetic_anomalies(): spike (one category 3–5x), multi_spike (2–3 categories elevated), zero (0.1x). Builds 100 synthetic anomalies, gets their recon errors, computes detection rate vs threshold.

13. **Visualize Normal vs Anomaly**  
    Overlays histograms of normal vs anomaly recon errors and threshold line; saves normal_vs_anomaly.png.

14. **Build Anomaly Detector Class**  
    AnomalyDetector(model_path=None): loads from checkpoint or uses in-memory model/scaler/threshold/category_cols. preprocess(spending dict) → normalized vector. detect(spending) → anomaly_score, is_anomaly, reconstruction_error, top_anomalous_categories.

15. **Test the Detector**  
    Instantiates detector, runs on four test cases (normal week, shopping spree, overspending, suspiciously quiet); prints status and top categories.

16. **Save Model**  
    Saves model.pt (state_dict, config, threshold, scaler_mean, scaler_std, category_cols) to models/anomaly_model/.

17. **Save Module for Production**  
    The notebook prints "Module saved to src/anomaly_detector.py". The file src/anomaly_detector.py is the production module (same VAE and AnomalyDetector logic as run in the notebook): AnomalyDetector(model_path) loads model.pt and detect(spending) returns anomaly_score, is_anomaly, reconstruction_error, top_anomalous_categories. Summary cell prints completion and detection rate.

Key outputs from Notebook 04:

models/anomaly_model/model.pt → VAE weights, config, threshold, scaler, category_cols
training_history.png, error_distribution.png, normal_vs_anomaly.png → Training and evaluation visuals
src/anomaly_detector.py → Production VAE and AnomalyDetector (matches notebook 04; use AnomalyDetector(model_path) then detect(spending))

**Outcomes when you run:** Loaded **122,754** transactions; **12** categories; **2,600** samples (user-weeks), **11** features. Train **2,080**, Val **520**; input dimension **11**. VAE: **12,971** parameters. Training: 50 epochs; final train loss **0.4451**, val **0.4508**, recon **0.1902**. Reconstruction error: min **0.0285**, max **0.6462**, mean **0.1840**, std **0.0890**. Threshold (95th %ile) **0.3546**; **130 (5.0%)** flagged. Synthetic anomalies detection rate **16%**. Detector test: normal vs anomaly cases with score and top categories. Model saved to models/anomaly_model/; "Module saved to src/anomaly_detector.py"; summary "detection rate on synthetic anomalies: 16%".


*** Notebook 05 – Spending Forecaster (Transformer) ***

1. **Imports & Setup**  
   Same pattern: torch, sklearn, pandas, set_seed, device, PROJECT_ROOT (robust), (PROJECT_ROOT / "models" / "forecaster_model").mkdir.

2. **Prepare Time-Series Data**  
   Load transactions_full.csv; define create_weekly_totals() (weekly total per user); print weekly_totals and statistics.

3. **Create Sequences**  
   LOOKBACK=8, HORIZON=1; create_sequences() sliding windows per user; X (samples, lookback), y (samples).

4. **Normalize Data & Split**  
   StandardScaler for X and y; 70/15/15 train/val/test split (shuffled indices).

5. **PyTorch Dataset**  
   TimeSeriesDataset(X, y) returns (x.unsqueeze(-1), y); DataLoaders.

6. **Positional Encoding**  
   PositionalEncoding(d_model, max_len, dropout): sin/cos; register_buffer pe.

7. **Build Transformer Forecaster**  
   SpendingForecaster: input_embedding, pos_encoder, TransformerEncoder, fc_out; forward: embed, pos, encoder, mean(dim=1), fc_out.

8. **Training Configuration**  
   LR 1e-3, 50 epochs, Adam, ReduceLROnPlateau(mode="min", factor=0.5, patience=5) — no `verbose` (removed in PyTorch 2.x), MSELoss.

9. **Training Functions**  
   train_epoch (clip_grad_norm 1.0), evaluate (returns loss, predictions, targets).

10. **Train the Model**  
    Loop; save best_model_state by val loss; scheduler.step(val_loss).

11. **Visualize Training**  
    Plot train/val loss; save training_history.png.

12. **Evaluate on Test Set**  
    Load best; preds/targets inverse transform; MAE, RMSE, MAPE.

13. **Visualize Predictions**  
    Scatter actual vs predicted; line plot; save predictions.png.

14. **Build Forecaster Class**  
    SpendingForecasterInference(model_path=None): load from checkpoint or use in-memory model/scalers/lookback; predict(history) returns predicted_spending, lower_bound, upper_bound.

15. **Test the Forecaster**  
    Four test cases (steady, increasing, decreasing, variable); print predicted and range.

16. **Save Model**  
    torch.save model_state_dict, config, scaler_X/y mean/std, metrics to models/forecaster_model/model.pt.

17. **Save Module**  
    src/spending_forecaster.py (PositionalEncoding, SpendingForecaster, SpendingForecasterInference); notebook prints "Module saved to src/spending_forecaster.py".

Key outputs from Notebook 05:

models/forecaster_model/model.pt, training_history.png, predictions.png
src/spending_forecaster.py → Production forecaster (SpendingForecasterInference)

**Outcomes when you run:** Loaded **122,754** transactions. Weekly totals: **2,600** records, **100** users, **26** weeks; spending stats Mean **$5,817.97**, Std **$3,175.47**, Min **$428.64**, Max **$21,275.87**. Sequences: X shape **(1,800, 8)**, y shape **(1,800)**; data split Train **1,260**, Val **270**, Test **270**. Forecaster: d_model 64, 4 heads, 2 layers; **69,185** total parameters. Training: 50 epochs; best val loss **0.7398**. Test: **MAE $1,768.01**, **RMSE $2,210.00**, **MAPE 43.94%**. Inference: Steady spender predicted **$2,604.31** (range $2,343.88–$2,864.74); Increasing trend **$2,605.79**; Decreasing trend **$2,802.55**; Variable spender **$2,669.67**. Model saved to `models/forecaster_model`; "Module saved to src/spending_forecaster.py`; summary cell at end.


*** Notebook 05v2 – ZICATT Category Forecaster (Zero-Inflated Cross-Attention Transformer) ***

1. **Imports & Setup**  
   Same base stack (torch, sklearn, pandas, numpy, matplotlib) plus a custom `ZeroInflatedGaussianNLLLoss`. Sets `device`, robust `PROJECT_ROOT` (falls back to the real repo if `data/` is missing), and `SAVE_DIR = models/forecaster_model`. Prints the active device and resolved project root.

2. **Category-Level Weekly Matrix**  
   Loads `data/synthetic/transactions_full.csv`, filters to expenses, and groups by (`user_id`, `week`, `category`) to build a per-user **weeks × categories** matrix. Prints counts for users, weeks, categories and a sampled first week for one user (each category’s weekly spend).

3. **Sliding-Window Sequences**  
   Uses `LOOKBACK = 8` and `HORIZON = 1` to create per-user sequences: X shape **(1,800, 8, 11)** and y shape **(1,800, 11)** (11 categories). Prints overall spending stats (mean, std, min, max) and that ~**10.9%** of target entries are exactly zero.

4. **Per-Category Normalization & Split**  
   Fits `StandardScaler` on flattened X; applies to each sequence and to y; keeps original y for evaluation. Randomly shuffles and splits into Train **1,260**, Val **270**, Test **270** and builds binary labels `y_binary` (spend vs no-spend). Prints split sizes and zero-spend ratio in training targets.

5. **Dataset & DataLoaders**  
   `SpendingDataset` returns `(x, y_scaled, y_binary, y_original)`; DataLoaders created with batch size **64**. First batch print verifies shapes: X **(64, 8, 11)**, y_scaled **(64, 11)**, y_binary **(64, 11)**, y_original **(64, 11)**.

6. **Positional Encoding & Temporal Encoder**  
   Re-uses sine/cosine `PositionalEncoding` so the transformer knows week positions, and builds `TemporalEncoder` (TransformerEncoder over time, pooled with mean over the lookback dimension).

7. **ZICATT Model (Temporal + Cross-Category Attention)**  
   `ZICATT` embeds each category, projects raw values to `d_model=64`, applies per-category temporal attention (`TemporalEncoder`), then cross-category attention (`CrossCategoryAttention`) so categories attend to each other. Three heads per category: gate logits (spend vs no-spend), mean `mu`, and `logvar` (uncertainty). Test forward pass prints **141,059** total trainable parameters and confirms outputs of shape **(batch, 11)**.

8. **Zero-Inflated Gaussian NLL Loss**  
   `ZeroInflatedGaussianNLLLoss` combines BCE-with-logits on the gate (spend vs no-spend) with Gaussian NLL on the normalized amount, computed only where `y_binary` is 1. A test call prints example total, gate, and amount losses to sanity-check shapes.

9. **Training Configuration & Functions**  
   Config: Adam (lr **1e-3**), up to **80** epochs, `ReduceLROnPlateau` (factor 0.5, patience 7) and early stopping patience **15**. `train_epoch` / `evaluate` loop over DataLoaders, compute total/gate/amount loss, clip gradients (`max_norm=1.0`), and track history.

10. **Training Loop with Early Stopping**  
    Main loop prints a compact table of epoch, train/val loss, gate loss, amount loss, and LR. Best validation loss snapshot is stored and restored at the end; training stops early when val loss hasn’t improved for 15 epochs.

11. **Training Curves**  
    Plots three panels: total loss, gate loss, and amount (Gaussian NLL) loss for train vs val. Saves the figure as `models/forecaster_model/training_history_zicatt.png` and prints that it was saved.

12. **Test-Set Evaluation (Gate + Amount + Uncertainty)**  
    Runs the best model on the test set and prints:\n   - **Gate accuracy (spend vs no-spend): 88.3% overall**.\n   - Per-category gate accuracy and spend frequency (e.g. Food & Dining 100%, Transportation 99.6%, Shopping 93%, etc.).\n   - Total weekly-amount metrics: **MAE $1,722.07**, **RMSE $2,211.36**, **MAPE 39.4%**.\n   - Per-category MAEs (e.g. Education ~$1,630, Shopping ~$436, Subscriptions ~$30).\n   - **95% prediction interval coverage: 94.5%**, close to the ideal 95%.

13. **Prediction Visualization**  
    For 6 categories, plots 50 test samples: blue bars = actual, red line = predicted expected spending; background shading marks weeks where the gate predicts “no-spend”. Saves `predictions_zicatt.png` under `models/forecaster_model/` and prints confirmation.

14. **Uncertainty Visualization**  
    Computes average σ (uncertainty) in dollars and average gate probability per category; plots a bar chart colored by gate (>0.5 blue, else red). Saves `uncertainty_zicatt.png` to `models/forecaster_model/uncertainty_zicatt.png`.

15. **Production Inference Class**  
    `ZICATTInference` loads the checkpoint and scaler stats and exposes `.predict(spending_history)`, which accepts\n   - a dict `{category: [week1, week2, ...]}`, \n   - a `(lookback, num_categories)` array, **or**\n   - a simple list of weekly totals (evenly distributed across categories for backward compatibility).\n    It returns per-category fields: `probability`, `predicted_amount`, `uncertainty`, `expected_spending`, `lower_bound`, `upper_bound`, and aggregate totals. A test call using 8 recent weeks prints: predicted total ≈ **$5,646.34**, range **$0.00–$15,653.17**, plus per-category probabilities and expected spends.

16. **Save Model & Metrics**  
    Saves `models/forecaster_model/model.pt` with model state, config (num_categories, lookback, d_model, heads, layers, etc.), scaler means/stds, categories, and metrics (gate_accuracy, MAE, RMSE, MAPE, coverage_95, best_val_loss). Prints the save path.

17. **Save Production Module (spending_forecaster.py v2)**  
    Final cell writes a clean `src/spending_forecaster.py` containing `PositionalEncoding`, `TemporalEncoder`, `CrossCategoryAttention`, `ZICATT`, and `ZICATTInference` (with the simple-total fallback). It asserts the file exists and prints the module path and file size.

Key outputs from Notebook 05v2:

- `models/forecaster_model/model.pt` → ZICATT checkpoint (per-category gate + amount + uncertainty)\n- `models/forecaster_model/training_history_zicatt.png` → Training/validation loss curves (total, gate, amount)\n- `models/forecaster_model/predictions_zicatt.png` → Per-category actual vs predicted plots\n- `models/forecaster_model/uncertainty_zicatt.png` → Average uncertainty per category\n- `src/spending_forecaster.py` → Production forecaster module (ZICATT + ZICATTInference)

**Outcomes when you run:** Builds a per-user category-week matrix and sliding-window sequences **(1,800, 8, 11)** with ~**10.9%** zero targets. Train/val/test split **1,260 / 270 / 270**. ZICATT has **141,059** trainable parameters with temporal + cross-category attention. Training converges with early stopping; training history is saved. On the test set, gate accuracy is **88.3%** overall, with very high accuracy for stable categories (Food & Dining, Transportation) and lower for volatile ones (Travel, Education). Total-spend metrics: **MAE ≈ $1.7K**, **RMSE ≈ $2.2K**, **MAPE ≈ 39%**. The 95% prediction interval covers **94.5%** of actual category spends, showing well-calibrated uncertainty. Inference demo produces a realistic per-category breakdown and a one-week total forecast around **$5.6K**. The model checkpoint and upgraded `src/spending_forecaster.py` are ready for use by the API and Streamlit app.


*** Notebook 06 – LLM Integration ***

1. **Imports & Setup**  
   json, os, Path, datetime, numpy, pandas; anthropic (optional); PROJECT_ROOT (robust: cwd, or parent if in src, or spendwise-ai).

2. **Load Transaction Data & Models**  
   Load transactions_full.csv, label_mappings.json; print count and date range.

3. **Build Data Query Functions**  
   FinancialDataManager: get_spending_by_category, get_spending_trend, get_subscriptions, get_spending_summary, compare_to_average, _filter_user_and_dates. Init and test (spending by category, subscriptions, last month summary).

4. **Define Tools for Claude**  
   TOOLS list with name, description, input_schema for each function (5 tools).

5. **Build Financial Assistant Class**  
   FinancialAssistant(data_manager, api_key): use_api if key + anthropic; system_prompt; tool_functions map. chat() → _chat_with_api (tool loop) or _chat_demo (keyword responses).

6. **Test the Assistant**  
   Instantiate assistant; run test queries (subscriptions, spending by category, trend, summary).

7. **Interactive Chat Interface**  
   ChatSession(assistant, user_id): send(), get_history(), clear(). Demo conversation.

8. **Add Anomaly & Forecast Integration**  
   EnhancedFinancialAssistant: load anomaly_detector and forecaster from models if paths exist; check_anomalies(user_id), forecast_spending(user_id).

9. **Test Enhanced Assistant**  
   Anomaly status/score; forecast predicted and range or error.

10. **Save the LLM Module**  
    Notebook prints "Module saved to src/llm_assistant.py"; file src/llm_assistant.py has FinancialDataManager and FinancialAssistant.

**Outcomes when you run:** Data loaded **122,754** transactions; date range **2025-08-25** to **2026-02-21**. Testing Data Functions: spending by category (total and top 3), subscriptions (count and monthly total), last month (total and transaction count). **5** tools defined; Financial Assistant ready; demo mode if no API key. Test queries (subscriptions, spending by category, trend, summary) return keyword-based responses. ChatSession demo; Enhanced assistant loads anomaly detector and forecaster if models exist; anomaly status/score and forecast predicted/range. "Module saved to src/llm_assistant.py".

Key outputs: src/llm_assistant.py. Next: Notebook 07 (Recommendation Engine).


*** Notebook 07 – Recommendation Engine ***

1. **Imports & Setup**  
   PROJECT_ROOT (robust); json, Path, datetime, dataclass, Enum, pandas.

2. **Load Data**  
   transactions_full.csv; month, week, day_of_week, is_expense columns.

3. **Recommendation Data Structures**  
   Priority (HIGH, MEDIUM, LOW, POSITIVE), RecommendationType, Recommendation dataclass with to_dict(); sample rec.

4. **Build Analysis Functions**  
   SpendingAnalyzer: get_category_stats, get_subscription_analysis, get_frequency_analysis, get_income_expense_ratio. Test analyzer.

5. **Build Recommendation Engine**  
   _get_category_tips; RecommendationEngine: OVERSPEND_THRESHOLD, _check_overspending, _check_subscriptions, _check_high_frequency, _check_positive_trends, _check_budget_health; generate_recommendations (sorted by priority).

6. **Test Engine**  
   generate_recommendations for user_0001; print recs and total potential savings.

7. **Test Multiple Users**  
   Summary table: User, High, Medium, Low, Positive, Savings.

8. **Recommendation Service**  
   RecommendationService: get_recommendations, get_top_recommendation, get_savings_summary; test service.

9. **Format for Display**  
   format_recommendations_text; test on 3 recs.

10. **Save Module**  
    Print "Module saved to src/recommendation_engine.py"; file at src/recommendation_engine.py.

**Outcomes when you run:** Loaded **122,754** transactions. Testing Analyzer: category stats (top 5: last_month, mean, vs_average_pct); subscriptions count and $/month; high-frequency categories count; savings rate (%). Sample recommendation (JSON). Engine: N recommendations for user_0001; total potential savings $/month and $/year. Multi-user table: 10 users with High/Medium/Low/Positive counts and Savings. Service: get_recommendations(limit=5), get_savings_summary (potential_monthly_savings, by_category). Format test on 3 recs. "Module saved to src/recommendation_engine.py".

Key outputs: src/recommendation_engine.py. Next: Notebook 08 (Final Pipeline).


*** Notebook 08 – Final Pipeline & Streamlit App ***

1. **Imports & Setup**  
   sys, json, Path, datetime, numpy, pandas; PROJECT_ROOT (robust); sys.path.insert(src).

2. **Load All Components**  
   transactions_df; load classifier, anomaly_detector, forecaster, recommender, assistant (try/except); components_status; print status (no emoji).

3. **Build SpendWise API**  
   SpendWiseAPI: get_dashboard_data, get_spending_summary, get_spending_by_category, get_monthly_trend, get_recent_transactions, classify_transaction, check_anomaly, get_forecast, get_recommendations, chat. api = SpendWiseAPI(...).

4. **Test API**  
   get_dashboard_data for user_0001; print summary, categories, trend, anomaly, forecast, recommendations.

5. **Create Streamlit App**  
   app/streamlit_app.py path; app dir created. (App source in app/streamlit_app.py.)

6. **Requirements**  
   requirements.txt written (numpy, pandas, torch, streamlit, etc.).

7. **README**  
   README.md written (features, tech stack, quick start, structure).

8. **Project Report**  
   generate_project_report(); print and save PROJECT_REPORT.txt.

**Outcomes when you run:** Data loaded **122,754** transactions. Components Status: data **OK**; classifier, anomaly, forecaster, recommender, assistant **OK** or not available. SpendWise API initialized: **100** users, **12** categories. Test API (user_0001): dashboard summary (total_expenses, income, net, change_pct), top categories (amount, %), monthly trend average, anomaly status and score, forecast predicted/range or error, top recommendation titles. Streamlit app path; requirements.txt and README.md created; PROJECT_REPORT.txt saved (components, files, skills). Run: `streamlit run app/streamlit_app.py`.

Key outputs: app/streamlit_app.py, requirements.txt, README.md, PROJECT_REPORT.txt.


---

## After Notebook 08 – What We Did in the App

This section is in simple language. It covers what was added or fixed in the Streamlit app *after* the final pipeline (Notebook 08).

### Two modes in the app

When you run the app, you can choose:

1. **ML Showcase** – Uses the synthetic data from the notebooks. You pick a user and see Dashboard, Transactions, Analytics, Insights, AI Assistant, and Receipt Scanner.
2. **My Account** – Personal mode. You log in (e.g. username/password), add your own expenses, and use "My Assistant" for your personal spending.

So the app is split into: demo mode (ML Showcase) and real personal use (My Account).

### Receipt Scanner and the worker (why we use a subprocess)

- The **Receipt Scanner** page lets you upload a receipt image. The app uses the Donut model (from Notebook 02) to turn the image into text/JSON (items, total, etc.).
- On some machines (e.g. macOS), loading Donut (PyTorch/transformers) inside the same process as Streamlit can cause a crash (mutex/libc++abi error).
- **What we did:** We run the receipt parsing in a **separate process** (a small worker script). The Streamlit app calls this worker, gets the result back, and shows it. If the worker fails or times out, the app shows an error instead of crashing. So the scanner works reliably without breaking the main app.

In short: Receipt Scanner = upload image → worker runs Donut in another process → app gets parsed result and displays it.

### AI Assistant chat – same font, clean chatbot look

- In the **AI Assistant** (and in **My Assistant** under My Account), you type a question and the AI replies.
- Before: The reply could look messy or use different fonts because of markdown (bold, lists, etc.).
- **What we did:** We added CSS in the app so that *all* chat messages (your question and the AI’s reply) use the **same font** and size (system font, 1rem). Bold is still slightly bolder but same font family. So it looks like a simple, clean chatbot UI—no fancy or inconsistent fonts.

So after Notebook 08 we: added two modes (ML Showcase + My Account), made Receipt Scanner use a worker so it doesn’t crash the app, and made the AI chat look clean and consistent (same font as the question).