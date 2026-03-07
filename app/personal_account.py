"""
SpendWise AI - Personal Account Module
=======================================
Personal finance tracker with login, expense entry, and AI insights.
All data stored in CSV (no database needed).
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
PERSONAL_DATA_DIR = PROJECT_ROOT / "data" / "personal"
PERSONAL_CSV = PERSONAL_DATA_DIR / "my_transactions.csv"
USER_ID = "personal_user"

# Hardcoded credentials — change these to your own
CREDENTIALS = {
    "username": "kishan",
    "password": "spendwise2026"
}

CATEGORIES = [
    "Food & Dining", "Transportation", "Shopping", "Bills & Utilities",
    "Subscriptions", "Entertainment", "Health & Wellness", "Travel",
    "Education", "Personal Care", "Financial", "Income"
]

SUBCATEGORIES = {
    "Food & Dining": ["Restaurants", "Groceries", "Coffee Shops", "Fast Food", "Food Delivery"],
    "Transportation": ["Gas", "Rideshare", "Public Transit", "Parking", "Car Maintenance"],
    "Shopping": ["Amazon", "Clothing", "Electronics", "Home & Garden", "General"],
    "Bills & Utilities": ["Rent", "Electricity", "Water", "Internet", "Phone"],
    "Subscriptions": ["Streaming", "Software", "Gym", "News", "Other"],
    "Entertainment": ["Movies", "Games", "Music", "Events", "Hobbies"],
    "Health & Wellness": ["Doctor", "Pharmacy", "Dental", "Vision", "Fitness"],
    "Travel": ["Flights", "Hotels", "Car Rental", "Activities", "Food"],
    "Education": ["Tuition", "Books", "Courses", "Supplies", "Software"],
    "Personal Care": ["Haircut", "Skincare", "Clothing", "Laundry", "Other"],
    "Financial": ["Transfer", "Investment", "Fees", "Insurance", "Savings"],
    "Income": ["Salary", "Freelance", "Refund", "Gift", "Other"],
}


# ============================================================
# DATA MANAGEMENT
# ============================================================
def init_personal_data():
    """Create personal data directory and CSV if they don't exist."""
    PERSONAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PERSONAL_CSV.exists():
        df = pd.DataFrame(columns=[
            "date", "user_id", "merchant", "amount", "category", "subcategory"
        ])
        df.to_csv(PERSONAL_CSV, index=False)


def load_personal_transactions():
    """Load personal transactions."""
    init_personal_data()
    try:
        df = pd.read_csv(PERSONAL_CSV)
        if len(df) == 0:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["is_expense"] = df["amount"] < 0
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "date", "user_id", "merchant", "amount", "category", "subcategory"
        ])


def add_transaction(merchant, amount, category, subcategory, date=None):
    """Add a transaction to personal CSV."""
    init_personal_data()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    elif isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")

    new_row = pd.DataFrame([{
        "date": date,
        "user_id": USER_ID,
        "merchant": merchant,
        "amount": amount,
        "category": category,
        "subcategory": subcategory,
    }])

    if PERSONAL_CSV.exists() and PERSONAL_CSV.stat().st_size > 0:
        df = pd.read_csv(PERSONAL_CSV)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(PERSONAL_CSV, index=False)
    return True


# ============================================================
# ML MODEL LOADERS (reuse from main app cache)
# ============================================================
@st.cache_resource
def _load_classifier():
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from transaction_classifier import TransactionClassifierInference
        return TransactionClassifierInference(str(PROJECT_ROOT / "models/classifier_model"))
    except Exception:
        return None


@st.cache_resource
def _load_anomaly_detector():
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from anomaly_detector import AnomalyDetector
        return AnomalyDetector(str(PROJECT_ROOT / "models/anomaly_model"))
    except Exception:
        return None


@st.cache_resource
def _load_forecaster():
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from spending_forecaster import ZICATTInference
        return ZICATTInference(str(PROJECT_ROOT / "models/forecaster_model"))
    except Exception:
        return None


def _parse_price(raw: str) -> float:
    """Parse price string from Donut OCR output.
    
    Handles formats like:
    - "$18.99" → 18.99
    - "18,99" → 18.99  (comma as decimal)
    - "18.99" → 18.99
    - "1,234.56" → 1234.56  (comma as thousands)
    - "07:42PM" → 0  (not a price)
    - "0029" → 0  (not a price)
    """
    s = raw.strip().replace("$", "").replace(" ", "")
    
    # Skip obvious non-prices
    if not s or any(c in s for c in [":", "AM", "PM", "am", "pm"]):
        return 0.0
    
    # Handle comma as decimal separator or thousands separator
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            # "18,99" → decimal separator
            s = s.replace(",", ".")
        else:
            # "1,234" → thousands separator
            s = s.replace(",", "")
    elif "," in s and "." in s:
        # "1,234.56" → remove comma (thousands separator)
        s = s.replace(",", "")
    
    try:
        val = float(s)
        # Sanity check: receipt prices are typically $0.50 - $500
        if val < 0.01 or val > 5000:
            return 0.0
        return round(val, 2)
    except ValueError:
        return 0.0


def _render_receipt_entry_form(
    items: list,
    total: float,
    key_prefix: str,
    default_merchant: str = "",
    default_date: datetime | None = None,
):
    """Render the receipt entry form (used for both OCR success and manual fallback)."""
    if default_date is None:
        default_date = datetime.now()

    merchant = st.text_input(
        "Store name",
        value=default_merchant,
        key=f"{key_prefix}_receipt_merchant",
        placeholder="e.g. Royal Taste of India, Starbucks"
    )
    
    amount = st.number_input(
        "Total amount ($)",
        min_value=0.00,
        value=round(total, 2) if total > 0 else 0.00,
        step=0.01,
        key=f"{key_prefix}_receipt_total"
    )
    
    tx_date = st.date_input("Date", value=default_date, key=f"{key_prefix}_receipt_date")
    
    # Auto-classify
    classifier = _load_classifier()
    auto_cat = "Food & Dining"
    auto_subcat = "Restaurants"
    if classifier and merchant:
        cls_result = classifier.classify(merchant)
        auto_cat = cls_result["category"]
        auto_subcat = cls_result["subcategory"]
        st.info(f"Auto-classified: **{auto_cat}** > {auto_subcat} ({cls_result['category_confidence']*100:.1f}%)")
    elif classifier and items:
        classify_text = " ".join([i["name"] for i in items[:3]])
        cls_result = classifier.classify(classify_text)
        auto_cat = cls_result["category"]
        auto_subcat = cls_result["subcategory"]
        st.info(f"Auto-classified from items: **{auto_cat}** > {auto_subcat}")
    
    category = st.selectbox(
        "Category", CATEGORIES, 
        index=CATEGORIES.index(auto_cat) if auto_cat in CATEGORIES else 0, 
        key=f"{key_prefix}_receipt_cat"
    )
    subcats = SUBCATEGORIES.get(category, ["Other"])
    subcategory = st.selectbox(
        "Subcategory", subcats, 
        index=subcats.index(auto_subcat) if auto_subcat in subcats else 0, 
        key=f"{key_prefix}_receipt_subcat"
    )
    
    if merchant and amount > 0:
        if st.button("Add to My Transactions", key=f"{key_prefix}_receipt_add", type="primary"):
            add_transaction(merchant, -abs(amount), category, subcategory, tx_date)
            st.success(f"Added: {merchant} — ${amount:.2f} ({category})")
    elif not merchant:
        st.caption("Enter a store name to enable the Add button.")
    elif amount <= 0:
        st.caption("Enter an amount greater than $0 to enable the Add button.")


# ============================================================
# LOGIN
# ============================================================
def render_login():
    """Render login screen. Returns True if logged in."""
    if st.session_state.get("logged_in", False):
        return True

    st.markdown("---")
    st.markdown(
        "<div style='max-width:400px; margin:60px auto; padding:40px; "
        "border-radius:16px; box-shadow:0 4px 24px rgba(0,0,0,0.08); "
        "background:white;'>"
        "<h2 style='text-align:center; margin-bottom:8px;'>Login</h2>"
        "<p style='text-align:center; color:#888; margin-bottom:24px;'>"
        "Sign in to your personal account</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Sign In", type="primary", width="stretch"):
            if username == CREDENTIALS["username"] and password == CREDENTIALS["password"]:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown(
            "<p style='text-align:center; color:#aaa; font-size:0.85rem; margin-top:16px;'>"
            "Demo credentials — Username: <b>kishan</b> | Password: <b>spendwise2026</b></p>",
            unsafe_allow_html=True,
        )

    return False


# ============================================================
# PERSONAL DASHBOARD
# ============================================================
def render_personal_dashboard():
    """Personal financial dashboard."""
    st.title("My Dashboard")

    df = load_personal_transactions()

    if len(df) == 0:
        st.info("No transactions yet. Go to **Add Expense** to start tracking!")
        return

    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()
    income_df = df[df["amount"] > 0]

    total_expenses = expenses["abs_amount"].sum()
    total_income = income_df["amount"].sum()
    net = total_income - total_expenses
    tx_count = len(df)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Expenses", f"${total_expenses:,.2f}")
    c2.metric("Total Income", f"${total_income:,.2f}")
    c3.metric("Net Balance", f"${net:,.2f}", "Positive" if net >= 0 else "Negative")
    c4.metric("Transactions", tx_count)

    st.markdown("---")

    if len(expenses) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Spending by Category")
            by_cat = expenses.groupby("category")["abs_amount"].sum().sort_values(ascending=False)
            fig = px.pie(
                values=by_cat.values, names=by_cat.index, hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Spending Over Time")
            daily = expenses.groupby(expenses["date"].dt.date)["abs_amount"].sum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(daily.index), y=list(daily.values),
                mode="lines+markers", fill="tozeroy", line=dict(color="#667eea"),
            ))
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Spending ($)",
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, width="stretch")

    # Recent transactions
    st.subheader("Recent Transactions")
    recent = df.sort_values("date", ascending=False).head(10).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["amount"] = recent["amount"].apply(
        lambda x: f"{'+' if x > 0 else '-'} ${abs(x):,.2f}"
    )
    st.dataframe(
        recent[["date", "merchant", "amount", "category", "subcategory"]],
        width="stretch", hide_index=True,
    )


def _parse_receipt_with_claude(image_bytes: bytes) -> tuple[dict | None, str | None]:
    """Parse receipt using Claude Vision API.
    Returns (result_dict, error_message). Error is None on success."""
    try:
        import anthropic
        import base64
    except ImportError:
        return None, "anthropic package not installed"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set in .env"

    client = anthropic.Anthropic(api_key=api_key)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Detect mime type
    if image_bytes[:3] == b'\xff\xd8\xff':
        media_type = "image/jpeg"
    elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        media_type = "image/png"
    else:
        media_type = "image/jpeg"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract ALL information from this receipt. Return ONLY valid JSON with this exact structure, no other text:
{
    "store_name": "store name",
    "date": "YYYY-MM-DD",
    "items": [
        {"name": "item name", "quantity": 1, "price": 10.99}
    ],
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00
}
Rules:
- price must be a number (not string), e.g. 10.99 not "$10.99"
- total should be the final amount paid (including tax)
- If you can't read a value, use 0
- Include ALL line items from the receipt"""
                    }
                ]
            }]
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json.loads(text), None

    except Exception as e:
        return None, str(e)


# ============================================================
# ADD EXPENSE — THREE METHODS
# ============================================================
def render_add_expense():
    """Three ways to add expenses."""
    st.title("Add Expense")
    st.markdown("Choose how to add your transaction:")

    tab1, tab2, tab3 = st.tabs(["Scan Receipt", "Quick Text", "Manual Entry"])

    # ---- TAB 1: Receipt Upload ----
    with tab1:
        st.markdown("Upload a receipt photo — AI extracts items and auto-classifies.")

        uploaded = st.file_uploader("Upload receipt", type=["png", "jpg", "jpeg"], key="personal_receipt")

        if uploaded:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded, caption="Uploaded Receipt", use_container_width=True)

            with col2:
                image_bytes = uploaded.getvalue()
                parsed_data = None
                parse_method = None

                with st.spinner("🔍 Scanning receipt with AI..."):

                    # Method 1: Try Claude Vision first (best quality)
                    claude_result, claude_error = _parse_receipt_with_claude(image_bytes)

                    if claude_result and claude_result.get("items"):
                        parsed_data = claude_result
                        parse_method = "claude"
                    elif claude_error:
                        # Store error to show user why Claude Vision failed
                        st.session_state["_claude_vision_error"] = claude_error
                    else:
                        # Method 2: Fall back to Donut OCR
                        worker = PROJECT_ROOT / "app" / "parse_receipt_worker.py"
                        if worker.exists():
                            with tempfile.NamedTemporaryFile(
                                suffix=Path(uploaded.name).suffix, delete=False
                            ) as tmp:
                                tmp.write(image_bytes)
                                tmp_path = tmp.name

                            try:
                                proc = subprocess.run(
                                    [sys.executable, str(worker), tmp_path],
                                    capture_output=True, text=True, timeout=120,
                                    cwd=str(PROJECT_ROOT),
                                )
                                Path(tmp_path).unlink(missing_ok=True)

                                stdout = proc.stdout.strip()
                                if stdout:
                                    try:
                                        donut_data = json.loads(stdout)
                                        parsed_data = donut_data
                                        parse_method = "donut"
                                    except json.JSONDecodeError:
                                        pass
                            except subprocess.TimeoutExpired:
                                try:
                                    Path(tmp_path).unlink(missing_ok=True)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                # --- Process results ---
                if parsed_data and parse_method == "claude":
                    st.success("✅ Parsed with Claude Vision AI")

                    store_name = parsed_data.get("store_name", "")
                    receipt_date = parsed_data.get("date", "")
                    items = parsed_data.get("items", [])
                    subtotal = parsed_data.get("subtotal", 0) or 0
                    tax = parsed_data.get("tax", 0) or 0
                    total = parsed_data.get("total", 0) or 0

                    if items:
                        st.markdown("**Items found:**")
                        for it in items:
                            qty = it.get("quantity", 1) or 1
                            price = it.get("price", 0) or 0
                            name = it.get("name", "Unknown")
                            if qty > 1:
                                st.markdown(f"- {name} x{qty}: ${price:.2f}")
                            else:
                                st.markdown(f"- {name}: ${price:.2f}")

                    if subtotal > 0:
                        st.markdown(f"Subtotal: ${subtotal:.2f}")
                    if tax > 0:
                        st.markdown(f"Tax: ${tax:.2f}")
                    if total > 0:
                        st.success(f"**Total: ${total:.2f}**")
                    elif subtotal > 0:
                        total = subtotal + tax
                        st.success(f"**Total: ${total:.2f}**")

                    # Parse date
                    default_date = datetime.now()
                    if receipt_date:
                        try:
                            default_date = datetime.strptime(receipt_date, "%Y-%m-%d")
                        except ValueError:
                            pass

                    _render_receipt_entry_form(
                        items=[{"name": i.get("name", ""), "price": i.get("price", 0)} for i in items],
                        total=total if total > 0 else subtotal + tax,
                        key_prefix="claude_ocr",
                        default_merchant=store_name,
                        default_date=default_date,
                    )

                elif parsed_data and parse_method == "donut":
                    claude_err = st.session_state.pop("_claude_vision_error", None)
                    if claude_err:
                        if "credit balance" in claude_err.lower():
                            st.warning("⚠️ Claude Vision unavailable (no API credits). Using Donut OCR fallback. Add credits at console.anthropic.com/settings/billing")
                        else:
                            st.warning(f"⚠️ Claude Vision failed: {claude_err[:150]}. Using Donut OCR fallback.")
                    else:
                        st.info("📋 Parsed with Donut OCR (fallback)")

                    with st.expander("Raw OCR output", expanded=False):
                        st.json(parsed_data)

                    items = []
                    if "menu" in parsed_data:
                        for item in parsed_data["menu"]:
                            name = item.get("nm", item.get("name", ""))
                            if isinstance(name, dict):
                                name = name.get("value", name.get("nm", str(name)))
                            if not name or not isinstance(name, str):
                                continue
                            name_lower = name.lower()
                            if any(skip in name_lower for skip in ["server", "cashier", "table", "open time", "date", "time", "thank"]):
                                continue
                            price_raw = item.get("unitprice", item.get("price", "0"))
                            price_val = _parse_price(str(price_raw))
                            if price_val <= 0 or price_val > 500:
                                continue
                            items.append({"name": name, "price": price_val})

                    receipt_total = None
                    for key in ["total", "sub_total"]:
                        if key in parsed_data and receipt_total is None:
                            val = parsed_data[key]
                            if isinstance(val, dict):
                                for sub_key in ["total_price", "total_etc", "subtotal_price", "sub_total_price"]:
                                    if sub_key in val:
                                        receipt_total = _parse_price(str(val[sub_key]))
                                        if receipt_total and receipt_total > 0:
                                            break
                            else:
                                receipt_total = _parse_price(str(val))

                    total = receipt_total if receipt_total and receipt_total > 0 else sum(i["price"] for i in items) if items else 0

                    if items:
                        st.markdown("**Items found:**")
                        for it in items:
                            st.markdown(f"- {it['name']}: ${it['price']:.2f}")
                    if total > 0:
                        st.success(f"**Receipt Total: ${total:.2f}**")

                    _render_receipt_entry_form(items, total, "donut_ocr")

                else:
                    claude_err = st.session_state.pop("_claude_vision_error", None)
                    if claude_err:
                        st.warning(f"📷 Claude Vision: {claude_err[:150]}")
                    st.warning("Could not read this receipt. Enter details manually:")
                    _render_receipt_entry_form([], 0, "manual_fallback")

    # ---- TAB 2: Quick Text ----
    with tab2:
        st.markdown("Type a transaction description — AI auto-classifies it.")
        st.markdown("*Examples: `STARBUCKS $5.75`, `UBER TRIP $12.50`, `Salary deposit`*")

        text_input = st.text_input("Transaction description", placeholder="STARBUCKS #1234 $5.75", key="quick_text")
        amount_input = st.number_input("Amount ($)", min_value=0.01, value=10.00, step=0.01, key="quick_amount")
        tx_type = st.radio("Type", ["Expense", "Income"], horizontal=True, key="quick_type")
        tx_date = st.date_input("Date", value=datetime.now(), key="quick_date")

        if text_input:
            classifier = _load_classifier()
            if classifier and tx_type == "Expense":
                result = classifier.classify(text_input)
                auto_cat = result["category"]
                auto_subcat = result["subcategory"]
                conf = result["category_confidence"]
                st.info(f"Auto-classified: **{auto_cat}** > {auto_subcat} ({conf*100:.1f}% confidence)")
            else:
                auto_cat = "Income" if tx_type == "Income" else "Shopping"
                auto_subcat = "Salary" if tx_type == "Income" else "General"

            category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(auto_cat) if auto_cat in CATEGORIES else 0, key="quick_cat")
            subcats = SUBCATEGORIES.get(category, ["Other"])
            subcategory = st.selectbox("Subcategory", subcats, index=subcats.index(auto_subcat) if auto_subcat in subcats else 0, key="quick_subcat")

            if st.button("Add Transaction", key="quick_add", type="primary"):
                final_amount = amount_input if tx_type == "Income" else -abs(amount_input)
                merchant = text_input.split("$")[0].strip() if "$" in text_input else text_input
                add_transaction(merchant, final_amount, category, subcategory, tx_date)
                st.success(f"Added: {merchant} — ${amount_input:.2f} ({category})")
                st.balloons()

    # ---- TAB 3: Manual Form ----
    with tab3:
        st.markdown("Enter transaction details manually.")

        merchant = st.text_input("Merchant / Description", placeholder="e.g. Walmart, Netflix, Salary", key="manual_merchant")
        amount = st.number_input("Amount ($)", min_value=0.01, value=25.00, step=0.01, key="manual_amount")
        tx_type = st.radio("Type", ["Expense", "Income"], horizontal=True, key="manual_type")
        tx_date = st.date_input("Date", value=datetime.now(), key="manual_date")
        category = st.selectbox("Category", CATEGORIES, key="manual_cat")
        subcats = SUBCATEGORIES.get(category, ["Other"])
        subcategory = st.selectbox("Subcategory", subcats, key="manual_subcat")

        if merchant and st.button("Add Transaction", key="manual_add", type="primary"):
            final_amount = amount if tx_type == "Income" else -abs(amount)
            add_transaction(merchant, final_amount, category, subcategory, tx_date)
            st.success(f"Added: {merchant} — ${amount:.2f} ({category})")
            st.balloons()


# ============================================================
# MY TRANSACTIONS
# ============================================================
def render_my_transactions():
    """View and manage personal transactions."""
    st.title("My Transactions")

    df = load_personal_transactions()

    if len(df) == 0:
        st.info("No transactions yet. Go to **Add Expense** to start!")
        return

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        cats = ["All"] + sorted(df["category"].unique().tolist())
        sel_cat = st.selectbox("Category", cats, key="my_tx_cat")
    with c2:
        sel_type = st.selectbox("Type", ["All", "Expenses", "Income"], key="my_tx_type")
    with c3:
        sel_sort = st.selectbox("Sort", ["Newest", "Oldest", "Highest", "Lowest"], key="my_tx_sort")

    filtered = df.copy()
    if sel_cat != "All":
        filtered = filtered[filtered["category"] == sel_cat]
    if sel_type == "Expenses":
        filtered = filtered[filtered["amount"] < 0]
    elif sel_type == "Income":
        filtered = filtered[filtered["amount"] > 0]

    if sel_sort == "Newest":
        filtered = filtered.sort_values("date", ascending=False)
    elif sel_sort == "Oldest":
        filtered = filtered.sort_values("date", ascending=True)
    elif sel_sort == "Highest":
        filtered = filtered.sort_values("amount", key=abs, ascending=False)
    else:
        filtered = filtered.sort_values("amount", key=abs, ascending=True)

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(filtered))
    c2.metric("Expenses", f"${filtered[filtered['amount'] < 0]['amount'].abs().sum():,.2f}")
    c3.metric("Income", f"${filtered[filtered['amount'] > 0]['amount'].sum():,.2f}")

    st.markdown("---")

    display = filtered[["date", "merchant", "amount", "category", "subcategory"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["amount"] = display["amount"].apply(
        lambda x: f"{'+' if x > 0 else '-'} ${abs(x):,.2f}"
    )
    st.dataframe(display, width="stretch", hide_index=True)

    # Delete all option
    st.markdown("---")
    with st.expander("Danger Zone"):
        if st.button("Delete ALL My Transactions", type="secondary"):
            PERSONAL_CSV.unlink(missing_ok=True)
            init_personal_data()
            st.success("All transactions deleted.")
            st.rerun()


# ============================================================
# MY AI ASSISTANT
# ============================================================
def render_my_assistant():
    """AI assistant on personal data."""
    st.title("My AI Assistant")
    st.markdown("Ask me anything about *your* spending!")

    df = load_personal_transactions()

    if len(df) == 0:
        st.info("Add some transactions first, then I can answer your questions!")
        return

    # Load assistant with personal data
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from llm_assistant import FinancialDataManager, FinancialAssistant
        data_manager = FinancialDataManager(df)
        assistant = FinancialAssistant(data_manager)
    except Exception:
        assistant = None

    # Mode indicator: Claude API vs demo
    if assistant and getattr(assistant, "use_api", False):
        st.caption("Powered by Claude")
    else:
        st.caption("Demo Mode")

    # Chat
    if "personal_messages" not in st.session_state:
        st.session_state.personal_messages = [
            {"role": "assistant", "content": "Hi! Ask me about your personal spending, subscriptions, or trends."}
        ]

    for msg in st.session_state.personal_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["content"].replace("$", "\\$"))
            else:
                st.text(msg["content"])

    if prompt := st.chat_input("Ask about your spending...", key="personal_chat"):
        st.session_state.personal_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.text(prompt)

        if assistant:
            response = assistant.chat(prompt, USER_ID)
        else:
            response = "Assistant not available."

        st.session_state.personal_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response.replace("$", "\\$"))

    st.markdown("---")
    st.markdown("**Quick Questions:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("My Summary", key="pa_summary"):
            st.session_state.personal_messages.append({"role": "user", "content": "Give me a spending summary"})
            st.rerun()
    with c2:
        if st.button("By Category", key="pa_cat"):
            st.session_state.personal_messages.append({"role": "user", "content": "Show spending by category"})
            st.rerun()
    with c3:
        if st.button("Trend", key="pa_trend"):
            st.session_state.personal_messages.append({"role": "user", "content": "What's my spending trend?"})
            st.rerun()
    with c4:
        if st.button("Clear", key="pa_clear"):
            st.session_state.personal_messages = [
                {"role": "assistant", "content": "Chat cleared! How can I help?"}
            ]
            st.rerun()


# ============================================================
# MY INSIGHTS
# ============================================================
def render_my_insights():
    """Personal anomaly detection and forecast."""
    st.title("My Insights")

    df = load_personal_transactions()
    if len(df) == 0:
        st.info("Add transactions to unlock AI-powered insights!")
        return

    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()

    col1, col2 = st.columns(2)

    # Anomaly Detection
    with col1:
        st.subheader("Anomaly Detection")
        detector = _load_anomaly_detector()

        if detector and len(expenses) > 0:
            by_cat = expenses.groupby("category")["abs_amount"].sum()
            spending = {cat: amt for cat, amt in by_cat.items()}

            try:
                result = detector.detect(spending)
                score = result.get("anomaly_score", 0)
                is_anomaly = result.get("is_anomaly", False)

                if is_anomaly:
                    st.error("Unusual spending pattern detected.")
                else:
                    st.success("Your spending looks normal.")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Anomaly Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#667eea"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
                    },
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=0))
                st.plotly_chart(fig, width="stretch")

                if result.get("top_anomalous_categories"):
                    st.markdown("**Top unusual categories:**")
                    for cat_info in result["top_anomalous_categories"]:
                        st.markdown(f"- {cat_info['category']}: score {cat_info['contribution']:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Need more transactions for anomaly detection.")

    # Forecast
    with col2:
        st.subheader("Spending Forecast")
        forecaster = _load_forecaster()

        if forecaster and len(expenses) > 0:
            if hasattr(forecaster, 'categories'):
                expenses["week"] = pd.to_datetime(expenses["date"]).dt.to_period("W")
                
                spending_history = {}
                for cat in forecaster.categories:
                    cat_data = expenses[expenses["category"] == cat]
                    weekly = cat_data.groupby("week")["abs_amount"].sum()
                    spending_history[cat] = weekly.tail(12).tolist()
                
                min_weeks = min(len(v) for v in spending_history.values()) if spending_history else 0
                
                if min_weeks >= 8:
                    try:
                        result = forecaster.predict(spending_history)

                        current_week = expenses.groupby("week")["abs_amount"].sum()
                        current_total = current_week.iloc[-1] if len(current_week) > 0 else 0
                        
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("This Week", f"${current_total:,.2f}")
                        with m2:
                            pred_change = ((result['predicted_total'] - current_total) / current_total * 100) if current_total > 0 else 0
                            st.metric(
                                "Next Week (Predicted)",
                                f"${result['predicted_total']:,.2f}",
                                f"{pred_change:+.1f}% vs this week"
                            )
                        
                        st.caption(f"Range: ${result['total_lower_bound']:,.2f} – ${result['total_upper_bound']:,.2f}")
                        
                        if 'per_category' in result:
                            st.markdown("**Per-category:**")
                            sorted_cats = sorted(
                                result['per_category'].items(),
                                key=lambda x: x[1]['expected_spending'],
                                reverse=True
                            )
                            for cat, pred in sorted_cats[:5]:
                                prob = pred['probability']
                                expected = pred['expected_spending']
                                prob_icon = "🟢" if prob > 0.8 else "🟡" if prob > 0.5 else "🔴"
                                st.text(f"{prob_icon} {cat}: ${expected:,.0f}  ({prob*100:.0f}% likely)")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info(f"Need 8+ weeks per category. Current: {min_weeks} weeks.")
            else:
                expenses["week"] = pd.to_datetime(expenses["date"]).dt.to_period("W")
                weekly = expenses.groupby("week")["abs_amount"].sum()
                history = weekly.tail(12).tolist()
                
                if len(history) >= 8:
                    try:
                        result = forecaster.predict(history)
                        st.metric(
                            "Predicted Next Week",
                            f"${result.get('predicted_spending', result.get('predicted_total', 0)):,.2f}",
                            f"Range: ${result.get('lower_bound', result.get('total_lower_bound', 0)):,.2f} - ${result.get('upper_bound', result.get('total_upper_bound', 0)):,.2f}",
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info(f"Need 8+ weeks of data. Current: {len(history)} weeks.")
        else:
            st.info("Need more transactions for forecasting.")

    # Classifier demo
    st.markdown("---")
    st.subheader("Test Classifier")
    test_text = st.text_input("Type a transaction to classify:", placeholder="UBER *TRIP $12.50", key="my_classify")
    if test_text:
        classifier = _load_classifier()
        if classifier:
            result = classifier.classify(test_text)
            c1, c2 = st.columns(2)
            c1.metric("Category", result["category"], f"{result['category_confidence']*100:.1f}%")
            c2.metric("Subcategory", result["subcategory"], f"{result['subcategory_confidence']*100:.1f}%")


# ============================================================
# MAIN: PERSONAL ACCOUNT ROUTER
# ============================================================
def render_personal_account():
    """Main entry point for personal account section."""

    # Check login
    if not render_login():
        return

    # Logged in — show personal sidebar nav
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{st.session_state.get('username', 'User')}**")

    personal_page = st.sidebar.radio(
        "My Account",
        [
            "My Dashboard",
            "Add Expense",
            "My Transactions",
            "My Assistant",
            "My Insights",
        ],
        key="personal_nav",
    )

    if st.sidebar.button("Logout", key="logout_btn"):
        st.session_state["logged_in"] = False
        st.session_state.pop("username", None)
        st.rerun()

    # Route
    if personal_page == "My Dashboard":
        render_personal_dashboard()
    elif personal_page == "Add Expense":
        render_add_expense()
    elif personal_page == "My Transactions":
        render_my_transactions()
    elif personal_page == "My Assistant":
        render_my_assistant()
    elif personal_page == "My Insights":
        render_my_insights()

