"""
SpendWise AI - Streamlit Dashboard
===================================

Run with: streamlit run app/streamlit_app.py
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from personal_account import render_personal_account

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SpendWise AI",
    page_icon="SW",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Recommendation cards */
    .rec-high {
        border-left: 4px solid #ff4444;
        background: #fff5f5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .rec-medium {
        border-left: 4px solid #ffaa00;
        background: #fffaf0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .rec-low {
        border-left: 4px solid #44aa44;
        background: #f0fff0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .rec-positive {
        border-left: 4px solid #4488ff;
        background: #f0f8ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* AI Assistant chat – same font as question, clean chatbot UI */
    [data-testid="stChatMessage"] div[data-testid="stMarkdown"] p,
    [data-testid="stChatMessage"] div[data-testid="stMarkdown"] li,
    [data-testid="stChatMessage"] div[data-testid="stMarkdown"] strong,
    [data-testid="stChatMessage"] div[data-testid="stMarkdown"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif !important;
        font-size: 1rem !important;
        font-weight: 400;
        line-height: 1.5;
    }
    [data-testid="stChatMessage"] div[data-testid="stMarkdown"] strong {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_transactions():
    """Load transaction data"""
    try:
        df = pd.read_csv(PROJECT_ROOT / "data/synthetic/transactions_full.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        df['week'] = df['date'].dt.to_period('W').astype(str)
        df['is_expense'] = df['amount'] < 0
        return df
    except FileNotFoundError:
        st.error("Transaction data not found. Please run Notebook 01 first.")
        return None

@st.cache_data
def load_label_mappings():
    """Load label mappings"""
    try:
        with open(PROJECT_ROOT / "data/processed/label_mappings.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ============================================================
# LOAD ML MODELS
# ============================================================
@st.cache_resource
def load_classifier():
    """Load transaction classifier"""
    try:
        from transaction_classifier import TransactionClassifierInference
        return TransactionClassifierInference(str(PROJECT_ROOT / "models/classifier_model"))
    except Exception as e:
        return None

@st.cache_resource
def load_anomaly_detector():
    """Load anomaly detector"""
    try:
        from anomaly_detector import AnomalyDetector
        return AnomalyDetector(str(PROJECT_ROOT / "models/anomaly_model"))
    except Exception as e:
        return None

@st.cache_resource
def load_forecaster():
    """Load spending forecaster (ZICATT)"""
    try:
        from spending_forecaster import ZICATTInference
        return ZICATTInference(str(PROJECT_ROOT / "models/forecaster_model"))
    except Exception as e:
        return None

@st.cache_resource
def load_recommender(_df):
    """Load recommendation engine"""
    try:
        from recommendation_engine import RecommendationService
        return RecommendationService(_df)
    except Exception as e:
        return None

@st.cache_resource
def load_assistant(_df):
    """Load LLM assistant"""
    try:
        from llm_assistant import FinancialDataManager, FinancialAssistant
        data_manager = FinancialDataManager(_df)
        return FinancialAssistant(data_manager)
    except Exception as e:
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_spending_summary(df, user_id, days=30):
    """Get spending summary for a user"""
    user_df = df[df['user_id'] == user_id]
    max_date = user_df['date'].max()
    
    # Current period
    current = user_df[user_df['date'] > max_date - timedelta(days=days)]
    current_expenses = current[current['is_expense']]['amount'].abs().sum()
    current_income = current[~current['is_expense']]['amount'].sum()
    current_count = len(current[current['is_expense']])
    
    # Previous period
    prev_start = max_date - timedelta(days=days*2)
    prev_end = max_date - timedelta(days=days)
    previous = user_df[(user_df['date'] > prev_start) & (user_df['date'] <= prev_end)]
    prev_expenses = previous[previous['is_expense']]['amount'].abs().sum()
    
    # Change
    change_pct = ((current_expenses - prev_expenses) / prev_expenses * 100) if prev_expenses > 0 else 0
    
    return {
        'expenses': current_expenses,
        'income': current_income,
        'net': current_income - current_expenses,
        'transactions': current_count,
        'daily_avg': current_expenses / days,
        'change_pct': change_pct
    }

def get_spending_by_category(df, user_id, days=30):
    """Get spending breakdown by category"""
    user_df = df[df['user_id'] == user_id]
    max_date = user_df['date'].max()
    
    recent = user_df[user_df['date'] > max_date - timedelta(days=days)]
    expenses = recent[recent['is_expense']].copy()
    expenses['amount'] = expenses['amount'].abs()
    
    by_cat = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
    total = by_cat.sum()
    
    result = []
    for cat, amount in by_cat.items():
        result.append({
            'category': cat,
            'amount': amount,
            'percentage': (amount / total * 100) if total > 0 else 0
        })
    return result

def get_monthly_trend(df, user_id, months=6):
    """Get monthly spending trend"""
    user_df = df[df['user_id'] == user_id]
    expenses = user_df[user_df['is_expense']].copy()
    expenses['amount'] = expenses['amount'].abs()
    
    monthly = expenses.groupby('month')['amount'].sum().tail(months)
    return monthly

def get_recent_transactions(df, user_id, limit=20):
    """Get recent transactions"""
    user_df = df[df['user_id'] == user_id].copy()
    return user_df.sort_values('date', ascending=False).head(limit)

# ============================================================
# DASHBOARD PAGE
# ============================================================
def render_dashboard(df, user_id, days):
    """Render main dashboard"""
    st.title("Financial Dashboard")
    st.markdown(f"**User:** `{user_id}` | **Period:** Last {days} days")
    
    # Summary metrics
    summary = get_spending_summary(df, user_id, days)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Expenses",
            f"${summary['expenses']:,.2f}",
            f"{summary['change_pct']:+.1f}% vs prev"
        )
    
    with col2:
        st.metric(
            "Total Income",
            f"${summary['income']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Net Balance",
            f"${summary['net']:,.2f}",
            "Positive" if summary['net'] > 0 else "Negative"
        )
    
    with col4:
        st.metric(
            "Transactions",
            f"{summary['transactions']}"
        )
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spending by Category")
        by_cat = get_spending_by_category(df, user_id, days)
        
        if by_cat:
            fig = px.pie(
                pd.DataFrame(by_cat),
                values='amount',
                names='category',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Monthly Trend")
        monthly = get_monthly_trend(df, user_id)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(monthly.index),
            y=list(monthly.values),
            marker_color='#667eea'
        ))
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Spending ($)",
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("Smart Recommendations")
    
    recommender = load_recommender(df)
    if recommender:
        try:
            recs = recommender.get_recommendations(user_id, limit=4)
            
            if recs and recs.get('recommendations'):
                cols = st.columns(2)
                for i, rec in enumerate(recs['recommendations'][:4]):
                    with cols[i % 2]:
                        priority = rec.get('priority', 'low')
                        label_prefix = {
                            'high': '[HIGH]',
                            'medium': '[MEDIUM]',
                            'low': '[LOW]',
                            'positive': '[POSITIVE]'
                        }.get(priority, '')
                        title = f"{label_prefix} {rec['title']}".strip()
                        
                        with st.expander(title, expanded=(i < 2)):
                            st.write(rec['description'])
                            if rec.get('potential_savings', 0) > 0:
                                st.success(f"Potential savings: ${rec['potential_savings']:.2f}/month")
                            if rec.get('action_items'):
                                st.markdown("**Actions:**")
                                for action in rec['action_items'][:2]:
                                    st.markdown(f"- {action}")
            else:
                st.info("No recommendations at this time. Keep up the good work!")
        except Exception as e:
            st.warning(f"Could not load recommendations: {e}")
    else:
        st.info("Recommendation engine not available. Run Notebook 07 first.")

# ============================================================
# TRANSACTIONS PAGE
# ============================================================
def render_transactions(df, user_id, days):
    """Render transactions page"""
    st.title("Transaction History")
    st.markdown(f"**User:** `{user_id}` | **Period:** Last {days} days")
    
    user_df = df[df['user_id'] == user_id].copy()
    max_date = user_df['date'].max()
    user_df = user_df[user_df['date'] > max_date - timedelta(days=days)]
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = ['All'] + sorted(user_df['category'].unique().tolist())
        selected_cat = st.selectbox("Category", categories)
    
    with col2:
        tx_type = st.selectbox("Type", ['All', 'Expenses', 'Income'])
    
    with col3:
        sort_order = st.selectbox("Sort", ['Newest First', 'Oldest First', 'Highest Amount', 'Lowest Amount'])
    
    # Apply filters
    filtered = user_df.copy()
    
    if selected_cat != 'All':
        filtered = filtered[filtered['category'] == selected_cat]
    
    if tx_type == 'Expenses':
        filtered = filtered[filtered['amount'] < 0]
    elif tx_type == 'Income':
        filtered = filtered[filtered['amount'] > 0]
    
    # Sort
    if sort_order == 'Newest First':
        filtered = filtered.sort_values('date', ascending=False)
    elif sort_order == 'Oldest First':
        filtered = filtered.sort_values('date', ascending=True)
    elif sort_order == 'Highest Amount':
        filtered['abs_amount'] = filtered['amount'].abs()
        filtered = filtered.sort_values('abs_amount', ascending=False)
    else:
        filtered['abs_amount'] = filtered['amount'].abs()
        filtered = filtered.sort_values('abs_amount', ascending=True)
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(filtered))
    with col2:
        expenses = filtered[filtered['amount'] < 0]['amount'].abs().sum()
        st.metric("Total Expenses", f"${expenses:,.2f}")
    with col3:
        income = filtered[filtered['amount'] > 0]['amount'].sum()
        st.metric("Total Income", f"${income:,.2f}")
    
    st.markdown("---")
    
    # Transaction table
    display_df = filtered[['date', 'merchant', 'amount', 'category', 'subcategory']].head(100).copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

# ============================================================
# ANALYTICS PAGE
# ============================================================
def render_analytics(df, user_id, days):
    """Render analytics page"""
    st.title("Spending Analytics")
    st.markdown(f"**User:** `{user_id}` | **Period:** Last {days} days")
    
    user_df = df[df['user_id'] == user_id]
    max_date = user_df['date'].max()
    user_df = user_df[user_df['date'] > max_date - timedelta(days=days)]
    expenses = user_df[user_df['is_expense']].copy()
    expenses['amount'] = expenses['amount'].abs()
    
    # Top spending categories
    st.subheader("Top Spending Categories")
    by_cat = get_spending_by_category(df, user_id, days)
    
    if by_cat:
        fig = px.bar(
            pd.DataFrame(by_cat).head(10),
            x='amount',
            y='category',
            orientation='h',
            color='amount',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Spending by day of week
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spending by Day of Week")
        expenses['day_of_week'] = pd.to_datetime(expenses['date']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        by_day = expenses.groupby('day_of_week')['amount'].sum().reindex(day_order)
        
        fig = px.bar(x=by_day.index, y=by_day.values, color=by_day.values,
                     color_continuous_scale='Blues')
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Total Spending ($)",
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.subheader("Spending Over Time")
        daily = expenses.groupby(expenses['date'].dt.date)['amount'].sum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(daily.index),
            y=list(daily.values),
            mode='lines',
            fill='tozeroy',
            line=dict(color='#667eea')
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily Spending ($)"
        )
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Subcategory breakdown
    st.subheader("Subcategory Breakdown")
    
    selected_category = st.selectbox(
        "Select Category",
        sorted(expenses['category'].unique())
    )
    
    cat_expenses = expenses[expenses['category'] == selected_category]
    by_subcat = cat_expenses.groupby('subcategory')['amount'].sum().sort_values(ascending=False)
    
    fig = px.pie(
        values=by_subcat.values,
        names=by_subcat.index,
        hole=0.3
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================
# INSIGHTS PAGE (AI Models)
# ============================================================
def render_insights(df, user_id, days):
    """Render AI insights page"""
    st.title("AI-Powered Insights")
    
    col1, col2 = st.columns(2)
    
    # Anomaly Detection
    with col1:
        st.subheader("Anomaly Detection")
        
        anomaly_detector = load_anomaly_detector()
        
        if anomaly_detector:
            by_cat = get_spending_by_category(df, user_id, days)
            spending = {item['category']: item['amount'] for item in by_cat}
            
            try:
                result = anomaly_detector.detect(spending)
                score = result.get('anomaly_score', 0)
                is_anomaly = result.get('is_anomaly', False)
                
                # Status
                if is_anomaly:
                    st.error("Unusual spending detected.")
                else:
                    st.success("Spending looks normal.")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Anomaly Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d4edda"},
                            {'range': [30, 70], 'color': "#fff3cd"},
                            {'range': [70, 100], 'color': "#f8d7da"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=0))
                st.plotly_chart(fig, width="stretch")
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Anomaly detector not available. Run Notebook 04 first.")
    
    # Spending Forecast
    with col2:
        st.subheader("Spending Forecast")
        
        forecaster = load_forecaster()
        
        if forecaster:
            user_df = df[df['user_id'] == user_id]
            expenses = user_df[user_df['is_expense']].copy()
            expenses['amount'] = expenses['amount'].abs()
            
            # Build per-category weekly history
            if hasattr(forecaster, 'categories'):
                expenses['week'] = pd.to_datetime(expenses['date']).dt.to_period('W')
                
                spending_history = {}
                for cat in forecaster.categories:
                    cat_data = expenses[expenses['category'] == cat]
                    weekly = cat_data.groupby('week')['amount'].sum()
                    spending_history[cat] = weekly.tail(12).tolist()
                
                min_weeks = min(len(v) for v in spending_history.values()) if spending_history else 0
                
                if min_weeks >= 8:
                    try:
                        result = forecaster.predict(spending_history)

                        # Calculate current week's actual spending
                        current_week = expenses.groupby('week')['amount'].sum()
                        current_week_total = current_week.iloc[-1] if len(current_week) > 0 else 0
                        prev_week_total = current_week.iloc[-2] if len(current_week) > 1 else 0
                        
                        # Show current vs predicted side by side
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            week_change = ((current_week_total - prev_week_total) / prev_week_total * 100) if prev_week_total > 0 else 0
                            st.metric(
                                "This Week (Actual)",
                                f"${current_week_total:,.2f}",
                                f"{week_change:+.1f}% vs last week"
                            )
                        with metric_col2:
                            pred_change = ((result['predicted_total'] - current_week_total) / current_week_total * 100) if current_week_total > 0 else 0
                            st.metric(
                                "Next Week (Predicted)",
                                f"${result['predicted_total']:,.2f}",
                                f"{pred_change:+.1f}% vs this week"
                            )
                        
                        st.caption(f"Prediction range: ${result['total_lower_bound']:,.2f} – ${result['total_upper_bound']:,.2f}")
                        
                        # Per-category breakdown
                        if 'per_category' in result:
                            st.markdown("**Per-category forecast:**")
                            
                            # Sort by expected spending
                            sorted_cats = sorted(
                                result['per_category'].items(),
                                key=lambda x: x[1]['expected_spending'],
                                reverse=True
                            )
                            
                            for cat, pred in sorted_cats[:6]:
                                prob = pred['probability']
                                expected = pred['expected_spending']
                                uncertainty = pred['uncertainty']
                                
                                st.text(
                                    f"{cat}: ${expected:,.0f}  "
                                    f"({prob*100:.0f}% likely, ±${uncertainty:,.0f})"
                                )

                            # Per-category comparison: this week vs predicted
                            st.markdown("---")
                            st.markdown("**This week vs Next week (per category):**")
                            
                            current_week_period = expenses['week'].max()
                            current_cat = expenses[expenses['week'] == current_week_period].groupby('category')['amount'].sum()
                            
                            comparison_rows = []
                            for cat, pred in sorted_cats[:6]:
                                actual_now = current_cat.get(cat, 0)
                                predicted = pred['expected_spending']
                                diff = predicted - actual_now
                                comparison_rows.append({
                                    'Category': cat,
                                    'This Week': f"${actual_now:,.0f}",
                                    'Predicted': f"${predicted:,.0f}",
                                    'Change': f"{'↑' if diff > 0 else '↓'} ${abs(diff):,.0f}",
                                    'Confidence': f"{pred['probability']*100:.0f}%"
                                })

                            st.dataframe(
                                pd.DataFrame(comparison_rows),
                                width="stretch",
                                hide_index=True
                            )

                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info(f"Need at least 8 weeks of data per category. Current minimum: {min_weeks} weeks.")
            else:
                # Backward compatible: old forecaster without categories
                expenses['week'] = pd.to_datetime(expenses['date']).dt.to_period('W')
                weekly = expenses.groupby('week')['amount'].sum()
                history = weekly.tail(12).tolist()
                
                if len(history) >= 8:
                    try:
                        result = forecaster.predict(history)
                        st.metric(
                            "Predicted Next Week",
                            f"${result.get('predicted_spending', result.get('predicted_total', 0)):,.2f}",
                            f"Range: ${result.get('lower_bound', result.get('total_lower_bound', 0)):,.2f} - ${result.get('upper_bound', result.get('total_upper_bound', 0)):,.2f}"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info(f"Need at least 8 weeks of history. Current: {len(history)} weeks")
        else:
            st.warning("Forecaster not available. Run Notebook 05 first.")
    
    st.markdown("---")
    
    # Transaction Classification Demo
    st.subheader("Transaction Classifier Demo")
    
    classifier = load_classifier()
    
    if classifier:
        sample_input = st.text_input(
            "Enter a transaction description:",
            placeholder="e.g., STARBUCKS #1234 $5.75"
        )
        
        if sample_input:
            try:
                result = classifier.classify(sample_input)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Category",
                        result['category'],
                        f"{result['category_confidence']*100:.1f}% confidence"
                    )
                with col2:
                    st.metric(
                        "Subcategory",
                        result['subcategory'],
                        f"{result['subcategory_confidence']*100:.1f}% confidence"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Classifier not available. Run Notebook 03 first.")

    # Model info boxes
    st.markdown("---")
    st.subheader("ℹ️ Models Used on This Page")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(
            "**Anomaly Detection**\n\n"
            "**Model:** VAE (Variational Autoencoder)\n\n"
            "**How:** Learns to compress and reconstruct 'normal' spending. "
            "High reconstruction error = unusual spending.\n\n"
            "**Params:** 12,971 | **Trained from scratch**"
        )
    
    with col2:
        st.info(
            "**Spending Forecast**\n\n"
            "**Model:** ZICATT (Zero-Inflated Cross-Attention Temporal Transformer)\n\n"
            "**How:** Temporal attention per category + cross-attention between categories. "
            "Predicts probability of spending AND amount with uncertainty.\n\n"
            "**Params:** 141,059 | **Trained from scratch**"
        )
    
    with col3:
        st.info(
            "**Transaction Classifier**\n\n"
            "**Model:** Fine-tuned DistilBERT\n\n"
            "**How:** Pre-trained language model fine-tuned on transaction descriptions. "
            "Predicts both category and subcategory.\n\n"
            "**Params:** 66.9M | **Accuracy:** 94.84%"
        )

# ============================================================
# AI ASSISTANT PAGE
# ============================================================
def render_assistant(df, user_id):
    """Render AI assistant page"""
    st.title("AI Financial Assistant")
    assistant = load_assistant(df)
    # Mode indicator: Claude API vs demo
    if assistant and getattr(assistant, "use_api", False):
        st.caption("Powered by Claude")
    else:
        st.caption("Demo Mode")
    st.markdown("Ask me anything about your finances!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your SpendWise AI assistant. Ask me about your spending, subscriptions, or financial trends!"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"].replace("$", "\\$"))
            else:
                st.text(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your spending..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.text(prompt)

        # Get response
        if assistant:
            response = assistant.chat(prompt, user_id)
        else:
            response = "Assistant not available. Please run Notebook 06 first."

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response.replace("$", "\\$"))
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**Quick Questions:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Spending Summary"):
            st.session_state.messages.append({"role": "user", "content": "Give me a spending summary"})
            st.rerun()
    
    with col2:
        if st.button("My Subscriptions"):
            st.session_state.messages.append({"role": "user", "content": "What are my subscriptions?"})
            st.rerun()
    
    with col3:
        if st.button("Spending Trend"):
            st.session_state.messages.append({"role": "user", "content": "What's my spending trend?"})
            st.rerun()
    
    with col4:
        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared! How can I help you?"}
            ]
            st.rerun()

# ============================================================
# RECEIPT SCANNER PAGE
# ============================================================
def _parse_price_receipt(raw: str) -> float:
    """Parse price string from Donut OCR (same logic as My Account)."""
    s = str(raw).strip().replace("$", "").replace(" ", "")
    if not s or any(c in s for c in [":", "AM", "PM", "am", "pm"]):
        return 0.0
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." in s:
        s = s.replace(",", "")
    try:
        val = float(s)
        if val < 0.01 or val > 5000:
            return 0.0
        return round(val, 2)
    except ValueError:
        return 0.0


def render_receipt_scanner():
    """Render receipt scanner page. Uses a subprocess for parsing to avoid
    PyTorch/transformers mutex crash (libc++abi) on macOS when loading Donut in-process."""
    import subprocess
    import tempfile

    st.title("Receipt Scanner")
    st.markdown("Upload a receipt image to extract transaction data")

    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of your receipt",
        key="receipt_scanner_upload",
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Receipt")
            st.image(uploaded_file, width="stretch")
            st.caption(f"Current: {uploaded_file.name}")

        with col2:
            st.subheader("Extracted Data")
            st.caption(f"Results for: **{uploaded_file.name}**")

            worker_script = PROJECT_ROOT / "app" / "parse_receipt_worker.py"
            if not worker_script.exists():
                st.warning("Receipt parser worker not found. Run Notebook 02 first.")
                _show_receipt_demo_output(st)
                return

            with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("Scanning receipt..."):
                    result = subprocess.run(
                        [sys.executable, str(worker_script), tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=str(PROJECT_ROOT),
                    )
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

                if result.returncode == 0:
                    out = result.stdout.strip()
                    if out:
                        try:
                            data = json.loads(out)
                            if data.get("error"):
                                st.error(f"Parser error: {data['error'][:200]}")
                                st.caption("Raw output below.")
                                with st.expander("Raw OCR output", expanded=False):
                                    st.json(data)
                            else:
                                # Parse items: support "items" (list of {name, price}) or CORD "menu"
                                items = []
                                if "items" in data and isinstance(data["items"], list):
                                    for it in data["items"]:
                                        n = it.get("name", it.get("nm", ""))
                                        if isinstance(n, dict):
                                            n = n.get("value", n.get("nm", str(n)))
                                        if not n or not isinstance(n, str):
                                            continue
                                        p = it.get("price", 0)
                                        if isinstance(p, (int, float)) and 0 < p <= 500:
                                            items.append({"name": n, "price": round(float(p), 2)})
                                        else:
                                            pv = _parse_price_receipt(str(p))
                                            if pv > 0:
                                                items.append({"name": n, "price": pv})
                                elif "menu" in data:
                                    menu = data["menu"]
                                    if isinstance(menu, dict):
                                        menu = list(menu.values()) if menu else []
                                    if isinstance(menu, list):
                                        for item in menu:
                                            if not isinstance(item, dict):
                                                continue
                                            name = item.get("nm", item.get("name", ""))
                                            if isinstance(name, dict):
                                                name = name.get("value", name.get("nm", str(name)))
                                            if not name or not isinstance(name, str):
                                                continue
                                            name_lower = name.lower()
                                            if any(skip in name_lower for skip in ["server", "cashier", "table", "open time", "date", "time", "thank"]):
                                                continue
                                            price_raw = item.get("unitprice", item.get("price", "0"))
                                            price_str = str(price_raw).strip().replace("$", "").replace(" ", "").replace("@", "")
                                            if "," in price_str and "." not in price_str:
                                                parts = price_str.split(",")
                                                if len(parts) == 2 and len(parts[1]) == 2:
                                                    price_str = price_str.replace(",", ".")
                                                else:
                                                    price_str = price_str.replace(",", "")
                                            elif "," in price_str and "." in price_str:
                                                price_str = price_str.replace(",", "")
                                            if any(c in price_str for c in [":", "AM", "PM"]):
                                                continue
                                            try:
                                                price_val = float(price_str)
                                                if price_val <= 0 or price_val > 500:
                                                    continue
                                                items.append({"name": name, "price": round(price_val, 2)})
                                            except ValueError:
                                                continue

                                # Get total
                                receipt_total = None
                                for key in ["total", "sub_total"]:
                                    if key in data and receipt_total is None:
                                        val = data[key]
                                        if isinstance(val, dict):
                                            for sub_key in ["total_price", "total_etc", "subtotal_price", "sub_total_price"]:
                                                if sub_key in val:
                                                    raw = str(val[sub_key]).replace("$", "").replace(",", ".").replace(" ", "")
                                                    try:
                                                        receipt_total = float(raw)
                                                        if receipt_total > 0:
                                                            break
                                                    except ValueError:
                                                        pass
                                        else:
                                            try:
                                                receipt_total = float(str(val).replace("$", "").replace(",", ".").replace(" ", ""))
                                            except ValueError:
                                                pass

                                total = receipt_total if receipt_total and receipt_total > 0 else (sum(i["price"] for i in items) if items else 0)

                                # Show Items and Total FIRST (above raw), so user always sees them
                                if items:
                                    st.markdown("**Items found:**")
                                    for it in items:
                                        st.markdown(f"- {it['name']}: ${it['price']:.2f}")
                                else:
                                    st.info("No line items could be extracted. See raw OCR output below.")

                                if total > 0:
                                    st.success(f"**Receipt Total: ${total:.2f}**")

                                st.caption("Raw OCR output is hidden — click to expand below.")
                                with st.expander("Raw OCR output", expanded=False):
                                    st.json(data)

                                if st.button("✅ Add to Transactions", key="add_tx_ok"):
                                    st.success("Transaction added successfully! (Demo)")

                        except json.JSONDecodeError:
                            st.code(out)
                    else:
                        st.info("No structured data extracted.")
                else:
                    # Try parsing stdout even with non-zero return code
                    # (HuggingFace warnings in stderr cause non-zero exit)
                    out = result.stdout.strip()
                    if out:
                        try:
                            data = json.loads(out)
                            if data.get("error"):
                                st.error(f"Parser error: {data['error'][:200]}")
                            else:
                                items = []
                                if "items" in data and isinstance(data["items"], list):
                                    for it in data["items"]:
                                        n = it.get("name", it.get("nm", ""))
                                        if isinstance(n, dict):
                                            n = n.get("value", n.get("nm", str(n)))
                                        if not n or not isinstance(n, str):
                                            continue
                                        p = it.get("price", 0)
                                        if isinstance(p, (int, float)) and 0 < p <= 500:
                                            items.append({"name": n, "price": round(float(p), 2)})
                                        else:
                                            pv = _parse_price_receipt(str(p))
                                            if pv > 0:
                                                items.append({"name": n, "price": pv})
                                elif "menu" in data:
                                    menu = data["menu"]
                                    if isinstance(menu, dict):
                                        menu = list(menu.values()) if menu else []
                                    if isinstance(menu, list):
                                        for item in menu:
                                            if not isinstance(item, dict):
                                                continue
                                            name = item.get("nm", item.get("name", ""))
                                            if isinstance(name, dict):
                                                name = name.get("value", name.get("nm", str(name)))
                                            if not name or not isinstance(name, str):
                                                continue
                                            if any(skip in name.lower() for skip in ["server", "cashier", "table", "open time", "date", "time", "thank"]):
                                                continue
                                            price_raw = item.get("unitprice", item.get("price", "0"))
                                            price_str = str(price_raw).strip().replace("$", "").replace(" ", "").replace("@", "")
                                            if "," in price_str and "." not in price_str:
                                                parts = price_str.split(",")
                                                if len(parts) == 2 and len(parts[1]) == 2:
                                                    price_str = price_str.replace(",", ".")
                                                else:
                                                    price_str = price_str.replace(",", "")
                                            if any(c in price_str for c in [":", "AM", "PM"]):
                                                continue
                                            try:
                                                price_val = float(price_str)
                                                if price_val <= 0 or price_val > 500:
                                                    continue
                                                items.append({"name": name, "price": round(price_val, 2)})
                                            except ValueError:
                                                continue
                                receipt_total = None
                                for key in ["total", "sub_total"]:
                                    if key in data and receipt_total is None:
                                        val = data[key]
                                        if isinstance(val, dict):
                                            for sk in ["total_price", "total_etc", "subtotal_price", "sub_total_price"]:
                                                if sk in val:
                                                    try:
                                                        receipt_total = float(str(val[sk]).replace("$", "").replace(",", ".").replace(" ", ""))
                                                        if receipt_total > 0:
                                                            break
                                                    except ValueError:
                                                        pass
                                        else:
                                            try:
                                                receipt_total = float(str(val).replace("$", "").replace(",", ".").replace(" ", ""))
                                            except ValueError:
                                                pass
                                total = receipt_total if receipt_total and receipt_total > 0 else (sum(i["price"] for i in items) if items else 0)
                                if items:
                                    st.markdown("**Items found:**")
                                    for it in items:
                                        st.markdown(f"- {it['name']}: ${it['price']:.2f}")
                                else:
                                    st.info("No line items could be extracted. See raw OCR output below.")
                                if total > 0:
                                    st.success(f"**Receipt Total: ${total:.2f}**")
                                st.caption("Raw OCR output is hidden — click to expand below.")
                                with st.expander("Raw OCR output", expanded=False):
                                    st.json(data)
                                if st.button("✅ Add to Transactions", key="add_tx_fallback"):
                                    st.success("Transaction added successfully! (Demo)")
                        except json.JSONDecodeError:
                            err = result.stderr.strip() or "Unknown error"
                            st.error(f"Parsing failed: {err[:200]}")
                    else:
                        err = result.stderr.strip() or "Unknown error"
                        if "No module named 'receipt_parser'" in err or "Run Notebook 02" in err:
                            st.warning("Receipt parser not available. Run Notebook 02 first.")
                            _show_receipt_demo_output(st)
                        else:
                            st.error(f"Parsing failed: {err[:200]}")
            except subprocess.TimeoutExpired:
                st.error("Receipt parsing timed out. Try a smaller image.")
            except Exception as e:
                st.error(f"Error running receipt parser: {e}")
                _show_receipt_demo_output(st)


def _show_receipt_demo_output(st):
    """Show demo JSON when parser is unavailable or fails."""
    st.markdown("**Demo Output:**")
    demo_result = {
        "items": [
            {"name": "Coffee", "price": 4.99},
            {"name": "Sandwich", "price": 8.99}
        ],
        "subtotal": 13.98,
        "tax": 1.12,
        "total": 15.10
    }
    st.json(demo_result)



# ============================================================
# MAIN APP
# ============================================================
def main():
    """Main application"""
    
    # Mode selector at top of sidebar
    st.sidebar.title("SpendWise AI")
    st.sidebar.markdown("*AI-Powered Finance*")
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio(
        "Mode",
        ["ML Showcase", "My Account"],
        key="app_mode"
    )
    
    if mode == "My Account":
        # Personal account (login required)
        render_personal_account()
        return
    
    # ---- ML Showcase (existing app) ----
    
    # Load data
    df = load_transactions()
    
    if df is None:
        st.error("❌ Could not load transaction data. Please run Notebook 01 first.")
        st.stop()
    
    # User selection
    users = sorted(df['user_id'].unique())
    selected_user = st.sidebar.selectbox(
        "👤 Select User",
        users,
        index=0
    )
    
    # Time period
    period_options = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90
    }
    selected_period = st.sidebar.selectbox(
        "Time Period",
        list(period_options.keys()),
        index=1
    )
    days = period_options[selected_period]
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Transactions",
            "Analytics",
            "Insights",
            "AI Assistant",
            "Receipt Scanner"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Model status
    st.sidebar.markdown("### Model Status")
    classifier = load_classifier()
    anomaly = load_anomaly_detector()
    forecaster = load_forecaster()
    st.sidebar.markdown(f"- Classifier: {'OK' if classifier else 'Missing'}")
    st.sidebar.markdown(f"- Anomaly: {'OK' if anomaly else 'Missing'}")
    st.sidebar.markdown(f"- Forecaster: {'OK' if forecaster else 'Missing'}")
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard(df, selected_user, days)
    elif page == "Transactions":
        render_transactions(df, selected_user, days)
    elif page == "Analytics":
        render_analytics(df, selected_user, days)
    elif page == "Insights":
        render_insights(df, selected_user, days)
    elif page == "AI Assistant":
        render_assistant(df, selected_user)
    elif page == "Receipt Scanner":
        render_receipt_scanner()

# Run the app
if __name__ == "__main__":
    main()