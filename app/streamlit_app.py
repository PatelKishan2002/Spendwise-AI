"""SpendWise Streamlit app. Run: streamlit run app/streamlit_app.py"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

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

st.set_page_config(
    page_title="SpendWise AI",
    page_icon="SW",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 1rem;
    }

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

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .css-1d391kg {
        padding: 2rem 1rem;
    }

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

@st.cache_data
def load_transactions():
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
    try:
        with open(PROJECT_ROOT / "data/processed/label_mappings.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_classifier():
    try:
        from transaction_classifier import TransactionClassifierInference
        return TransactionClassifierInference(str(PROJECT_ROOT / "models/classifier_model"))
    except Exception as e:
        return None

@st.cache_resource
def load_anomaly_detector():
    try:
        from anomaly_detector import AnomalyDetector
        return AnomalyDetector(str(PROJECT_ROOT / "models/anomaly_model"))
    except Exception as e:
        return None

@st.cache_resource
def load_forecaster():
    try:
        from spending_forecaster import ZICATTInference
        return ZICATTInference(str(PROJECT_ROOT / "models/forecaster_model"))
    except Exception as e:
        return None

@st.cache_resource
def load_recommender(_df):
    try:
        from recommendation_engine import RecommendationService
        return RecommendationService(_df)
    except Exception as e:
        return None

@st.cache_resource
def load_assistant(_df):
    try:
        from llm_assistant import FinancialDataManager, FinancialAssistant
        data_manager = FinancialDataManager(_df)
        return FinancialAssistant(data_manager, mode="showcase")
    except Exception as e:
        return None

def get_spending_summary(df, user_id, days=30):
    user_df = df[df['user_id'] == user_id]
    max_date = user_df['date'].max()
    
    current = user_df[user_df['date'] > max_date - timedelta(days=days)]
    current_expenses = current[current['is_expense']]['amount'].abs().sum()
    current_income = current[~current['is_expense']]['amount'].sum()
    current_count = len(current[current['is_expense']])
    
    prev_start = max_date - timedelta(days=days*2)
    prev_end = max_date - timedelta(days=days)
    previous = user_df[(user_df['date'] > prev_start) & (user_df['date'] <= prev_end)]
    prev_expenses = previous[previous['is_expense']]['amount'].abs().sum()
    
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
    user_df = df[df['user_id'] == user_id]
    expenses = user_df[user_df['is_expense']].copy()
    expenses['amount'] = expenses['amount'].abs()
    
    monthly = expenses.groupby('month')['amount'].sum().tail(months)
    return monthly

def get_recent_transactions(df, user_id, limit=20):
    user_df = df[df['user_id'] == user_id].copy()
    return user_df.sort_values('date', ascending=False).head(limit)

def render_dashboard(df, user_id, days):
    st.title("Financial Dashboard")
    st.markdown(f"**User:** `{user_id}` | **Period:** Last {days} days")
    
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
            st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
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

def render_transactions(df, user_id, days):
    st.title("Transaction History")
    st.markdown(f"**User:** `{user_id}` | **Period:** Last {days} days")
    
    user_df = df[df['user_id'] == user_id].copy()
    max_date = user_df['date'].max()
    user_df = user_df[user_df['date'] > max_date - timedelta(days=days)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = ['All'] + sorted(user_df['category'].unique().tolist())
        selected_cat = st.selectbox("Category", categories)
    
    with col2:
        tx_type = st.selectbox("Type", ['All', 'Expenses', 'Income'])
    
    with col3:
        sort_order = st.selectbox("Sort", ['Newest First', 'Oldest First', 'Highest Amount', 'Lowest Amount'])
    
    filtered = user_df.copy()
    
    if selected_cat != 'All':
        filtered = filtered[filtered['category'] == selected_cat]
    
    if tx_type == 'Expenses':
        filtered = filtered[filtered['amount'] < 0]
    elif tx_type == 'Income':
        filtered = filtered[filtered['amount'] > 0]
    
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
    
    display_df = filtered[['date', 'merchant', 'amount', 'category', 'subcategory']].head(100).copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def render_analytics(df, user_id, days):
    st.title("Spending Analytics")
    st.markdown(f"**User:** `{user_id}` | **Period:** Last {days} days")
    
    user_df = df[df['user_id'] == user_id]
    max_date = user_df['date'].max()
    user_df = user_df[user_df['date'] > max_date - timedelta(days=days)]
    expenses = user_df[user_df['is_expense']].copy()
    expenses['amount'] = expenses['amount'].abs()
    
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
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
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
    st.plotly_chart(fig, use_container_width=True)

def render_insights(df, user_id, days):
    st.title("AI-Powered Insights")
    st.caption(f"*Same period as Dashboard: last {days} days for user {user_id}*")
    
    col1, col2 = st.columns(2)
    
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
                
                if is_anomaly:
                    st.error("Unusual spending detected.")
                else:
                    st.success("Spending looks normal.")
                
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
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Anomaly detector not available. Run Notebook 04 first.")
    
    with col2:
        st.subheader("Spending Forecast")
        
        forecaster = load_forecaster()
        
        if forecaster:
            user_df = df[df['user_id'] == user_id]
            expenses = user_df[user_df['is_expense']].copy()
            expenses['amount'] = expenses['amount'].abs()
            
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

                        summary = get_spending_summary(df, user_id, days)
                        actual_total = summary["expenses"]
                        change_pct = summary["change_pct"]
                        
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric(
                                f"Last {days} days (Actual)",
                                f"${actual_total:,.2f}",
                                f"{change_pct:+.1f}% vs prev period"
                            )
                        with metric_col2:
                            pred_change = ((result['predicted_total'] - actual_total) / actual_total * 100) if actual_total > 0 else 0
                            st.metric(
                                "Next Week (Predicted)",
                                f"${result['predicted_total']:,.2f}",
                                f"{pred_change:+.1f}% vs last {days} days"
                            )
                        
                        st.caption(f"Prediction range: ${result['total_lower_bound']:,.2f} – ${result['total_upper_bound']:,.2f}")
                        
                        if 'per_category' in result:
                            st.markdown("**Per-category forecast:**")
                            
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

                            st.markdown("---")
                            st.markdown(f"**Last {days} days (Actual) vs Next week (Predicted) by category:**")
                            
                            by_cat_actual = get_spending_by_category(df, user_id, days)
                            actual_by_cat = {item['category']: item['amount'] for item in by_cat_actual}
                            
                            comparison_rows = []
                            for cat, pred in sorted_cats[:6]:
                                actual_now = actual_by_cat.get(cat, 0)
                                predicted = pred['expected_spending']
                                diff = predicted - actual_now
                                comparison_rows.append({
                                    'Category': cat,
                                    f'Last {days}d': f"${actual_now:,.0f}",
                                    'Predicted': f"${predicted:,.0f}",
                                    'Change': f"{'↑' if diff > 0 else '↓'} ${abs(diff):,.0f}",
                                    'Confidence': f"{pred['probability']*100:.0f}%"
                                })

                            st.dataframe(
                                pd.DataFrame(comparison_rows),
                                use_container_width=True,
                                hide_index=True
                            )

                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.info(f"Need at least 8 weeks of data per category. Current minimum: {min_weeks} weeks.")
            else:
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

def render_assistant(df, user_id, days=30):
    st.title("AI Financial Assistant")
    assistant = load_assistant(df)
    if assistant and getattr(assistant, "data_manager", None):
        assistant.data_manager.default_period_days = days
    st.caption(f"*Using same period as Dashboard: last {days} days*")
    if assistant and getattr(assistant, "use_api", False):
        st.caption("Powered by Claude")
    else:
        st.caption("Demo Mode")
    st.markdown("Ask me anything about your finances!")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your SpendWise AI assistant. Ask me about your spending, subscriptions, or financial trends!"}
        ]

    if (st.session_state.messages 
        and st.session_state.messages[-1]["role"] == "user"
        and len(st.session_state.messages) >= 2
        and st.session_state.messages[-2]["role"] != "user"):
        pending_prompt = st.session_state.messages[-1]["content"]
        if assistant:
            response = assistant.chat(pending_prompt, user_id)
        else:
            response = "Assistant not available. Please run Notebook 06 first."
        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"].replace("$", "\\$"))
            else:
                st.text(message["content"])

    if prompt := st.chat_input("Ask about your spending..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.text(prompt)

        if assistant:
            response = assistant.chat(prompt, user_id)
        else:
            response = "Assistant not available. Please run Notebook 06 first."

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response.replace("$", "\\$"))
    
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

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("This Month"):
            st.session_state.messages.append({"role": "user", "content": "How much have I spent this month?"})
            st.rerun()
    with col2:
        if st.button("By Category"):
            st.session_state.messages.append({"role": "user", "content": "Show spending by category"})
            st.rerun()
    with col3:
        if st.button("Compare to Average"):
            st.session_state.messages.append({"role": "user", "content": "Compare my food spending to average"})
            st.rerun()

def _parse_price_receipt(raw: str) -> float:
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


def _looks_like_store_or_header(name: str) -> bool:
    if not name or len(name) < 2:
        return True
    n = name.lower()
    skip = [
        "server", "cashier", "table", "open time", "date", "time", "thank", "gst", "pst",
        "visit ", "owned and operated", "wholesale", "store", "member", "address", "blvd",
        "invoice", "acct", "auth", "interac", "receipt", "operated", "visit", "www.",
        "owned", "transaction", "record", "type:", "purchase", "debit", "credit",
    ]
    if any(s in n for s in skip):
        return True
    if "#" in name and any(c.isdigit() for c in name):
        return True
    digits = sum(c.isdigit() for c in name)
    if len(name) <= 12 and digits >= len(name) * 0.6:
        return True
    return False


def _extract_receipt_items(data: dict) -> list:
    items = []
    root_nm = (data.get("nm") or data.get("name") or "") if isinstance(data.get("nm"), str) else ""

    def get_name(obj):
        if not isinstance(obj, dict):
            return str(obj).strip() if obj else ""
        for key in ("nm", "name", "value"):
            v = obj.get(key)
            if v is None:
                continue
            if isinstance(v, dict):
                return get_name(v)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def get_price(obj):
        if not isinstance(obj, dict):
            return None
        for key in ("price", "unitprice", "total", "total_price", "unit_price"):
            v = obj.get(key)
            if v is None:
                continue
            if isinstance(v, (int, float)) and 0 < v <= 5000:
                return round(float(v), 2)
            if isinstance(v, str):
                p = _parse_price_receipt(v)
                if p > 0:
                    return p
        return None

    def add_item(name: str, price: float):
        if not name or price <= 0 or price > 5000:
            return
        if _looks_like_store_or_header(name):
            return
        if root_nm and name.strip().lower() == root_nm.strip().lower():
            return
        items.append({"name": name, "price": round(float(price), 2)})

    if "items" in data and isinstance(data["items"], list):
        for it in data["items"]:
            name = get_name(it)
            price = it.get("price")
            if isinstance(price, (int, float)) and 0 < price <= 5000:
                add_item(name, float(price))
            else:
                p = get_price(it) or _parse_price_receipt(str(price or ""))
                if p > 0:
                    add_item(name, p)

    if not items and "menu" in data:
        menu = data["menu"]
        if isinstance(menu, dict):
            menu = list(menu.values())
        if isinstance(menu, list):
            for it in menu:
                if not isinstance(it, dict):
                    continue
                name = get_name(it)
                price = get_price(it)
                if price is None:
                    price = _parse_price_receipt(str(it.get("unitprice", it.get("price", ""))))
                if price and price > 0:
                    add_item(name, price)

    if not items:
        for key in ("line_items", "entries", "products", "lines"):
            if key not in data or not isinstance(data[key], list):
                continue
            for it in data[key]:
                if not isinstance(it, dict):
                    continue
                name = get_name(it)
                price = get_price(it)
                if price is None:
                    price = _parse_price_receipt(str(it.get("price", it.get("unitprice", ""))))
                if price and price > 0:
                    add_item(name, price)

    return items


def render_receipt_scanner():
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
            st.image(uploaded_file, use_container_width=True)
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
                                items = _extract_receipt_items(data)

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

                                items_sum = sum(i["price"] for i in items) if items else 0
                                if receipt_total and receipt_total > 0:
                                    if receipt_total > 5000 or (items and abs(receipt_total - items_sum) > 1000):
                                        total = round(items_sum, 2) if items else 0
                                        if receipt_total > 5000 and items:
                                            st.caption(f"*Total from Donut (${receipt_total:,.2f}) looked incorrect; showing sum of items.*")
                                    else:
                                        total = receipt_total
                                else:
                                    total = round(items_sum, 2) if items else 0

                                if data.get("parser") == "donut":
                                    st.caption("**Parser:** Donut (CORD)")
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
                    out = result.stdout.strip()
                    if out:
                        try:
                            data = json.loads(out)
                            if data.get("error"):
                                st.error(f"Parser error: {data['error'][:200]}")
                            else:
                                items = _extract_receipt_items(data)
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
                                items_sum = sum(i["price"] for i in items) if items else 0
                                if receipt_total and receipt_total > 0:
                                    if receipt_total > 5000 or (items and abs(receipt_total - items_sum) > 1000):
                                        total = round(items_sum, 2) if items else 0
                                        if receipt_total > 5000 and items:
                                            st.caption(f"*Total from Donut (${receipt_total:,.2f}) looked incorrect; showing sum of items.*")
                                    else:
                                        total = receipt_total
                                else:
                                    total = round(items_sum, 2) if items else 0
                                if data.get("parser") == "donut":
                                    st.caption("**Parser:** Donut (CORD)")
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


def main():
    st.sidebar.title("SpendWise AI")
    st.sidebar.markdown("*AI-Powered Finance*")
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio(
        "Mode",
        ["ML Showcase", "My Account"],
        key="app_mode"
    )
    
    if mode == "My Account":
        render_personal_account()
        return

    df = load_transactions()
    
    if df is None:
        st.error("❌ Could not load transaction data. Please run Notebook 01 first.")
        st.stop()
    
    users = sorted(df['user_id'].unique())
    selected_user = st.sidebar.selectbox(
        "👤 Select User",
        users,
        index=0
    )
    
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
    
    st.sidebar.markdown("### Components Status")
    classifier = load_classifier()
    anomaly = load_anomaly_detector()
    forecaster = load_forecaster()
    recommender = load_recommender(df)
    assistant = load_assistant(df)
    st.sidebar.markdown(f"- Classifier: {'OK' if classifier else 'Missing'}")
    st.sidebar.markdown(f"- Anomaly: {'OK' if anomaly else 'Missing'}")
    st.sidebar.markdown(f"- Forecaster: {'OK' if forecaster else 'Missing'}")
    st.sidebar.markdown(f"- Recommender: {'OK' if recommender else 'Missing'}")
    st.sidebar.markdown(f"- Assistant: {'OK' if assistant else 'Missing'}")
    
    if page == "Dashboard":
        render_dashboard(df, selected_user, days)
    elif page == "Transactions":
        render_transactions(df, selected_user, days)
    elif page == "Analytics":
        render_analytics(df, selected_user, days)
    elif page == "Insights":
        render_insights(df, selected_user, days)
    elif page == "AI Assistant":
        render_assistant(df, selected_user, days)
    elif page == "Receipt Scanner":
        render_receipt_scanner()

if __name__ == "__main__":
    main()
