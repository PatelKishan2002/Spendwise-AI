"""
LLM Assistant Module - SpendWise AI
Financial assistant using Claude API with tool use.
Produced by: 06_llm_integration.ipynb
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Union

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except Exception:
    Anthropic = None
    ANTHROPIC_AVAILABLE = False


class FinancialDataManager:
    """Manages financial data queries for the LLM."""

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["month"] = self.df["date"].dt.to_period("M")
        self.df["week"] = self.df["date"].dt.to_period("W")
        self.df["is_expense"] = self.df["amount"] < 0

    def get_spending_by_category(
        self,
        user_id: str,
        start_date: str = None,
        end_date: str = None,
        start_after: Union[datetime, pd.Timestamp, None] = None,
    ) -> dict:
        df = self._filter_user_and_dates(user_id, start_date, end_date, start_after)
        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        by_category = (
            expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
        )
        return {
            "user_id": user_id,
            "period": f"{start_date or 'all time'} to {end_date or 'now'}",
            "total_spending": round(expenses["amount"].sum(), 2),
            "by_category": {cat: round(amt, 2) for cat, amt in by_category.items()},
            "top_category": by_category.index[0] if len(by_category) > 0 else None,
        }

    def get_spending_trend(
        self, user_id: str, category: str = None, period: str = "monthly"
    ) -> dict:
        df = self._filter_user_and_dates(user_id)
        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        if category:
            expenses = expenses[expenses["category"] == category]
        grouped = (
            expenses.groupby("week")["amount"].sum()
            if period == "weekly"
            else expenses.groupby("month")["amount"].sum()
        )
        trend = {str(k): round(v, 2) for k, v in grouped.tail(6).to_dict().items()}
        values = list(trend.values())
        change = (
            ((values[-1] - values[-2]) / values[-2]) * 100
            if len(values) >= 2 and values[-2] > 0
            else 0
        )
        return {
            "user_id": user_id,
            "category": category or "all",
            "period": period,
            "trend": trend,
            "latest_change_percent": round(change, 1),
            "average": round(np.mean(values), 2) if values else 0,
        }

    def get_subscriptions(self, user_id: str) -> dict:
        df = self._filter_user_and_dates(user_id)
        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        merchant_counts = expenses.groupby("merchant").agg(
            {"amount": ["count", "mean", "std"], "date": ["min", "max"]}
        )
        merchant_counts.columns = [
            "count",
            "avg_amount",
            "std_amount",
            "first_date",
            "last_date",
        ]
        subscriptions = merchant_counts[
            (merchant_counts["count"] >= 3) & (merchant_counts["std_amount"] < 1)
        ].copy()
        subscriptions["monthly_cost"] = subscriptions["avg_amount"]
        sub_list = [
            {
                "merchant": m,
                "monthly_cost": round(row["monthly_cost"], 2),
                "occurrences": int(row["count"]),
            }
            for m, row in subscriptions.iterrows()
        ]
        total_monthly = sum(s["monthly_cost"] for s in sub_list)
        return {
            "user_id": user_id,
            "subscriptions": sorted(
                sub_list, key=lambda x: x["monthly_cost"], reverse=True
            ),
            "total_monthly": round(total_monthly, 2),
            "total_yearly": round(total_monthly * 12, 2),
            "count": len(sub_list),
        }

    def get_spending_summary(
        self, user_id: str, period: str = "last_month"
    ) -> dict:
        # Use this user's max date so "last N days" matches the Dashboard (per-user).
        user_df = self.df[self.df["user_id"] == user_id]
        max_date = user_df["date"].max()
        # Use the exact same predicate as the Dashboard: date > max_date - timedelta(days=N).
        if period == "last_week":
            start_after = max_date - timedelta(days=7)
        elif period == "last_month":
            start_after = max_date - timedelta(days=30)
        elif period == "last_3_months":
            start_after = max_date - timedelta(days=90)
        else:
            start_after = None
        category_data = self.get_spending_by_category(
            user_id, start_after=start_after
        )
        trend_data = self.get_spending_trend(user_id)
        df = self._filter_user_and_dates(user_id, start_after=start_after)
        n_transactions = len(df[df["is_expense"]])
        return {
            "user_id": user_id,
            "period": period,
            "total_spending": category_data["total_spending"],
            "transaction_count": n_transactions,
            "top_categories": dict(
                list(category_data["by_category"].items())[:5]
            ),
            "spending_change": trend_data["latest_change_percent"],
            "daily_average": round(category_data["total_spending"] / 30, 2)
            if period == "last_month"
            else None,
        }

    def compare_to_average(self, user_id: str, category: str) -> dict:
        user_df = self.df[self.df["user_id"] == user_id]
        max_date = user_df["date"].max()
        # Same 30-day window as Dashboard "Last 30 days": date > max_date - 30
        start_after_30 = max_date - timedelta(days=30)
        recent = self.get_spending_by_category(user_id, start_after=start_after_30)
        recent_amount = recent["by_category"].get(category, 0)
        all_time = self.get_spending_by_category(user_id)
        df = self._filter_user_and_dates(user_id)
        n_months = df["month"].nunique()
        historical_avg = all_time["by_category"].get(category, 0) / max(n_months, 1)
        diff_percent = (
            ((recent_amount - historical_avg) / historical_avg) * 100
            if historical_avg > 0
            else 0
        )
        return {
            "user_id": user_id,
            "category": category,
            "recent_spending": round(recent_amount, 2),
            "historical_monthly_avg": round(historical_avg, 2),
            "difference_percent": round(diff_percent, 1),
            "status": "above"
            if diff_percent > 10
            else "below"
            if diff_percent < -10
            else "normal",
        }

    def _filter_user_and_dates(
        self,
        user_id: str,
        start_date: str = None,
        end_date: str = None,
        start_after: Union[datetime, pd.Timestamp, None] = None,
    ) -> pd.DataFrame:
        df = self.df[self.df["user_id"] == user_id]
        if start_after is not None:
            df = df[df["date"] > start_after]
        if start_date and start_after is None:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df


TOOLS = [
    {
        "name": "get_spending_by_category",
        "description": "Get total spending by category for a time period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "get_spending_trend",
        "description": "Get spending trend over time (weekly or monthly).",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "category": {"type": "string"},
                "period": {"type": "string", "enum": ["weekly", "monthly"]},
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "get_subscriptions",
        "description": "Detect recurring charges/subscriptions.",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "get_spending_summary",
        "description": "Get spending summary for a period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["last_week", "last_month", "last_3_months"],
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "compare_to_average",
        "description": "Compare recent spending in a category to historical average.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["user_id", "category"],
        },
    },
]


class FinancialAssistant:
    """AI-powered financial assistant using Claude."""

    def __init__(self, data_manager: FinancialDataManager = None, api_key: str = None):
        self.data_manager = data_manager
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_api = bool(self.api_key)
        if self.use_api:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
        self.system_prompt = (
            "You are a helpful financial assistant for SpendWise AI. "
            "Be concise; use numbers from the data; format currency with $ and 2 decimal places. "
            "Categories: Food & Dining, Transportation, Shopping, Bills & Utilities, "
            "Subscriptions, Entertainment, Health & Wellness, Travel, Education, "
            "Personal Care, Financial, Income"
        )
        self.tool_functions = {
            "get_spending_by_category": self.data_manager.get_spending_by_category,
            "get_spending_trend": self.data_manager.get_spending_trend,
            "get_subscriptions": self.data_manager.get_subscriptions,
            "get_spending_summary": self.data_manager.get_spending_summary,
            "compare_to_average": self.data_manager.compare_to_average,
        }

    def chat(self, user_message: str, user_id: str = "user_0001") -> str:
        if not self.use_api:
            return self._chat_demo(user_message, user_id)
        full_message = f"[User ID: {user_id}] {user_message}"
        try:
            return self._chat_with_api(full_message, user_id)
        except Exception:
            return self._chat_demo(user_message, user_id)

    def _chat_with_api(self, message: str, user_id: str) -> str:
        try:
            messages = [{"role": "user", "content": message}]
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self.system_prompt,
                tools=TOOLS,
                messages=messages,
            )
            while response.stop_reason == "tool_use":
                tool_use_block = next(
                    (b for b in response.content if getattr(b, "type", None) == "tool_use"),
                    None,
                )
                if not tool_use_block:
                    break
                tool_input = dict(tool_use_block.input)
                if "user_id" not in tool_input:
                    tool_input["user_id"] = user_id
                tool_result = self.tool_functions[tool_use_block.name](**tool_input)
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "content": json.dumps(tool_result),
                            }
                        ],
                    }
                )
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=self.system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "I couldn't generate a response. Please try again."
        except Exception:
            raise

    def _chat_demo(self, message: str, user_id: str) -> str:
        msg = message.lower()
        if any(w in msg for w in ["subscription", "recurring", "monthly"]):
            r = self.data_manager.get_subscriptions(user_id)
            top = "\n".join(
                [f"- {s['merchant']}: ${s['monthly_cost']:,.2f}/month" for s in r["subscriptions"][:3]]
            )
            return (
                f"Found {r['count']} recurring charges:\n{top}\n"
                f"Total monthly: ${r['total_monthly']:,.2f} (${r['total_yearly']:,.2f}/year)"
            )
        if any(w in msg for w in ["category", "breakdown", "where", "spent"]):
            r = self.data_manager.get_spending_by_category(user_id)
            top = "\n".join([f"- {c}: ${a:,.2f}" for c, a in list(r["by_category"].items())[:5]])
            return (
                f"Spending breakdown:\n{top}\n"
                f"Total: ${r['total_spending']:,.2f}. Top category: {r['top_category']}"
            )
        if any(w in msg for w in ["trend", "pattern", "change", "over time"]):
            r = self.data_manager.get_spending_trend(user_id, period="monthly")
            d = "increased" if r["latest_change_percent"] > 0 else "decreased"
            recent_lines = "\n".join([f"- {k}: ${v:,.2f}" for k, v in r["trend"].items()])
            return (
                f"Spending {d} by {abs(r['latest_change_percent']):.1f}%. "
                f"Average: ${r['average']:,.2f}.\nRecent months:\n{recent_lines}"
            )
        if any(w in msg for w in ["summary", "overview", "total", "how much", "last month", "last week", "expense"]):
            period = "last_week" if "last week" in msg else "last_month"
            r = self.data_manager.get_spending_summary(user_id, period)
            top_lines = "\n".join([f"- {c}: ${a:,.2f}" for c, a in list(r["top_categories"].items())[:3]])
            period_label = "Last week" if period == "last_week" else "Last month"
            daily_line = (
                f"Daily average: ${r['daily_average']:,.2f}.\n"
                if r.get("daily_average") is not None
                else ""
            )
            return (
                f"{period_label} you spent ${r['total_spending']:,.2f} across {r['transaction_count']} transactions.\n"
                f"{daily_line}"
                f"Change vs previous period: {r['spending_change']:+.1f}%.\n"
                f"Top categories:\n{top_lines}"
            )
        return (
            "Try: 'What are my subscriptions?', 'Show spending by category', "
            "'Spending trend?', 'Summary of last month', 'Compare food spending to average?'"
        )