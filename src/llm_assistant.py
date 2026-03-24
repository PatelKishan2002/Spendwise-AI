import json
import os
import re
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    Anthropic = None
    ANTHROPIC_AVAILABLE = False


class FinancialDataManager:
    """Loads filters from a transaction dataframe; ``default_period_days=None`` means all-time (personal dashboard)."""

    def __init__(self, transactions_df: pd.DataFrame, default_period_days: Optional[int] = 30):
        self.df = transactions_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["month"] = self.df["date"].dt.to_period("M")
        self.df["week"] = self.df["date"].dt.to_period("W")
        self.df["is_expense"] = self.df["amount"] < 0
        self.default_period_days = default_period_days

    def get_spending_by_category(self, user_id: str, start_date: str = None, end_date: str = None) -> dict:
        user_df = self.df[self.df["user_id"] == user_id]

        if start_date is None and end_date is None and len(user_df) > 0 and self.default_period_days is not None:
            max_date = user_df["date"].max()
            df = user_df[user_df["date"] > max_date - timedelta(days=self.default_period_days)]
            period_str = f"last {self.default_period_days} days"
        else:
            df = self._filter_user_and_dates(user_id, start_date, end_date)
            period_str = f"{start_date or 'all time'} to {end_date or 'now'}"

        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        by_category = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
        return {
            "user_id": user_id,
            "period": period_str,
            "total_spending": round(expenses["amount"].sum(), 2),
            "by_category": {cat: round(amt, 2) for cat, amt in by_category.items()},
            "top_category": by_category.index[0] if len(by_category) > 0 else None,
        }

    def get_spending_trend(self, user_id: str, category: str = None, period: str = "monthly") -> dict:
        df = self._filter_user_and_dates(user_id)
        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        if category:
            expenses = expenses[expenses["category"] == category]
        grouped = (expenses.groupby("week")["amount"].sum()
                   if period == "weekly"
                   else expenses.groupby("month")["amount"].sum())
        trend = {str(k): round(v, 2) for k, v in grouped.tail(6).to_dict().items()}
        values = list(trend.values())
        change = (((values[-1] - values[-2]) / values[-2]) * 100
                  if len(values) >= 2 and values[-2] > 0 else 0)
        return {
            "user_id": user_id, "category": category or "all", "period": period,
            "trend": trend, "latest_change_percent": round(change, 1),
            "average": round(np.mean(values), 2) if values else 0,
        }

    def get_subscriptions(self, user_id: str) -> dict:
        df = self._filter_user_and_dates(user_id)
        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        merchant_counts = expenses.groupby("merchant").agg(
            {"amount": ["count", "mean", "std"], "date": ["min", "max"]}
        )
        merchant_counts.columns = ["count", "avg_amount", "std_amount", "first_date", "last_date"]
        subscriptions = merchant_counts[
            (merchant_counts["count"] >= 3) & (merchant_counts["std_amount"] < 1)
        ].copy()
        subscriptions["monthly_cost"] = subscriptions["avg_amount"]
        sub_list = [
            {"merchant": m, "monthly_cost": round(row["monthly_cost"], 2),
             "occurrences": int(row["count"])}
            for m, row in subscriptions.iterrows()
        ]
        total_monthly = sum(s["monthly_cost"] for s in sub_list)
        return {
            "user_id": user_id,
            "subscriptions": sorted(sub_list, key=lambda x: x["monthly_cost"], reverse=True),
            "total_monthly": round(total_monthly, 2),
            "total_yearly": round(total_monthly * 12, 2),
            "count": len(sub_list),
        }

    def compare_to_average(self, user_id: str, category: str) -> dict:
        user_df = self.df[self.df["user_id"] == user_id]
        max_date = user_df["date"].max()
        last_month_start = (max_date - timedelta(days=30)).strftime("%Y-%m-%d")
        recent = self.get_spending_by_category(user_id, last_month_start)
        recent_amount = recent["by_category"].get(category, 0)
        all_time = self.get_spending_by_category(user_id)
        n_months = user_df["month"].nunique()
        historical_avg = all_time["by_category"].get(category, 0) / max(n_months, 1)
        diff_percent = (((recent_amount - historical_avg) / historical_avg) * 100
                        if historical_avg > 0 else 0)
        return {
            "user_id": user_id, "category": category,
            "recent_spending": round(recent_amount, 2),
            "historical_monthly_avg": round(historical_avg, 2),
            "difference_percent": round(diff_percent, 1),
            "status": "above" if diff_percent > 10 else "below" if diff_percent < -10 else "normal",
        }

    def get_user_date_bounds(self, user_id: str) -> dict:
        user_df = self.df[self.df["user_id"] == user_id]
        if len(user_df) == 0:
            return {"user_id": user_id, "min_date": None, "max_date": None, "has_data": False}
        mn = pd.Timestamp(user_df["date"].min()).strftime("%Y-%m-%d")
        mx = pd.Timestamp(user_df["date"].max()).strftime("%Y-%m-%d")
        return {"user_id": user_id, "min_date": mn, "max_date": mx, "has_data": True}

    def get_spending_for_date(self, user_id: str, date: str) -> dict:
        df = self._filter_user_and_dates(user_id, start_date=date, end_date=date)
        expenses = df[df["is_expense"]].copy()
        expenses["amount"] = expenses["amount"].abs()
        by_cat = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
        total = round(expenses["amount"].sum(), 2)
        n = len(expenses)
        return {
            "user_id": user_id,
            "date": date,
            "total_spending": total,
            "transaction_count": n,
            "by_category": {cat: round(amt, 2) for cat, amt in by_cat.items()},
            "has_data": n > 0 or total > 0,
            "transactions": [
                {"merchant": row["merchant"], "amount": round(abs(row["amount"]), 2),
                 "category": row["category"]}
                for _, row in expenses.iterrows()
            ][:10],
        }

    def get_spending_summary(self, user_id: str, period: str = "last_month") -> dict:
        user_df = self.df[self.df["user_id"] == user_id]
        if len(user_df) == 0:
            return {
                "user_id": user_id, "period": period,
                "total_spending": 0, "transaction_count": 0,
                "top_categories": {}, "spending_change": 0,
                "daily_average": 0, "has_data": False,
            }

        if period == "all_time":
            cat_data = self.get_spending_by_category(user_id)
            n = len(user_df[user_df["is_expense"]])
            total = cat_data["total_spending"]
            span = max((user_df["date"].max() - user_df["date"].min()).days, 1)
            return {
                "user_id": user_id, "period": "all_time",
                "total_spending": total, "transaction_count": n,
                "top_categories": dict(list(cat_data["by_category"].items())[:5]),
                "spending_change": 0,
                "daily_average": round(total / span, 2),
                "has_data": True,
            }

        if period == "this_month":
            if self.default_period_days is not None:
                period = "last_month"
            else:
                user_max = pd.Timestamp(user_df["date"].max())
                start = user_max.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                s, e = start.strftime("%Y-%m-%d"), user_max.strftime("%Y-%m-%d")
                df_filt = self._filter_user_and_dates(user_id, s, e)
                expenses = df_filt[df_filt["is_expense"]].copy()
                expenses["amount"] = expenses["amount"].abs()
                by_cat = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
                total = round(expenses["amount"].sum(), 2)
                n = len(expenses)
                days_in = max((user_max - start).days + 1, 1)
                return {
                    "user_id": user_id, "period": "this_month",
                    "total_spending": total, "transaction_count": n,
                    "top_categories": dict(list(by_cat.to_dict().items())[:5]),
                    "spending_change": 0,
                    "daily_average": round(total / days_in, 2),
                    "month_label": start.strftime("%B %Y"),
                    "has_data": True,
                }

        days_map = {"last_week": 7, "last_month": 30, "last_3_months": 90}
        days = days_map.get(period, self.default_period_days or 30)
        max_date = user_df["date"].max()

        current = user_df[user_df["date"] > max_date - timedelta(days=days)]
        current_exp = current[current["is_expense"]].copy()
        current_exp["amount"] = current_exp["amount"].abs()
        total = round(float(current_exp["amount"].sum()), 2)
        n = int(len(current_exp))
        by_cat = current_exp.groupby("category")["amount"].sum().sort_values(ascending=False)

        prev_start = max_date - timedelta(days=days * 2)
        prev_end = max_date - timedelta(days=days)
        previous = user_df[(user_df["date"] > prev_start) & (user_df["date"] <= prev_end)]
        prev_exp = previous[previous["is_expense"]].copy()
        prev_exp["amount"] = prev_exp["amount"].abs()
        prev_total = float(prev_exp["amount"].sum())
        change_pct = ((total - prev_total) / prev_total * 100) if prev_total > 0 else 0

        return {
            "user_id": user_id,
            "period": period,
            "total_spending": total,
            "transaction_count": n,
            "top_categories": {k: round(float(v), 2) for k, v in list(by_cat.to_dict().items())[:5]},
            "spending_change": round(change_pct, 1),
            "daily_average": round(total / days, 2),
            "has_data": True,
        }

    def _filter_user_and_dates(self, user_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = self.df[self.df["user_id"] == user_id]
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df


def _to_json_safe(obj):
    """Make nested structures safe for ``json`` (NumPy scalars use ``.item()``)."""
    if obj is None:
        return None
    if isinstance(obj, (str, bytes)):
        return obj.decode() if isinstance(obj, bytes) else obj
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _to_json_safe(obj.item())
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    if hasattr(obj, "model_dump"):
        return _to_json_safe(obj.model_dump(mode="python"))
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (int, float)):
        return obj
    return obj


def _json_dumps_tool_result(obj) -> str:
    return json.dumps(_to_json_safe(obj))


def _serialize_assistant_content(content) -> list:
    out = []
    for block in content:
        t = getattr(block, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(block, "text", "")})
        elif t == "tool_use":
            inp = getattr(block, "input", None)
            if isinstance(inp, dict):
                inp = _to_json_safe(dict(inp))
            else:
                inp = _to_json_safe(inp)
            out.append({
                "type": "tool_use",
                "id": str(getattr(block, "id", "")),
                "name": str(getattr(block, "name", "")),
                "input": inp,
            })
    return out


TOOLS = [
    {
        "name": "get_spending_by_category",
        "description": "Get total spending by category for a time period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
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
        "description": "Get spending summary for a period: last_week, last_month, last_3_months, this_month, or all_time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "period": {
                    "type": "string",
                    "enum": ["last_week", "last_month", "last_3_months", "this_month", "all_time"],
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "get_user_date_bounds",
        "description": "Get min/max dates available for the user (helps interpret 'today' in historical datasets).",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "get_spending_for_date",
        "description": "Get spending for a specific date. Use for today, yesterday, 2 days ago, etc. Pass the date as YYYY-MM-DD.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["user_id", "date"],
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



_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
}


def _parse_days_ago(msg: str) -> Optional[int]:
    m = msg.lower()
    if "today" in m:
        return 0
    if "yesterday" in m and "day before" not in m and "before yesterday" not in m:
        return 1
    if "day before yesterday" in m or "before yesterday" in m:
        return 2
    match = re.search(r"(\w+)\s+days?\s+ago", m)
    if match:
        w = match.group(1)
        n = _WORD_TO_NUM.get(w)
        if n is not None:
            return n
        try:
            return int(w)
        except ValueError:
            pass
    return None


class FinancialAssistant:

    def __init__(self, data_manager: FinancialDataManager, api_key: str = None, mode: str = "showcase"):
        self.data_manager = data_manager
        self.mode = mode
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=self.api_key)
            self.use_api = True
        else:
            self.client = None
            self.use_api = False
            print("Running in demo mode (no API key). Responses will be simulated.")

        _today = date.today().strftime("%Y-%m-%d")
        _yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        if self.mode == "personal":
            self.system_prompt = (
                "You are a financial assistant for SpendWise AI analyzing the user's real personal expenses. "
                "Be concise; use numbers from the data; format currency with $ and 2 decimal places.\n\n"
                f"Today's date is {_today}.\n"
                "When the user asks about a specific day:\n"
                f"- 'today' = {_today}\n"
                f"- 'yesterday' = {_yesterday}\n"
                "- 'N days ago' = today minus N days\n"
                "Use get_spending_for_date with the correct YYYY-MM-DD date.\n\n"
                "For period queries:\n"
                "- 'total expense' or 'all time' → get_spending_summary with period 'all_time'\n"
                "- 'this month' → get_spending_summary with period 'this_month'\n"
                "- 'last week' → get_spending_summary with period 'last_week'\n\n"
                "If a date query returns 0 transactions, say 'No expenses recorded for that day.' "
                "and suggest checking another period.\n\n"
                "Categories: Food & Dining, Transportation, Shopping, Bills & Utilities, "
                "Subscriptions, Entertainment, Health & Wellness, Travel, Education, "
                "Personal Care, Financial, Income."
            )
        else:
            self.system_prompt = (
                "You are a financial assistant for SpendWise AI analyzing HISTORICAL demo data. "
                "Be concise; use numbers from the data; format currency with $ and 2 decimal places.\n\n"
                "CRITICAL RULES FOR HISTORICAL DATA:\n"
                "1. This data is NOT live. It covers a past date range, NOT the real calendar date.\n"
                "2. When you call any tool, the response contains the actual dates/labels from the data. "
                "ALWAYS use the dates and labels returned by the tool in your response. "
                "NEVER substitute the real calendar date.\n"
                "3. IMPORTANT OUTPUT FORMAT: Do NOT mention specific month names (e.g., 'February 2026'). "
                "Always answer using relative periods only: 'this month', 'last month', 'last week', 'last 90 days'.\n"
                "4. If the tool returns period data, report the numbers exactly, but keep the wording relative.\n"
                "5. Do NOT attempt single-day lookups (today, yesterday, specific dates). "
                "If the user asks, politely say this is historical data and suggest period queries.\n\n"
                "Available queries:\n"
                "- get_spending_summary: periods are last_week, last_month, last_3_months, this_month, all_time\n"
                "- get_spending_by_category: category breakdowns with optional date range\n"
                "- get_spending_trend: weekly or monthly trends\n"
                "- get_subscriptions: recurring charges\n"
                "- compare_to_average: compare a category to historical average\n\n"
                "When reporting results, ALWAYS use the exact values from the tool response, "
                "but keep wording relative (no month names, no real calendar dates).\n\n"
                "Categories: Food & Dining, Transportation, Shopping, Bills & Utilities, "
                "Subscriptions, Entertainment, Health & Wellness, Travel, Education, "
                "Personal Care, Financial, Income."
            )

        tools_showcase = [t for t in TOOLS if t["name"] not in ("get_spending_for_date", "get_user_date_bounds")]
        tools_personal = TOOLS
        self.active_tools = tools_personal if self.mode == "personal" else tools_showcase

        self.tool_functions = {
            "get_spending_by_category": self.data_manager.get_spending_by_category,
            "get_spending_trend": self.data_manager.get_spending_trend,
            "get_subscriptions": self.data_manager.get_subscriptions,
            "get_spending_summary": self.data_manager.get_spending_summary,
            "get_spending_for_date": self.data_manager.get_spending_for_date,
            "get_user_date_bounds": self.data_manager.get_user_date_bounds,
            "compare_to_average": self.data_manager.compare_to_average,
        }

    def chat(self, user_message: str, user_id: str = "user_0001") -> str:
        full_message = f"[User ID: {user_id}] {user_message}"
        return (self._chat_with_api(full_message, user_id)
                if self.use_api
                else self._chat_demo(user_message, user_id))

    def _chat_with_api(self, message: str, user_id: str) -> str:
        messages = [{"role": "user", "content": message}]
        tools_safe = _to_json_safe(self.active_tools)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024,
            system=self.system_prompt,
            tools=tools_safe,
            messages=_to_json_safe(messages),
        )
        while response.stop_reason == "tool_use":
            tool_use_block = next(
                (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
            )
            if not tool_use_block:
                break
            tool_input = _to_json_safe(dict(tool_use_block.input))
            tool_input["user_id"] = user_id
            tool_result = self.tool_functions[tool_use_block.name](**tool_input)
            messages.append({"role": "assistant", "content": _serialize_assistant_content(response.content)})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id,
                             "content": _json_dumps_tool_result(tool_result)}],
            })
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1024,
                system=self.system_prompt,
                tools=tools_safe,
                messages=_to_json_safe(messages),
            )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return "I couldn't generate a response. Please try again."

    def _chat_demo(self, message: str, user_id: str) -> str:
        msg = message.lower()
        use_all_time = self.data_manager.default_period_days is None

        if any(w in msg for w in ["subscription", "recurring"]):
            r = self.data_manager.get_subscriptions(user_id)
            top = "\n".join(
                [f"  {s['merchant']}: ${s['monthly_cost']:.2f}/month" for s in r["subscriptions"][:3]]
            )
            return (f"Found {r['count']} recurring charges:\n{top}\n"
                    f"Total monthly: ${r['total_monthly']:.2f} (${r['total_yearly']:.2f}/year)")

        if any(w in msg for w in ["category", "breakdown", "where", "spent"]):
            r = self.data_manager.get_spending_by_category(user_id)
            top = "\n".join([f"  {c}: ${a:.2f}" for c, a in list(r["by_category"].items())[:5]])
            return (f"Spending breakdown:\n{top}\n"
                    f"Total: ${r['total_spending']:.2f}. Top category: {r['top_category']}")

        if any(w in msg for w in ["trend", "pattern", "change", "over time"]):
            r = self.data_manager.get_spending_trend(user_id, period="monthly")
            d = "increased" if r["latest_change_percent"] > 0 else "decreased"
            return (f"Spending {d} by {abs(r['latest_change_percent']):.1f}%. "
                    f"Average: ${r['average']:.2f}. Recent: {r['trend']}")

        days_ago = _parse_days_ago(msg)
        if self.mode == "personal":
            if days_ago is not None:
                anchor_date = datetime.now().date()
                target = (anchor_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                r = self.data_manager.get_spending_for_date(user_id, target)
                label = {0: "today", 1: "yesterday", 2: "day before yesterday"}.get(days_ago, f"{days_ago} days ago")
                if not r["has_data"]:
                    return (f"No expenses recorded for **{label}** ({target}). "
                            f"Try asking for **this month** or **total expense** instead.")
                txns = "\n".join(
                    [f"  {t['merchant']}: ${t['amount']:.2f} ({t['category']})" for t in r["transactions"][:5]]
                )
                extra = f"\n{txns}" if txns else ""
                return (f"**{label.capitalize()}** ({target}): **${r['total_spending']:.2f}** "
                        f"across **{r['transaction_count']}** transaction(s).{extra}")
        else:
            if days_ago is not None:
                return ("This is historical demo data, so day-specific queries like 'today' or "
                        "'yesterday' aren't available here.\n\n"
                        "Try: **Last month**, **Last week**, **This month**, **By category**, **Trend**, or **Subscriptions**.")

        if any(w in msg for w in ["this month", "current month", "current expense",
                                  "so far this month", "spending this month"]):
            r = self.data_manager.get_spending_summary(user_id, "this_month")
            label = "this month"
            if r["transaction_count"] == 0 and r["total_spending"] == 0:
                return (f"No expenses recorded for **{label}** yet. "
                        f"Try **total expense** or **last month**.")
            top = list(r["top_categories"].items())[:3]
            return (f"**{label}** so far: **${r['total_spending']:.2f}** across "
                    f"**{r['transaction_count']}** transactions "
                    f"(daily avg ${r['daily_average']:.2f}). Top: {top}")

        if any(w in msg for w in ["summary", "overview", "total", "how much",
                                  "expense", "expenses", "all time", "overall"]):
            if self.mode == "personal":
                r = self.data_manager.get_spending_summary(user_id, "all_time")
                return (f"Your **total expenses** (all time): **${r['total_spending']:.2f}** "
                        f"across **{r['transaction_count']}** transactions. "
                        f"Daily avg: ${r['daily_average']:.2f}. "
                        f"Top: {list(r['top_categories'].items())[:3]}")
            r = self.data_manager.get_spending_summary(user_id, "last_month")
            return (f"Last month: ${r['total_spending']:.2f}, {r['transaction_count']} txns, "
                    f"daily avg ${r['daily_average']:.2f}, change {r['spending_change']:+.1f}%. "
                    f"Top: {list(r['top_categories'].items())[:3]}")

        if self.mode == "personal":
            return (
                "Try: **Total expense**, **Today's expense**, **Yesterday**, "
                "**2 days ago**, **This month**, **By category**, **Trend**, or **Subscriptions**."
            )
        return (
            "Try: **Last month**, **Last week**, **This month**, **By category**, **Trend**, or **Subscriptions**."
        )
