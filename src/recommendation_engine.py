"""
Recommendation Engine Module - SpendWise AI
Personalized savings recommendations based on spending patterns.
Produced by: 07_recommendation_engine.ipynb
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import List, Optional

import pandas as pd


class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POSITIVE = "positive"


class RecommendationType(Enum):
    OVERSPENDING = "overspending"
    SUBSCRIPTION = "subscription"
    FREQUENCY = "frequency"
    TREND = "trend"
    BUDGET = "budget"
    POSITIVE = "positive"


@dataclass
class Recommendation:
    type: RecommendationType
    priority: Priority
    title: str
    description: str
    potential_savings: float = 0.0
    category: Optional[str] = None
    action_items: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "potential_savings": self.potential_savings,
            "category": self.category,
            "action_items": self.action_items,
        }


class SpendingAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.expenses = df[df["is_expense"]].copy()
        self.expenses["amount"] = self.expenses["amount"].abs()

    def get_category_stats(self, user_id: str) -> pd.DataFrame:
        user_expenses = self.expenses[self.expenses["user_id"] == user_id]
        monthly = (
            user_expenses.groupby(["month", "category"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )
        stats = pd.DataFrame(
            {
                "mean": monthly.mean(),
                "std": monthly.std(),
                "min": monthly.min(),
                "max": monthly.max(),
                "last_month": monthly.iloc[-1] if len(monthly) > 0 else 0,
                "prev_month": monthly.iloc[-2] if len(monthly) > 1 else 0,
            }
        )
        stats["change_pct"] = (
            (stats["last_month"] - stats["prev_month"])
            / stats["prev_month"]
            * 100
        ).fillna(0)
        stats["vs_average_pct"] = (
            (stats["last_month"] - stats["mean"]) / stats["mean"] * 100
        ).fillna(0)
        return stats

    def get_subscription_analysis(self, user_id: str) -> dict:
        user_expenses = self.expenses[self.expenses["user_id"] == user_id]
        merchant_stats = user_expenses.groupby("merchant").agg(
            {
                "amount": ["count", "mean", "std"],
                "category": "first",
                "subcategory": "first",
            }
        )
        merchant_stats.columns = [
            "count",
            "avg_amount",
            "std_amount",
            "category",
            "subcategory",
        ]
        subscriptions = merchant_stats[
            (merchant_stats["count"] >= 3) & (merchant_stats["std_amount"] < 2)
        ].copy()
        by_category = subscriptions.groupby("category").agg(
            {"avg_amount": ["sum", "count"]}
        )
        by_category.columns = ["total", "count"]
        duplicates = by_category[by_category["count"] > 1]
        return {
            "subscriptions": subscriptions.to_dict("index"),
            "total_monthly": subscriptions["avg_amount"].sum(),
            "duplicate_categories": duplicates.to_dict("index"),
            "count": len(subscriptions),
        }

    def get_frequency_analysis(self, user_id: str) -> dict:
        user_expenses = self.expenses[self.expenses["user_id"] == user_id]
        max_date = user_expenses["date"].max()
        recent = user_expenses[user_expenses["date"] > max_date - timedelta(days=30)]
        freq = recent.groupby("subcategory").agg(
            {"amount": ["count", "sum", "mean"]}
        )
        freq.columns = ["count", "total", "avg_per_transaction"]
        freq = freq.sort_values("count", ascending=False)
        high_freq = freq[freq["count"] > 10]
        return {
            "high_frequency": high_freq.to_dict("index"),
            "all_frequencies": freq.head(10).to_dict("index"),
        }

    def get_income_expense_ratio(self, user_id: str) -> dict:
        user_data = self.df[self.df["user_id"] == user_id]
        max_date = user_data["date"].max()
        last_month = user_data[user_data["date"] > max_date - timedelta(days=30)]
        income = last_month[last_month["amount"] > 0]["amount"].sum()
        expenses = last_month[last_month["amount"] < 0]["amount"].abs().sum()
        savings_rate = (income - expenses) / income * 100 if income > 0 else 0
        return {
            "income": income,
            "expenses": expenses,
            "net": income - expenses,
            "savings_rate": savings_rate,
            "expense_ratio": expenses / income * 100 if income > 0 else 100,
        }


def _get_category_tips(category: str) -> List[str]:
    tips = {
        "Food & Dining": [
            "Try meal prepping on weekends",
            "Use grocery pickup to avoid impulse buys",
            "Set a weekly dining out budget",
        ],
        "Transportation": [
            "Consider carpooling or public transit",
            "Combine errands to reduce trips",
        ],
        "Shopping": [
            "Wait 24 hours before non-essential purchases",
            "Use price tracking tools",
        ],
        "Entertainment": [
            "Look for free local events",
            "Use library resources",
        ],
        "Subscriptions": [
            "Audit subscriptions quarterly",
            "Share family plans when possible",
        ],
    }
    return tips.get(category, ["Review spending in this category", "Set a monthly budget"])


class RecommendationEngine:
    OVERSPEND_THRESHOLD = 25
    HIGH_FREQ_THRESHOLD = 12
    LOW_SAVINGS_THRESHOLD = 10
    SUBSCRIPTION_DUPLICATE_THRESHOLD = 2

    def __init__(self, df: pd.DataFrame):
        self.analyzer = SpendingAnalyzer(df)

    def generate_recommendations(self, user_id: str) -> List[Recommendation]:
        recs = []
        recs.extend(self._check_overspending(user_id))
        recs.extend(self._check_subscriptions(user_id))
        recs.extend(self._check_high_frequency(user_id))
        recs.extend(self._check_positive_trends(user_id))
        recs.extend(self._check_budget_health(user_id))
        priority_order = {
            Priority.HIGH: 0,
            Priority.MEDIUM: 1,
            Priority.LOW: 2,
            Priority.POSITIVE: 3,
        }
        recs.sort(key=lambda x: (priority_order[x.priority], -x.potential_savings))
        return recs

    def _check_overspending(self, user_id: str) -> List[Recommendation]:
        recs = []
        stats = self.analyzer.get_category_stats(user_id)
        for category, row in stats.iterrows():
            if row["vs_average_pct"] > self.OVERSPEND_THRESHOLD:
                excess = row["last_month"] - row["mean"]
                priority = (
                    Priority.HIGH
                    if row["vs_average_pct"] > 50
                    else Priority.MEDIUM
                    if row["vs_average_pct"] > 30
                    else Priority.LOW
                )
                recs.append(
                    Recommendation(
                        type=RecommendationType.OVERSPENDING,
                        priority=priority,
                        title=f"High {category} Spending",
                        description=(
                            f"Your {category} spending is {row['vs_average_pct']:.0f}% above average "
                            f"(${row['last_month']:.2f} vs ${row['mean']:.2f} typical)"
                        ),
                        potential_savings=round(excess, 2),
                        category=category,
                        action_items=_get_category_tips(category),
                    )
                )
        return recs

    def _check_subscriptions(self, user_id: str) -> List[Recommendation]:
        recs = []
        analysis = self.analyzer.get_subscription_analysis(user_id)
        for category, data in analysis["duplicate_categories"].items():
            if data["count"] >= self.SUBSCRIPTION_DUPLICATE_THRESHOLD:
                subs_in_cat = [
                    (n, i["avg_amount"])
                    for n, i in analysis["subscriptions"].items()
                    if i["category"] == category
                ]
                potential_savings = min(s[1] for s in subs_in_cat)
                recs.append(
                    Recommendation(
                        type=RecommendationType.SUBSCRIPTION,
                        priority=Priority.MEDIUM,
                        title=f"Multiple {category} Subscriptions",
                        description=(
                            f"You have {data['count']} subscriptions in {category} "
                            f"totaling ${data['total']:.2f}/month"
                        ),
                        potential_savings=round(potential_savings, 2),
                        category=category,
                        action_items=[
                            "Review if you need all services",
                            "Consider consolidating",
                            "Check for family/bundle plans",
                        ],
                    )
                )
        if analysis["total_monthly"] > 150:
            recs.append(
                Recommendation(
                    type=RecommendationType.SUBSCRIPTION,
                    priority=Priority.MEDIUM,
                    title="High Total Subscription Cost",
                    description=(
                        f"Subscriptions total ${analysis['total_monthly']:.2f}/month "
                        f"(${analysis['total_monthly'] * 12:.2f}/year)"
                    ),
                    potential_savings=round(analysis["total_monthly"] * 0.2, 2),
                    action_items=[
                        "Audit subscriptions quarterly",
                        "Cancel unused services",
                        "Look for annual billing discounts",
                    ],
                )
            )
        return recs

    def _check_high_frequency(self, user_id: str) -> List[Recommendation]:
        recs = []
        analysis = self.analyzer.get_frequency_analysis(user_id)
        for subcategory, data in analysis["high_frequency"].items():
            if data["avg_per_transaction"] < 20 and data["count"] > self.HIGH_FREQ_THRESHOLD:
                recs.append(
                    Recommendation(
                        type=RecommendationType.FREQUENCY,
                        priority=Priority.LOW,
                        title=f"Frequent {subcategory} Purchases",
                        description=(
                            f"You made {data['count']} {subcategory} purchases "
                            f"totaling ${data['total']:.2f}"
                        ),
                        potential_savings=round(data["total"] * 0.3, 2),
                        category=subcategory,
                        action_items=[
                            "Consider reducing frequency",
                            "Set a weekly budget for this category",
                        ],
                    )
                )
        return recs

    def _check_positive_trends(self, user_id: str) -> List[Recommendation]:
        recs = []
        stats = self.analyzer.get_category_stats(user_id)
        for category, row in stats.iterrows():
            if row["change_pct"] < -15 and row["prev_month"] > 50:
                savings = row["prev_month"] - row["last_month"]
                recs.append(
                    Recommendation(
                        type=RecommendationType.POSITIVE,
                        priority=Priority.POSITIVE,
                        title=f"Great Job on {category}",
                        description=(
                            f"Your {category} spending decreased by "
                            f"{abs(row['change_pct']):.0f}% from last month"
                        ),
                        potential_savings=0,
                        category=category,
                        action_items=[
                            f"You saved ${savings:.2f} compared to last month",
                            "Keep up the good work!",
                        ],
                    )
                )
        return recs

    def _check_budget_health(self, user_id: str) -> List[Recommendation]:
        recs = []
        ratio = self.analyzer.get_income_expense_ratio(user_id)
        if ratio["savings_rate"] < self.LOW_SAVINGS_THRESHOLD:
            recs.append(
                Recommendation(
                    type=RecommendationType.BUDGET,
                    priority=Priority.HIGH,
                    title="Low Savings Rate",
                    description=(
                        f"Your savings rate is {ratio['savings_rate']:.1f}%. "
                        f"Experts recommend at least 20%."
                    ),
                    potential_savings=round(ratio["income"] * 0.1, 2),
                    action_items=[
                        "Review discretionary spending",
                        "Set up automatic savings transfers",
                        "Create a monthly budget plan",
                    ],
                )
            )
        elif ratio["savings_rate"] > 30:
            recs.append(
                Recommendation(
                    type=RecommendationType.POSITIVE,
                    priority=Priority.POSITIVE,
                    title="Excellent Savings Rate",
                    description=(
                        f"Your savings rate is {ratio['savings_rate']:.1f}% - "
                        f"well above the recommended 20%"
                    ),
                    potential_savings=0,
                    action_items=[
                        "Consider investing your surplus",
                        "You're building great financial habits!",
                    ],
                )
            )
        return recs


class RecommendationService:
    def __init__(self, transactions_df: pd.DataFrame):
        self.engine = RecommendationEngine(transactions_df)

    def get_recommendations(self, user_id: str, limit: int = 10) -> dict:
        recs = self.engine.generate_recommendations(user_id)[:limit]
        by_priority = {p.value: [] for p in Priority}
        for r in recs:
            by_priority[r.priority.value].append(r.to_dict())
        total_savings = sum(r.potential_savings for r in recs)
        return {
            "user_id": user_id,
            "total_recommendations": len(recs),
            "potential_monthly_savings": round(total_savings, 2),
            "potential_yearly_savings": round(total_savings * 12, 2),
            "by_priority": by_priority,
            "recommendations": [r.to_dict() for r in recs],
        }

    def get_top_recommendation(self, user_id: str) -> dict:
        recs = self.engine.generate_recommendations(user_id)
        return recs[0].to_dict() if recs else None

    def get_savings_summary(self, user_id: str) -> dict:
        recs = self.engine.generate_recommendations(user_id)
        actionable = [r for r in recs if r.priority != Priority.POSITIVE]
        by_category = {}
        for r in actionable:
            if r.category:
                by_category[r.category] = (
                    by_category.get(r.category, 0) + r.potential_savings
                )
        return {
            "user_id": user_id,
            "total_monthly_savings": round(
                sum(r.potential_savings for r in actionable), 2
            ),
            "by_category": {
                k: round(v, 2)
                for k, v in sorted(by_category.items(), key=lambda x: -x[1])
            },
            "action_count": len(actionable),
        }


def format_recommendations_text(recommendations: List[dict]) -> str:
    lines = []
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"\n**{rec['title']}**")
        lines.append(f"   {rec['description']}")
        if rec.get("potential_savings", 0) > 0:
            lines.append(f"   Save up to ${rec['potential_savings']:.2f}/month")
        if rec.get("action_items"):
            for action in rec["action_items"][:2]:
                lines.append(f"   -> {action}")
    return "\n".join(lines)