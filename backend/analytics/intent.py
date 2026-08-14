"""Offline keyword-based intent matching for judge questions."""

from backend.analytics import data

CLASSES = ("plastic", "metal", "paper", "glass", "organic")


def answer(question: str) -> str:
    """Return a plain-text answer for a natural-language question."""
    q = question.lower()

    # class-specific count: "how many plastic?"
    for cls in CLASSES:
        if cls in q:
            n = data.count_by_class().get(cls, 0)
            return f"{n} {cls} items detected so far."

    if "total" in q or "how many" in q or "count" in q:
        return f"{data.total_count()} items detected in total."

    if "most" in q or "common" in q or "top" in q:
        top = data.most_common_class()
        return f"Most common class: {top}." if top else "No detections yet."

    if "all" in q or "breakdown" in q or "each" in q:
        counts = data.count_by_class()
        return ", ".join(f"{k}: {v}" for k, v in counts.items()) or "No detections yet."

    return "I can report counts per class, totals, and the most common class."
