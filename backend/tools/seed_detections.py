"""Insert sample detections so the chatbot demo has data. Dev use only."""

import random
import sys

from backend.analytics import store

CLASSES = ("plastic", "metal", "paper", "glass", "organic")


def seed(n: int = 60) -> None:
    """Write n random detections to the configured database."""
    store.init_schema()
    items = [
        {
            "class_name": random.choice(CLASSES),
            "confidence": round(random.uniform(0.55, 0.98), 3),
        }
        for _ in range(n)
    ]
    store.record_detections(items, source="seed")
    print(f"seeded {n} detections", file=sys.stderr)


if __name__ == "__main__":
    seed(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
