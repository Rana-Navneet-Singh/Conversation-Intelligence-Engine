"""Small overview calculations for parsed messages."""

from collections import Counter


def get_overview(messages):
    if not messages:
        return {"total_messages": 0, "participants": 0, "start": None, "end": None,
                "most_active": None, "active_participants": [], "activity": []}
    counts = Counter(row["sender"] for row in messages)
    dates = sorted(row["timestamp"] for row in messages)
    activity = Counter(value.date().isoformat() for value in dates)
    return {"total_messages": len(messages), "participants": len(counts),
            "start": dates[0], "end": dates[-1], "most_active": counts.most_common(1)[0][0],
            "active_participants": counts.most_common(),
            "activity": sorted(activity.items())}
