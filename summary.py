"""Chronological, source-grounded summaries for a topic conversation."""

from datetime import datetime

from decisions import extract_decisions


def summarize(messages, sentence_count=8, topic_title=None):
    """Describe the conversation by ordering and attributing the original messages."""
    del sentence_count  # Kept for compatibility with earlier callers.
    rows = [row for row in messages if isinstance(row, dict)
            and isinstance(row.get("message"), str) and row["message"].strip()]
    rows.sort(key=lambda row: row.get("timestamp") or datetime.min)
    if not rows:
        return {"title": topic_title or "Topic summary", "text": "No messages to summarize.",
                "timeline": [], "subtopics": [], "participants": [], "dates": [],
                "conclusion": "No clear conclusion identified.", "tasks": [],
                "important_messages": []}
    people = sorted({row.get("sender", "Unknown") for row in rows})
    dates = sorted({row["timestamp"].date().isoformat() for row in rows
                    if hasattr(row.get("timestamp"), "date")})
    items = extract_decisions(rows)
    evidence_roles = {}
    for item in items:
        for evidence in item["evidence"]:
            evidence_roles[id(evidence)] = evidence["role"]
    timeline = []
    for row in rows:
        role = "message"
        for item in items:
            for evidence in item["evidence"]:
                if evidence.get("timestamp") == row.get("timestamp") and evidence.get("sender") == row.get("sender") and evidence.get("message") == row.get("message"):
                    role = evidence["role"]
                    break
        timeline.append({"sender": row.get("sender", "Unknown"),
                         "timestamp": row.get("timestamp"), "role": role,
                         "text": row["message"].strip()})
    outcomes = [item for item in items if item["status"] in {"agreed", "confirmed", "rejected"}]
    if outcomes:
        latest = outcomes[-1]
        conclusion = f"{latest['status'].capitalize()}: {latest['decision']}"
    else:
        conclusion = "No clear conclusion identified. The discussion contains no explicit agreement or confirmation."
    tasks = [{"responsible": item["responsible"] or "Unknown",
              "task": item["decision"], "deadline": item["deadline"],
              "status": item["status"]} for item in items
             if item["type"] == "commitment"]
    text = "\n".join(f"{row['sender']} ({row['role']}): {row['text']}" for row in timeline)
    return {"title": topic_title or "Topic summary", "text": text, "timeline": timeline,
            "subtopics": [], "participants": people, "dates": dates,
            "conclusion": conclusion, "tasks": tasks, "important_messages": rows}
