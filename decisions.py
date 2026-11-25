"""Explainable decision and commitment extraction for short conversations."""

import re
from datetime import timedelta


PROPOSAL = re.compile(r"\b(should we|we should|let's|lets|how about|could we|shall we|i suggest|suggest we|propose)\b", re.I)
REJECT = re.compile(r"\b(no|can't|cannot|won't|will not|disagree|rather not|not possible|reject)\b", re.I)
AGREE = re.compile(r"\b(agreed|i agree|yes|okay|ok|sounds good|works for me|haan|theek hai|done)\b", re.I)
CONFIRM = re.compile(r"\b(decided|confirmed|we will|we'll|final plan|decision is)\b", re.I)
SELF_TASK = re.compile(r"\b(i can|i will|i'll|i shall|i am going to|i'm going to)\s+(.+)", re.I)
DEADLINE = re.compile(r"\b(?:by|before|on|at)\s+((?:\w+\s+){0,3}(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|today))\b", re.I)


def _valid(row):
    return isinstance(row, dict) and isinstance(row.get("message"), str) and row["message"].strip()


def _nearby(first, second):
    a, b = first.get("timestamp"), second.get("timestamp")
    if a is None or b is None:
        return True
    try:
        return timedelta(0) <= b - a <= timedelta(minutes=30)
    except TypeError:
        return False


def extract_decisions(messages):
    """Return only explicit proposals, responses, confirmations, and commitments.

    Agreement is attached only when a nearby explicit reply follows an open
    proposal. Each item retains original, timestamped evidence for review.
    """
    rows = [row for row in messages if _valid(row)]
    items = []
    for index, row in enumerate(rows):
        text = row["message"].strip()
        if PROPOSAL.search(text):
            evidence = [{"role": "proposal", **row}]
            status, agreeers = "proposed", []
            for reply in rows[index + 1:index + 5]:
                if not _nearby(row, reply):
                    break
                reply_text = reply["message"].strip()
                if REJECT.search(reply_text):
                    status = "rejected"
                    evidence.append({"role": "rejection", **reply})
                    break
                if AGREE.search(reply_text):
                    status = "agreed"
                    agreeers.append(reply.get("sender", "Unknown"))
                    evidence.append({"role": "agreement", **reply})
            deadline = DEADLINE.search(text)
            items.append({"type": "proposal", "decision": text, "status": status,
                          "proposer": row.get("sender", "Unknown"),
                          "agreed_by": agreeers, "responsible": None,
                          "deadline": deadline.group(1) if deadline else None,
                          "confidence": "medium" if status != "proposed" else "low",
                          "evidence": evidence, "support": row,
                          "cue": "proposal", "author": row.get("sender", "Unknown"),
                          "nearby_speakers": agreeers})
        task = SELF_TASK.search(text)
        if task:
            deadline = DEADLINE.search(text)
            items.append({"type": "commitment", "decision": task.group(2).strip(),
                          "status": "confirmed", "proposer": row.get("sender", "Unknown"),
                          "agreed_by": [], "responsible": row.get("sender", "Unknown"),
                          "deadline": deadline.group(1) if deadline else None,
                          "confidence": "high", "evidence": [{"role": "commitment", **row}],
                          "support": row, "cue": "commitment", "author": row.get("sender", "Unknown"),
                          "nearby_speakers": []})
        elif (CONFIRM.search(text) and not PROPOSAL.search(text)
              and not AGREE.search(text)
              and not any(PROPOSAL.search(previous["message"])
                          and _nearby(previous, row) for previous in rows[max(0, index - 4):index])):
            items.append({"type": "confirmed decision", "decision": text,
                          "status": "confirmed", "proposer": row.get("sender", "Unknown"),
                          "agreed_by": [], "responsible": None,
                          "deadline": (DEADLINE.search(text).group(1) if DEADLINE.search(text) else None),
                          "confidence": "medium", "evidence": [{"role": "confirmation", **row}],
                          "support": row, "cue": "confirmation", "author": row.get("sender", "Unknown"),
                          "nearby_speakers": []})
    return items
