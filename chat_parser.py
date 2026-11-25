"""Parse text-only WhatsApp exports into simple message dictionaries."""

import re
from datetime import datetime

PATTERNS = [
    re.compile(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s+-\s+(.*)$"),
    re.compile(r"^\[(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s+(.*)$"),
]
FORMATS = ("%d/%m/%y %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%y %H:%M",
           "%m/%d/%Y %H:%M", "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p",
           "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p",
           "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
           "%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
           "%d/%m/%y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p",
           "%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p")

LEFT_TO_RIGHT_MARKS = "\ufeff\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"


def parse_timestamp(date_text, time_text):
    value = re.sub(r"\s+", " ", f"{date_text} {time_text}").strip().upper()
    for fmt in FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_chat(text):
    """Return messages with timestamp, sender, and text; join multiline text."""
    messages = []
    current = None
    for line in text.splitlines():
        line = line.strip().lstrip(LEFT_TO_RIGHT_MARKS).strip()
        match = None
        for pattern in PATTERNS:
            match = pattern.match(line)
            if match:
                break
        if match:
            timestamp = parse_timestamp(match.group(1), match.group(2))
            body = match.group(3)
            if timestamp is None or "<media omitted>" in body.lower():
                current = None
                continue
            # WhatsApp system notices have no "sender: message" separator.
            if ": " not in body:
                current = None
                continue
            sender, message = body.split(": ", 1)
            if not sender.strip() or not message.strip():
                current = None
                continue
            current = {"timestamp": timestamp, "sender": sender.strip(), "message": message.strip()}
            messages.append(current)
        elif line.strip() and messages and current is not None:
            current["message"] += "\n" + line.strip()
    return messages
