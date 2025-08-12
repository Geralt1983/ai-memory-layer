from datetime import datetime
from typing import Optional


def parse_timestamp(timestamp_str: Optional[str]) -> datetime:
    """Parse timestamp string in ISO-8601 or Unix format.

    Args:
        timestamp_str: Timestamp as ISO-8601 string or Unix seconds.

    Returns:
        Parsed ``datetime`` without timezone information. If parsing fails or
        ``timestamp_str`` is ``None``, the current time is returned.
    """
    if not timestamp_str:
        return datetime.now()

    try:
        if 'T' in timestamp_str:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=None)
        return datetime.fromtimestamp(float(timestamp_str))
    except Exception:
        return datetime.now()
