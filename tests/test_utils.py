from datetime import datetime
from core.utils import parse_timestamp


def test_parse_timestamp_iso():
    ts = parse_timestamp("2023-01-01T12:00:00Z")
    assert ts == datetime(2023, 1, 1, 12, 0, 0)


def test_parse_timestamp_unix():
    unix = 1700000000
    ts = parse_timestamp(str(unix))
    assert ts == datetime.fromtimestamp(unix)


def test_parse_timestamp_invalid():
    before = datetime.now()
    ts = parse_timestamp("invalid")
    after = datetime.now()
    assert before <= ts <= after
