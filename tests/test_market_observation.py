import json
import stat
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from src import market_observation


VALID_PAYLOAD = {
    "error": [],
    "result": {
        "XXBTZUSD": {
            "a": ["82625.0", "1", "1.0"],
            "b": ["82624.0", "1", "1.0"],
            "c": ["82624.5", "0.001"],
        }
    },
}


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_get_market_pair_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("MARKET_OBSERVATION_PAIR", raising=False)

    assert market_observation.get_market_pair() == "XBTUSD"


def test_get_market_pair_normalizes_value(monkeypatch):
    monkeypatch.setenv("MARKET_OBSERVATION_PAIR", " ethusd ")

    assert market_observation.get_market_pair() == "ETHUSD"


@pytest.mark.parametrize("value", ["", "   "])
def test_get_market_pair_rejects_empty_value(monkeypatch, value):
    monkeypatch.setenv("MARKET_OBSERVATION_PAIR", value)

    with pytest.raises(ValueError, match="must not be empty"):
        market_observation.get_market_pair()


@pytest.mark.parametrize("value", ["XBT/USD", "XBT-USD", "XBT USD", "XBT_USD"])
def test_get_market_pair_rejects_non_alphanumeric_value(monkeypatch, value):
    monkeypatch.setenv("MARKET_OBSERVATION_PAIR", value)

    with pytest.raises(ValueError, match="only letters and numbers"):
        market_observation.get_market_pair()


def test_validate_ticker_payload_accepts_valid_payload():
    ticker_name, price = market_observation.validate_ticker_payload(
        VALID_PAYLOAD, "XBTUSD"
    )

    assert ticker_name == "XXBTZUSD"
    assert price == 82624.5


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be a JSON object"),
        ({"error": "bad", "result": {}}, "invalid error field"),
        ({"error": ["EGeneral:Invalid arguments"], "result": {}}, "Kraken returned"),
        ({"error": [], "result": {}}, "No ticker result"),
        ({"error": [], "result": {"XXBTZUSD": []}}, "entry must be an object"),
        (
            {"error": [], "result": {"XXBTZUSD": {}}},
            "missing last-trade price",
        ),
        (
            {"error": [], "result": {"XXBTZUSD": {"c": []}}},
            "missing last-trade price",
        ),
        (
            {"error": [], "result": {"XXBTZUSD": {"c": ["not-a-price"]}}},
            "price is invalid",
        ),
        (
            {"error": [], "result": {"XXBTZUSD": {"c": ["0"]}}},
            "price must be positive",
        ),
    ],
)
def test_validate_ticker_payload_rejects_invalid_payloads(payload, message):
    with pytest.raises(ValueError, match=message):
        market_observation.validate_ticker_payload(payload, "XBTUSD")


def test_fetch_public_ticker_returns_valid_payload_and_uses_public_url(monkeypatch):
    captured = {}

    def fake_urlopen(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(json.dumps(VALID_PAYLOAD).encode("utf-8"))

    monkeypatch.setattr(market_observation, "urlopen", fake_urlopen)

    payload = market_observation.fetch_public_ticker("XBTUSD")

    assert payload == VALID_PAYLOAD
    assert captured["url"] == (
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
    )
    assert captured["timeout"] == 15


def test_fetch_public_ticker_rejects_non_200_response(monkeypatch):
    monkeypatch.setattr(
        market_observation,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"{}", status=503),
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        market_observation.fetch_public_ticker("XBTUSD")


def test_fetch_public_ticker_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(
        market_observation,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"not-json"),
    )

    with pytest.raises(RuntimeError, match="not valid JSON"):
        market_observation.fetch_public_ticker("XBTUSD")


def test_fetch_public_ticker_handles_url_error(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        raise URLError("offline test")

    monkeypatch.setattr(market_observation, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="network failure: offline test"):
        market_observation.fetch_public_ticker("XBTUSD")


def test_fetch_public_ticker_handles_http_error(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        raise HTTPError(
            "https://api.kraken.com/0/public/Ticker",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(market_observation, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 429"):
        market_observation.fetch_public_ticker("XBTUSD")


def test_write_observation_creates_restricted_record(monkeypatch, tmp_path):
    observation_dir = tmp_path / "market_observations"
    monkeypatch.setattr(market_observation, "DATA_DIR", observation_dir)
    monkeypatch.setattr(
        market_observation, "utc_timestamp", lambda: "20261010T020000Z"
    )
    monkeypatch.setattr(
        market_observation, "utc_iso_timestamp", lambda: "2026-10-10T02:00:00Z"
    )

    output_path, record = market_observation.write_observation(
        "XBTUSD", VALID_PAYLOAD
    )

    assert output_path == (
        observation_dir / "kraken_ticker_XBTUSD_20261010T020000Z.json"
    )
    assert output_path.exists()
    assert stat.S_IMODE(observation_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600

    saved_record = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_record == record
    assert record["schema_version"] == 1
    assert record["mode"] == "PUBLIC_MARKET_DATA_ONLY"
    assert record["trading_permitted"] is False
    assert record["order_submission_attempted"] is False
    assert record["requested_pair"] == "XBTUSD"
    assert record["kraken_result_pair"] == "XXBTZUSD"
    assert record["last_trade_price"] == 82624.5
    assert record["observed_at_utc"] == "2026-10-10T02:00:00Z"


def test_write_observation_fails_before_creating_directory_for_invalid_payload(
    monkeypatch, tmp_path
):
    observation_dir = tmp_path / "market_observations"
    monkeypatch.setattr(market_observation, "DATA_DIR", observation_dir)

    invalid_payload = {"error": [], "result": {}}

    with pytest.raises(ValueError, match="No ticker result"):
        market_observation.write_observation("XBTUSD", invalid_payload)

    assert not observation_dir.exists()


def test_main_reports_success_without_network(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "record.json"
    record = {
        "requested_pair": "XBTUSD",
        "last_trade_price": 82624.5,
    }

    monkeypatch.setattr(market_observation, "get_market_pair", lambda: "XBTUSD")
    monkeypatch.setattr(
        market_observation, "fetch_public_ticker", lambda pair: VALID_PAYLOAD
    )
    monkeypatch.setattr(
        market_observation,
        "write_observation",
        lambda pair, payload: (output_path, record),
    )
    monkeypatch.setattr(market_observation, "PROJECT_ROOT", tmp_path)

    assert market_observation.main() == 0

    captured = capsys.readouterr()
    assert "MARKET_OBSERVATION_PASSED" in captured.out
    assert "pair=XBTUSD" in captured.out
    assert "last_trade_price=82624.5" in captured.out
    assert "trading_permitted=false" in captured.out
    assert "order_submission_attempted=false" in captured.out
    assert captured.err == ""


def test_main_fails_closed_and_writes_error_to_stderr(monkeypatch, capsys):
    def fail():
        raise ValueError("invalid observation pair")

    monkeypatch.setattr(market_observation, "get_market_pair", fail)

    assert market_observation.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "MARKET_OBSERVATION_FAILED: invalid observation pair" in captured.err
