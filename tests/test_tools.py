from unittest.mock import MagicMock, patch
from tools import search_wikipedia


def _mock_search_response(titles: list[str]):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "query": {"search": [{"title": t} for t in titles]}
    }
    return mock


def _mock_summary_response(extract: str):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"extract": extract}
    return mock


def test_search_wikipedia_returns_title_and_extract():
    with patch("tools.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_search_response(["Python (programming language)"]),
            _mock_summary_response("Python is a high-level language."),
        ]
        result = search_wikipedia("Python programming")
    assert "Python (programming language)" in result
    assert "Python is a high-level language." in result


def test_search_wikipedia_no_results():
    with patch("tools.requests.get") as mock_get:
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"query": {"search": []}}
        mock_get.return_value = mock
        result = search_wikipedia("xyzzy nonexistent topic 12345")
    assert "No Wikipedia article found" in result


def test_search_wikipedia_http_error_raises():
    import requests as req
    with patch("tools.requests.get") as mock_get:
        mock = MagicMock()
        mock.raise_for_status.side_effect = req.HTTPError("404")
        mock_get.return_value = mock
        try:
            search_wikipedia("anything")
            assert False, "Expected HTTPError"
        except req.HTTPError:
            pass
