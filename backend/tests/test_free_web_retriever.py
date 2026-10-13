"""Unit tests for the FreeWebRetriever service using mocked HTTP responses."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.schemas import EvidenceItem
from app.services.free_web_retriever import FreeWebRetriever
from app.services.web_cache import WebCache


MOCK_DDG_HTML = """
<!DOCTYPE html>
<html>
<body>
    <div class="result results_links results_links_deep web-result">
        <div class="result__body links_main links_deep">
            <h2 class="result__title">
                <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fscience.nasa.gov%2Fmissions%2Fwebb%2Fnasas-webb-detects-co2&amp;rut=1">
                    NASA Webb Detects Carbon Dioxide in Exoplanet Atmosphere
                </a>
            </h2>
            <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fscience.nasa.gov%2Fmissions%2Fwebb%2Fnasas-webb-detects-co2">
                NASA James Webb Space Telescope has captured the first clear evidence for carbon dioxide in the atmosphere of WASP-39b.
            </a>
        </div>
    </div>
    <div class="result results_links results_links_deep web-result">
        <div class="result__body links_main links_deep">
            <h2 class="result__title">
                <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41586-022-05269-w">
                    Identification of carbon dioxide in an exoplanet atmosphere | Nature
                </a>
            </h2>
            <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41586-022-05269-w">
                Transmission spectroscopy of WASP-39b shows a prominent carbon dioxide absorption feature at 4.3 micrometres.
            </a>
        </div>
    </div>
</body>
</html>
"""

MOCK_WIKI_JSON = {
    "query": {
        "search": [
            {
                "title": "WASP-39b",
                "snippet": '<span class="searchmatch">WASP-39b</span> is a hot Jupiter extrasolar planet discovered in February 2011, containing high amounts of carbon dioxide.',
                "timestamp": "2024-03-15T12:00:00Z",
            },
            {
                "title": "James Webb Space Telescope",
                "snippet": 'The <span class="searchmatch">James Webb Space Telescope</span> (JWST) is an infrared space observatory launched in 2021.',
                "timestamp": "2024-04-01T08:30:00Z",
            },
        ]
    }
}


@pytest.fixture
def isolated_cache(tmp_path: Path) -> WebCache:
    """Provide an isolated WebCache backed by a temporary SQLite database."""
    db_file = tmp_path / "test_retriever_cache.db"
    return WebCache(db_path=db_file, default_ttl_seconds=3600)


def test_extract_duckduckgo_url():
    """Verify unwrapping uddg redirection parameters from DuckDuckGo links."""
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fscience.nasa.gov%2Fmissions%2Fwebb&rut=1"
    extracted = FreeWebRetriever._extract_duckduckgo_url(wrapped)
    assert extracted == "https://science.nasa.gov/missions/webb"

    plain = "https://example.org/article"
    assert FreeWebRetriever._extract_duckduckgo_url(plain) == plain
    assert FreeWebRetriever._extract_duckduckgo_url("") == ""


def test_clean_html_text():
    """Verify HTML tags and spacing are normalized."""
    raw = '<b>Hello</b>  <span class="match">World</span>\n\tTest'
    cleaned = FreeWebRetriever._clean_html_text(raw)
    assert cleaned == "Hello World Test"
    assert FreeWebRetriever._clean_html_text("") == ""


def test_search_duckduckgo_mocked(isolated_cache: WebCache):
    """Verify DuckDuckGo HTML scraping with mocked HTTP response."""
    retriever = FreeWebRetriever(cache=isolated_cache, min_request_interval=0.0)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = MOCK_DDG_HTML
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp
    retriever._client = mock_client

    results = retriever.search_duckduckgo("JWST WASP-39b", max_results=5)
    assert len(results) == 2

    # Check first item
    first = results[0]
    assert "NASA Webb Detects Carbon Dioxide" in first.source_title
    assert "science.nasa.gov" in first.domain
    assert first.source_url == "https://science.nasa.gov/missions/webb/nasas-webb-detects-co2"
    assert "carbon dioxide in the atmosphere of WASP-39b" in first.text
    assert first.relevance_score == 0.95
    assert first.retrieval_method == "free_web_duckduckgo"

    # Check second item
    second = results[1]
    assert "nature.com" in second.domain
    assert second.relevance_score == 0.90


def test_search_wikipedia_mocked(isolated_cache: WebCache):
    """Verify Wikimedia REST API extraction with mocked JSON response."""
    retriever = FreeWebRetriever(cache=isolated_cache, min_request_interval=0.0)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_WIKI_JSON
    mock_resp.raise_for_status.return_value = None
    mock_client.get.return_value = mock_resp
    retriever._client = mock_client

    results = retriever.search_wikipedia("WASP-39b", max_results=5)
    assert len(results) == 2

    first = results[0]
    assert first.source_title == "Wikipedia: WASP-39b"
    assert first.domain == "en.wikipedia.org"
    assert first.source_url == "https://en.wikipedia.org/wiki/WASP-39b"
    assert "hot Jupiter extrasolar planet" in first.text
    # Ensure HTML searchmatch spans were stripped
    assert "<span" not in first.text
    assert first.retrieval_method == "free_web_wikipedia"


def test_search_fallback_on_duckduckgo_failure(isolated_cache: WebCache):
    """Verify transparent fallback to Wikipedia when DuckDuckGo fails."""
    retriever = FreeWebRetriever(cache=isolated_cache, min_request_interval=0.0)

    mock_client = MagicMock()
    # DDG fails with connection error
    mock_client.post.side_effect = Exception("HTTP 403 Forbidden")
    # Wikipedia succeeds
    wiki_resp = MagicMock()
    wiki_resp.status_code = 200
    wiki_resp.json.return_value = MOCK_WIKI_JSON
    wiki_resp.raise_for_status.return_value = None
    mock_client.get.return_value = wiki_resp
    retriever._client = mock_client

    results = retriever.search("WASP-39b", max_results=5)
    assert len(results) == 2
    assert results[0].retrieval_method == "free_web_wikipedia"


def test_search_caching_integration(isolated_cache: WebCache):
    """Verify that identical queries are served from cache without outbound HTTP calls."""
    retriever = FreeWebRetriever(cache=isolated_cache, min_request_interval=0.0)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = MOCK_DDG_HTML
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp
    retriever._client = mock_client

    # First call: hits client.post
    items1 = retriever.search("JWST WASP-39b", max_results=5)
    assert len(items1) == 2
    assert mock_client.post.call_count == 1

    # Second call with same query: should hit SQLite cache, call_count remains 1
    items2 = retriever.search("JWST WASP-39b", max_results=5)
    assert len(items2) == 2
    assert mock_client.post.call_count == 1
    assert items2[0].source_url == items1[0].source_url


def test_retrieve_stance_adaptation(isolated_cache: WebCache):
    """Verify stance-aware search query adaptation."""
    retriever = FreeWebRetriever(cache=isolated_cache, min_request_interval=0.0)

    with patch.object(retriever, "search", return_value=[MagicMock()]) as mock_search:
        # Supporting stance
        retriever.retrieve("The Earth orbits the Sun", stance="supporting")
        mock_search.assert_called_with(query="The Earth orbits the Sun", max_results=retriever.max_results)

        # Contradicting stance
        retriever.retrieve("The Earth orbits the Sun", stance="contradicting")
        assert "controversy" in mock_search.call_args[1]["query"] or "dispute" in mock_search.call_args[1]["query"]


def test_retrieve_invalid_stance(isolated_cache: WebCache):
    """Verify that an unsupported stance raises ValueError."""
    retriever = FreeWebRetriever(cache=isolated_cache, min_request_interval=0.0)
    with pytest.raises(ValueError, match="Stance must be 'supporting' or 'contradicting'"):
        retriever.retrieve("Some claim", stance="neutral")
