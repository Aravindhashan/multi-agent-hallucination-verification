"""Free and open-source web evidence retriever using DuckDuckGo HTML and Wikipedia APIs."""

from datetime import datetime, timezone
import logging
import re
import time
from urllib.parse import parse_qs, quote, unquote, urlparse
from uuid import UUID, uuid4
from typing import Any

from bs4 import BeautifulSoup
import httpx

from app.config import settings
from app.schemas import EvidenceItem
from app.services.evidence_retriever import BaseEvidenceRetriever
from app.services.web_cache import WebCache

logger = logging.getLogger("hallucination_verifier.free_web_retriever")

# Singleton web cache instance
_GLOBAL_WEB_CACHE: WebCache | None = None


def get_web_cache() -> WebCache:
    """Return shared WebCache singleton."""
    global _GLOBAL_WEB_CACHE
    if _GLOBAL_WEB_CACHE is None:
        _GLOBAL_WEB_CACHE = WebCache()
    return _GLOBAL_WEB_CACHE


class FreeWebRetriever(BaseEvidenceRetriever):
    """Retrieve real-world evidence using 100% free, zero-key open web scrapers.

    Architecture:
    1. Check persistent SQLite WebCache. If hit, return cached evidence instantly (0.001s).
    2. Primary: DuckDuckGo HTML Search (`https://html.duckduckgo.com/html/`) via httpx + bs4.
    3. Secondary / Fallback: Wikimedia REST API (`https://en.wikipedia.org/w/api.php`).
    4. Auto-cleans redirect URLs, extracts domain authority tiers, and maps into EvidenceItem.
    """

    DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        cache: WebCache | None = None,
        timeout: float | None = None,
        max_results: int | None = None,
        min_request_interval: float = 0.3,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize retriever with cache, timeout, and custom HTTP client."""
        self.cache = cache if cache is not None else get_web_cache()
        self.timeout = (
            float(timeout) if timeout is not None else float(settings.WEB_SEARCH_TIMEOUT)
        )
        self.max_results = (
            int(max_results)
            if max_results is not None
            else int(settings.WEB_SEARCH_MAX_RESULTS)
        )
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        """Return persistent client with connection pooling and keep-alive."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    def close(self) -> None:
        """Close underlying HTTP client if owned."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "FreeWebRetriever":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        """Enforce polite rate limiting between external network calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> httpx.Response | None:
        """Execute request with keep-alive connection reuse and exponential backoff retry."""
        client = self._get_client()
        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                if method.upper() == "POST":
                    resp = client.post(url, **kwargs)
                else:
                    resp = client.get(url, **kwargs)
                resp.raise_for_status()
                return resp
            except Exception as exc:
                if attempt < max_retries:
                    backoff = 0.6 * (2**attempt)
                    logger.debug(
                        "Request to %s failed (attempt %d/%d): %s; retrying in %.1fs",
                        url,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "Request to %s failed after %d attempts: %s",
                        url,
                        max_retries + 1,
                        exc,
                    )
                    return None

    @staticmethod
    def _extract_duckduckgo_url(raw_href: str) -> str:
        """Unwrap DuckDuckGo redirection link to extract the actual destination URL."""
        if not raw_href:
            return ""
        parsed = urlparse(raw_href)
        # DuckDuckGo wraps links as /l/?uddg=https%3A%2F%2F...
        if "uddg" in parsed.query:
            qs = parse_qs(parsed.query)
            dest = qs.get("uddg", [raw_href])[0]
            return dest
        return raw_href

    @staticmethod
    def _clean_html_text(raw_html: str) -> str:
        """Remove HTML tags and normalize extra spaces from snippets."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def search_duckduckgo(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[EvidenceItem]:
        """Perform free HTML search on DuckDuckGo and parse real-world web snippets."""
        limit = max_results if max_results is not None else self.max_results
        normalized_q = query.strip()
        if not normalized_q:
            return []

        headers = {
            "User-Agent": settings.WEB_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = self._request_with_retry(
            "POST",
            self.DUCKDUCKGO_URL,
            data={"q": normalized_q},
            headers=headers,
            max_retries=1,
            timeout=self.timeout,
        )
        if resp is None:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result")
        if not results:
            logger.debug("DuckDuckGo returned 0 results for query: '%s'", normalized_q)
            return []

        items: list[EvidenceItem] = []
        for rank, r in enumerate(results[:limit]):
            title_elem = r.select_one(".result__title a")
            snippet_elem = r.select_one(".result__snippet")

            if not title_elem:
                continue

            raw_href = title_elem.get("href", "")
            actual_url = self._extract_duckduckgo_url(raw_href)
            title_text = self._clean_html_text(title_elem.get_text())
            snippet_text = (
                self._clean_html_text(snippet_elem.get_text()) if snippet_elem else ""
            )

            # Require minimum substantive passage content
            combined_text = snippet_text if len(snippet_text) >= 20 else f"{title_text}: {snippet_text}"
            if not combined_text.strip():
                continue

            parsed_url = urlparse(actual_url)
            domain = parsed_url.netloc.lower() or None

            # Deterministic relevance heuristic: top result = 0.95, decays by 0.05
            relevance = max(0.50, round(0.95 - (rank * 0.05), 3))

            items.append(
                EvidenceItem(
                    evidence_id=uuid4(),
                    text=combined_text,
                    source_url=actual_url if actual_url else None,
                    domain=domain,
                    source_title=title_text if title_text else "Web Source",
                    retrieval_method="free_web_duckduckgo",
                    relevance_score=relevance,
                )
            )

        return items

    @staticmethod
    def _condense_to_keywords(text: str, max_words: int = 6) -> str:
        """Extract high-information search keywords from a conversational claim or statement."""
        from app.services.evidence_pollution_config import DEFAULT_STOPWORDS

        tokens = [
            w for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", text)
            if w.lower() not in DEFAULT_STOPWORDS
            and len(w) >= 2
            and w.lower() not in ("controversy", "dispute", "refuted", "debunked", "false", "true")
        ]
        if not tokens:
            return text
        special = [w for w in tokens if any(c.isdigit() for c in w) or w.isupper()]
        others = [w for w in tokens if w not in special]
        ordered = special + others
        return " ".join(ordered[:max_words])

    def search_wikipedia(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[EvidenceItem]:
        """Query official Wikimedia REST API for encyclopedia passages and abstracts."""
        limit = max_results if max_results is not None else self.max_results
        normalized_q = query.strip()
        if not normalized_q:
            return []

        headers = {
            "User-Agent": settings.WIKIPEDIA_USER_AGENT,
            "Accept": "application/json",
        }
        # Primary attempt: Query MediaWiki generator search with prop=extracts for plain-text lead sections
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": normalized_q,
            "gsrlimit": str(limit),
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": str(limit),
            "format": "json",
            "utf8": "1",
        }

        resp = self._request_with_retry(
            "GET",
            self.WIKIPEDIA_API_URL,
            params=params,
            headers=headers,
        )
        if resp is None:
            # Fallback attempt: Standard list=search
            params_fallback = {
                "action": "query",
                "list": "search",
                "srsearch": normalized_q,
                "format": "json",
                "utf8": "1",
                "srlimit": str(limit),
            }
            resp = self._request_with_retry(
                "GET",
                self.WIKIPEDIA_API_URL,
                params=params_fallback,
                headers=headers,
            )

        if resp is None:
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Failed to decode Wikipedia JSON: %s", exc)
            return []

        query_data = data.get("query", {})
        items: list[EvidenceItem] = []

        # 1. Parse generator pages with full plain-text lead extracts (un-truncated context)
        pages = query_data.get("pages", {})
        if pages and isinstance(pages, dict):
            sorted_pages = sorted(
                pages.values(),
                key=lambda p: (p.get("index", 999), p.get("pageid", 0)),
            )
            for rank, p in enumerate(sorted_pages[:limit]):
                title = p.get("title", "").strip()
                extract_text = p.get("extract", "").strip()
                if not title and not extract_text:
                    continue

                page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                combined_text = (
                    f"{title}: {extract_text}"
                    if extract_text and not extract_text.startswith(title)
                    else (extract_text or title)
                )
                relevance = max(0.50, round(0.92 - (rank * 0.05), 3))
                items.append(
                    EvidenceItem(
                        evidence_id=uuid4(),
                        text=combined_text,
                        source_url=page_url,
                        domain="en.wikipedia.org",
                        source_title=f"Wikipedia: {title}",
                        retrieval_method="free_web_wikipedia",
                        relevance_score=relevance,
                    )
                )

        # 2. Fallback to list=search snippet results (mock tests and fallback)
        if not items and "search" in query_data:
            search_results = query_data.get("search", [])
            for rank, r in enumerate(search_results[:limit]):
                title = r.get("title", "").strip()
                raw_snippet = r.get("snippet", "")
                snippet_text = self._clean_html_text(raw_snippet)

                if not title and not snippet_text:
                    continue

                page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                combined_text = (
                    f"{title}: {snippet_text}"
                    if snippet_text and not snippet_text.startswith(title)
                    else (snippet_text or title)
                )

                relevance = max(0.50, round(0.90 - (rank * 0.05), 3))
                items.append(
                    EvidenceItem(
                        evidence_id=uuid4(),
                        text=combined_text,
                        source_url=page_url,
                        domain="en.wikipedia.org",
                        source_title=f"Wikipedia: {title}",
                        retrieval_method="free_web_wikipedia",
                        relevance_score=relevance,
                    )
                )

        # 3. If still no items, retry with condensed keyword query
        if not items:
            condensed = self._condense_to_keywords(normalized_q)
            if condensed and condensed != normalized_q:
                params_condensed = dict(params)
                params_condensed["gsrsearch"] = condensed
                resp_condensed = self._request_with_retry(
                    "GET",
                    self.WIKIPEDIA_API_URL,
                    params=params_condensed,
                    headers=headers,
                )
                if resp_condensed is not None:
                    try:
                        data_c = resp_condensed.json()
                        pages_c = data_c.get("query", {}).get("pages", {})
                        if pages_c and isinstance(pages_c, dict):
                            sorted_pages_c = sorted(
                                pages_c.values(),
                                key=lambda p: (p.get("index", 999), p.get("pageid", 0)),
                            )
                            for rank, p in enumerate(sorted_pages_c[:limit]):
                                title = p.get("title", "").strip()
                                extract_text = p.get("extract", "").strip()
                                if not title and not extract_text:
                                    continue
                                page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                                combined_text = (
                                    f"{title}: {extract_text}"
                                    if extract_text and not extract_text.startswith(title)
                                    else (extract_text or title)
                                )
                                relevance = max(0.50, round(0.92 - (rank * 0.05), 3))
                                items.append(
                                    EvidenceItem(
                                        evidence_id=uuid4(),
                                        text=combined_text,
                                        source_url=page_url,
                                        domain="en.wikipedia.org",
                                        source_title=f"Wikipedia: {title}",
                                        retrieval_method="free_web_wikipedia",
                                        relevance_score=relevance,
                                    )
                                )
                    except Exception as exc:
                        logger.warning("Failed to decode condensed Wikipedia JSON: %s", exc)

        return items

    @classmethod
    def strip_meta_instructions(cls, query: str) -> str:
        """Strip internal verification plan, anchor labels, and debate boilerplate from search queries."""
        # Strip "plan: ..."
        cleaned = re.sub(r"\bplan:\s*[^;\n]+(?:;[^;\n]+)*", "", query, flags=re.IGNORECASE)
        # Strip label prefixes like "with dates:", "verify quantity numbers:", "locations:", "dates:", "numbers:"
        cleaned = re.sub(
            r"\b(?:with dates|verify quantity numbers|locations|dates|numbers):\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Strip debate refinement filler
        cleaned = re.sub(
            r"\b(?:corroboration confirmation official archive|alternative findings discrepancy errata|counter-evidence)\b:?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned if cleaned else query.strip()

    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[EvidenceItem]:
        """Unified search with persistent caching and transparent DuckDuckGo -> Wikipedia fallback."""
        normalized_q = self.strip_meta_instructions(query)
        if not normalized_q:
            return []

        limit = max_results if max_results is not None else self.max_results

        # 1. Check SQLite Cache
        cached_data = self.cache.get(normalized_q, engine="free_web")
        if cached_data is not None:
            logger.debug("WebCache hit for query: '%s' (%d items)", normalized_q, len(cached_data))
            items: list[EvidenceItem] = []
            for d in cached_data[:limit]:
                try:
                    items.append(EvidenceItem(**d))
                except Exception as exc:
                    logger.warning("Failed to deserialize cached item: %s", exc)
            if items:
                return items

        # 2. Try DuckDuckGo
        items = self.search_duckduckgo(normalized_q, max_results=limit)

        # 3. Augment or fallback with Wikipedia authoritative lead sections
        if not items:
            logger.info("DuckDuckGo returned 0 results for '%s'; falling back to Wikimedia API.", normalized_q)
            items = self.search_wikipedia(normalized_q, max_results=limit)
        elif len(items) < limit or any(len(it.text) < 150 for it in items):
            # If snippets are short, augment with Wikipedia lead paragraphs to prevent 120-char truncation
            wiki_items = self.search_wikipedia(normalized_q, max_results=min(2, limit))
            seen_urls = {it.source_url for it in items if it.source_url}
            for w in wiki_items:
                if w.source_url not in seen_urls and len(items) < limit:
                    items.append(w)
                    seen_urls.add(w.source_url)

        # 4. Cache valid results
        if items:
            serialized = [item.model_dump(mode="json") for item in items]
            self.cache.set(normalized_q, engine="free_web", results=serialized)

        return items

    def retrieve(self, claim_text: str, stance: str = "supporting") -> list[EvidenceItem]:
        """Retrieve stance-aware evidence implementing BaseEvidenceRetriever."""
        normalized_claim = claim_text.strip()
        if not normalized_claim:
            return []

        normalized_stance = stance.strip().lower()
        if normalized_stance not in ("supporting", "contradicting"):
            raise ValueError("Stance must be 'supporting' or 'contradicting'.")

        if normalized_stance == "contradicting":
            # Targeted counter-query: query with controversy/dispute keyword
            counter_query = f"{normalized_claim} controversy"
            items = self.search(query=counter_query, max_results=self.max_results)
            if not items:
                logger.debug(
                    "Counter-query yielded 0 results; falling back to base claim query: '%s'",
                    normalized_claim,
                )
                items = self.search(query=normalized_claim, max_results=self.max_results)
            return items

        return self.search(query=normalized_claim, max_results=self.max_results)

