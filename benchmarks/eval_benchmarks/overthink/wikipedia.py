"""
Wikipedia article fetching for OverThink Benchmark.

This module provides functionality to fetch Wikipedia articles for use as
context in the slowdown attack evaluation.
"""

from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Optional dependency import
try:
    import requests
    from bs4 import BeautifulSoup

    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False

from inspect_evals.utils import require_optional_dependency


def _get_request_proxies() -> dict[str, str] | None:
    """Get proxy configuration for HTTP requests.

    Reads from HTTP_REQUEST_PROXY and HTTPS_REQUEST_PROXY environment variables.
    These are separate from the standard HTTP_PROXY/HTTPS_PROXY used by LLM APIs.

    Returns:
        Proxies dict for requests library or None if no proxy configured.
    """
    http_proxy = os.getenv("HTTP_REQUEST_PROXY") or os.getenv("WIKIPEDIA_PROXY")
    https_proxy = os.getenv("HTTPS_REQUEST_PROXY") or os.getenv("WIKIPEDIA_PROXY")

    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        return proxies
    return None


def fetch_wikipedia_article(
    url: str,
    max_retries: int = 3,
    timeout: int = 15,
    proxies: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Fetch Wikipedia article with URL fragment cleaning and retry logic.

    Args:
        url: Wikipedia URL (may contain fragments like #:~:text=)
        max_retries: Number of retry attempts for transient failures
        timeout: Request timeout in seconds
        proxies: Optional proxy configuration for HTTP requests.
            If None, will read from HTTP_REQUEST_PROXY/HTTPS_REQUEST_PROXY
            environment variables. These are separate from the HTTP_PROXY
            used by LLM API calls.

    Returns:
        Tuple of (title, article_text) or (None, "") on failure.
        The article_text is truncated to 5000 characters.

    Raises:
        ImportError: If required dependencies (requests, beautifulsoup4) are not installed.
    """
    require_optional_dependency(
        module="bs4",
        uv_group="overthink",  # Use keyword arg to specify --group
        pip_target="beautifulsoup4",
        display_name="beautifulsoup4",
    )

    # Use provided proxies or get from environment
    if proxies is None:
        proxies = _get_request_proxies()

    # Strip URL fragments that cause 403 errors (e.g., #:~:text=...)
    if "#:~:text=" in url:
        clean_url = url.split("#:~:text=")[0]
    elif "#" in url:
        clean_url = url.split("#")[0]
    else:
        clean_url = url

    headers = {
        "User-Agent": "OverThink/1.0 (Research Project; contact: research@example.com)"
    }

    for attempt in range(max_retries):
        try:
            if not _HAS_DEPENDENCIES:
                raise ImportError(
                    "The 'beautifulsoup4' and 'requests' packages are required for "
                    "Wikipedia fetching. Install them with: pip install beautifulsoup4 requests"
                )

            response = requests.get(
                clean_url, timeout=timeout, headers=headers, proxies=proxies
            )
            response.raise_for_status()

            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract the title of the article
            title_element = soup.find("h1", {"id": "firstHeading"})
            if title_element is None:
                return None, ""
            title = title_element.text

            # Extract the main content of the article
            content_div = soup.find("div", {"id": "mw-content-text"})
            if content_div is None:
                return None, ""
            paragraphs = content_div.find_all("p")

            # Combine all paragraph texts into a single string, max 5000 chars
            article_text = "\n".join(
                [para.text for para in paragraphs if para.text.strip()]
            )
            article_text = article_text[:5000]

            return title, article_text

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter: 2s, 4s, 8s
                wait_time = 2**attempt + (attempt * 0.5)
                time.sleep(wait_time)
            else:
                # Return empty content on final failure
                return None, ""
        except Exception:
            # Return empty content on any other error
            return None, ""

    return None, ""


def extract_description(text: str, tag: str) -> str | None:
    """Extract content between XML-style tags using regex.

    Args:
        text: Text to search
        tag: Tag name to extract (e.g., "DESCRIPTION", "REWRITE")

    Returns:
        Extracted content or None if not found
    """
    search_string = f"<{tag}>(.*?)</{tag}>"
    match = re.search(search_string, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


__all__ = [
    "fetch_wikipedia_article",
    "extract_description",
]
