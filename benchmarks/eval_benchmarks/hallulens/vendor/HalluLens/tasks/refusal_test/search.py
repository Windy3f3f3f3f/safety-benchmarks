# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import certifi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from retry import retry


BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _build_session() -> requests.Session:
    """
    Build a requests session with connection pooling + lightweight HTTP retries.
    This helps reduce transient TLS/connection hiccups.
    """
    total = int(os.getenv("BRAVE_HTTP_RETRIES", "2"))  # small: real retry handled by @retry
    retry_cfg = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_SESSION = _build_session()

# Free plan / unstable network: default 1 req/s is safest
_BRAVE_RPS = int(os.getenv("BRAVE_CALLS_PER_SEC", "1"))
_BRAVE_PERIOD = int(os.getenv("BRAVE_PERIOD_SEC", "1"))
_BRAVE_TIMEOUT = float(os.getenv("BRAVE_TIMEOUT", "60"))  # read timeout
_BRAVE_CONNECT_TIMEOUT = float(os.getenv("BRAVE_CONNECT_TIMEOUT", "10"))

# retry library settings (retries search_brave on SearchException)
_BRAVE_MAX_RETRIES = int(os.getenv("BRAVE_MAX_RETRIES", "12"))
_BRAVE_RETRY_DELAY = float(os.getenv("BRAVE_RETRY_DELAY", "1"))


class SearchException(Exception):
    pass


def batch_search(queries: List[str], max_workers=60, cache=None) -> List[Dict]:
    """
    Return results aligned with `queries`.
    IMPORTANT: never crash the whole job when one query fails.
    """
    # Hard cap concurrency for Brave (avoid SSL EOF / rate limiting)
    cap = int(os.getenv("BRAVE_MAX_WORKERS", "1"))
    max_workers = max(1, min(int(max_workers), cap))

    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(search, q, cache): i for i, q in enumerate(queries)}
        progress = tqdm(total=len(queries), desc="Searching", unit="request")
        for fut in as_completed(future_map):
            i = future_map[fut]
            q = queries[i]
            try:
                results[i] = fut.result()
            except Exception as e:
                # Don't break the whole benchmark; return empty search results for this query
                results[i] = {"query": q, "search_result": [], "error": str(e)}
            progress.update(1)
        progress.close()

    return results


def search(query_string: str, cache=None) -> Dict:
    if cache:
        cached_result = cache.get_item(query_string)
        if cached_result:
            return cached_result

    result = search_brave(query_string)

    if cache:
        cache.set_item(query_string, result)

    return result


@retry(SearchException, tries=_BRAVE_MAX_RETRIES, delay=_BRAVE_RETRY_DELAY, backoff=2, jitter=0.2)
@sleep_and_retry
@limits(calls=_BRAVE_RPS, period=_BRAVE_PERIOD)
def search_brave(query_string: str) -> Dict:
    api_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if not api_key:
        raise SearchException("Missing BRAVE_API_KEY")

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": "HalluLens/RefusalTest",
    }

    try:
        resp = _SESSION.get(
            BRAVE_URL,
            headers=headers,
            params={"result_filter": "web", "q": query_string},
            timeout=(_BRAVE_CONNECT_TIMEOUT, _BRAVE_TIMEOUT),
            verify=certifi.where(),
        )
    except requests.exceptions.RequestException as e:
        # Includes SSLError / ConnectionError / Timeout etc.
        raise SearchException(f"RequestException: {e}")

    # Handle HTTP status explicitly
    if resp.status_code == 429:
        raise SearchException("RATE_LIMITED(HTTP 429)")
    if resp.status_code >= 500:
        raise SearchException(f"BRAVE_HTTP_{resp.status_code}")
    if resp.status_code >= 400:
        # auth / bad request etc -> do not loop forever, but still retry a bit
        raise SearchException(f"BRAVE_HTTP_{resp.status_code}: {resp.text[:200]}")

    try:
        resp_json = resp.json()
    except Exception as e:
        raise SearchException(f"BadJSON: {e}")

    # Brave sometimes returns ErrorResponse inside 200
    if isinstance(resp_json, dict) and resp_json.get("type") == "ErrorResponse":
        code = (resp_json.get("error") or {}).get("code")
        if code == "RATE_LIMITED":
            raise SearchException("RATE_LIMITED(ErrorResponse)")
        # Other error: return empty but keep going
        return {"query": query_string, "search_result": [], "error": str(resp_json)}

    return clean_brave_response(resp_json, top_k=10, fallback_query=query_string)


def clean_brave_response(search_response, top_k=3, fallback_query: str = ""):
    """
    Keep the output schema:
      {"query": <str>, "search_result": [ {"title","link","snippet"}, ... ]}
    """
    selected_keys_infobox = [
        "type",
        "title",
        "url",
        "description",
        "long_desc",
    ]

    query = fallback_query
    if isinstance(search_response, dict):
        qobj = search_response.get("query") or {}
        if isinstance(qobj, dict):
            query = qobj.get("original") or query

    clean_response = []
    if isinstance(search_response, dict) and "mixed" in search_response:
        mixed_results = search_response["mixed"]
        for m in mixed_results.get("main", [])[:top_k]:
            r_type = m.get("type")
            if not r_type:
                continue
            results_block = search_response.get(r_type, {})
            results = results_block.get("results", [])

            if r_type == "web":
                idx = m.get("index", 0)
                if idx < 0 or idx >= len(results):
                    continue
                result = results[idx]
                snippet = ""
                if "description" in result:
                    snippet = result["description"] or ""
                if "extra_snippets" in result and result["extra_snippets"]:
                    snippet += "\n\n" + "\n".join(result["extra_snippets"])
                clean_response.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("url", ""),
                        "snippet": snippet,
                    }
                )

            elif r_type == "faq":
                for q in results:
                    snippet = f"Question: {q.get('question','')}\nAnswer: {q.get('answer','')}"
                    clean_response.append(
                        {"title": q.get("title", ""), "link": q.get("url", ""), "snippet": snippet}
                    )

            elif r_type == "infobox":
                idx = m.get("index", 0)
                if idx < 0 or idx >= len(results):
                    continue
                result = results[idx]
                snippet = f"{result.get('description','')} {result.get('long_desc','')}".strip()
                clean_response.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("url", ""),
                        "snippet": snippet,
                        **{k: result.get(k) for k in selected_keys_infobox if k in result},
                    }
                )

            elif r_type in ("videos", "locations", "news"):
                for q in results[:top_k]:
                    clean_response.append(
                        {"title": q.get("title", ""), "link": q.get("url", ""), "snippet": q.get("description", "")}
                    )

    return {"query": query, "search_result": clean_response}