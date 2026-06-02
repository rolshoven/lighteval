"""OpenCaseLaw.ch REST helpers for LexamHarborAgent tools.

API docs: https://mcp.opencaselaw.ch/api/openapi.json
Corpus: https://opencaselaw.ch/
"""

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

OPENCASELAW_API_BASE = "https://mcp.opencaselaw.ch/api"
ALLOWED_HOSTS = frozenset({"mcp.opencaselaw.ch", "opencaselaw.ch", "www.opencaselaw.ch"})


def _post_json(path: str, payload: dict[str, Any], timeout_seconds: float = 60.0) -> dict[str, Any]:
    url = f"{OPENCASELAW_API_BASE}/{path.lstrip('/')}"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}", "detail": detail[:4000]}
    except urllib.error.URLError as exc:
        return {"error": "request_failed", "detail": str(exc)}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "detail": raw[:4000]}
    if isinstance(parsed, dict):
        return parsed
    return {"results": parsed}


def search_decisions(
    query: str,
    *,
    limit: int = 5,
    canton: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "limit": max(1, min(limit, 20))}
    if canton:
        payload["canton"] = canton
    return _post_json("search_decisions", payload)


def get_doctrine(query: str) -> dict[str, Any]:
    return _post_json("get_doctrine", {"query": query})


def http_get_json(url: str, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
    """GET JSON from an allowlisted OpenCaseLaw host (escape hatch)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"error": "unsupported_scheme", "detail": parsed.scheme}
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        return {
            "error": "host_not_allowed",
            "detail": f"Only {sorted(ALLOWED_HOSTS)} are permitted.",
        }

    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}", "detail": detail[:4000]}
    except urllib.error.URLError as exc:
        return {"error": "request_failed", "detail": str(exc)}

    try:
        parsed_json = json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw[:8000]}
    if isinstance(parsed_json, dict):
        return parsed_json
    return {"results": parsed_json}


def run_opencaselaw_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "opencaselaw_search_decisions":
        result = search_decisions(
            query=str(arguments.get("query", "")),
            limit=int(arguments.get("limit", 5)),
            canton=arguments.get("canton"),
        )
    elif tool_name == "opencaselaw_get_doctrine":
        result = get_doctrine(query=str(arguments.get("query", "")))
    elif tool_name == "http_get":
        result = http_get_json(url=str(arguments.get("url", "")))
    else:
        result = {"error": "unknown_tool", "tool": tool_name}
    return json.dumps(result, indent=2, ensure_ascii=False)[:12000]
