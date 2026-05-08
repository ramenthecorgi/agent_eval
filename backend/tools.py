import requests

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_HEADERS = {"User-Agent": "agent-eval/1.0 (educational project)"}


def search_wikipedia(query: str) -> str:
    search_resp = requests.get(
        _SEARCH_URL,
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=_HEADERS,
        timeout=10,
    )
    search_resp.raise_for_status()
    results = search_resp.json().get("query", {}).get("search", [])
    if not results:
        return f"No Wikipedia article found for '{query}'."

    title = results[0]["title"]
    summary_resp = requests.get(
        _SUMMARY_URL.format(title=requests.utils.quote(title, safe="")),
        headers=_HEADERS,
        timeout=10,
    )
    summary_resp.raise_for_status()
    extract = summary_resp.json().get("extract", "No summary available.")
    return f"{title}: {extract}"
