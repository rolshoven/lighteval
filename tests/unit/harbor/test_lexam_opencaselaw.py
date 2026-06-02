import json
from unittest.mock import MagicMock, patch

from lighteval.tasks.multilingual.tasks.swiss_legal.lexam_opencaselaw import (
    get_doctrine,
    http_get_json,
    run_opencaselaw_tool,
    search_decisions,
)


@patch("lighteval.tasks.multilingual.tasks.swiss_legal.lexam_opencaselaw.urllib.request.urlopen")
def test_search_decisions_posts_to_api(mock_urlopen):
    response = MagicMock()
    response.read.return_value = json.dumps({"hits": [{"id": "4A_1/2023"}]}).encode()
    response.__enter__.return_value = response
    mock_urlopen.return_value = response

    result = search_decisions("Art. 41 OR", limit=3)
    assert result["hits"][0]["id"] == "4A_1/2023"

    request = mock_urlopen.call_args.args[0]
    assert request.full_url.endswith("/api/search_decisions")
    assert json.loads(request.data.decode())["query"] == "Art. 41 OR"


@patch("lighteval.tasks.multilingual.tasks.swiss_legal.lexam_opencaselaw.urllib.request.urlopen")
def test_get_doctrine(mock_urlopen):
    response = MagicMock()
    response.read.return_value = json.dumps({"article": "Art. 41 OR"}).encode()
    response.__enter__.return_value = response
    mock_urlopen.return_value = response

    result = get_doctrine("Art. 41 OR")
    assert result["article"] == "Art. 41 OR"


def test_http_get_rejects_unknown_host():
    result = http_get_json("https://example.com/evil")
    assert result["error"] == "host_not_allowed"


def test_run_opencaselaw_tool_dispatches_search():
    with patch(
        "lighteval.tasks.multilingual.tasks.swiss_legal.lexam_opencaselaw.search_decisions",
        return_value={"ok": True},
    ) as mock_search:
        output = run_opencaselaw_tool("opencaselaw_search_decisions", {"query": "Mietrecht", "limit": 2})
        mock_search.assert_called_once_with(query="Mietrecht", limit=2, canton=None)
        assert '"ok": true' in output.lower()
