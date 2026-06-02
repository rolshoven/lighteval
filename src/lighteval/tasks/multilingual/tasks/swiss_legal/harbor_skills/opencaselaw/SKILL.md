---
name: opencaselaw
description: >-
  Query OpenCaseLaw.ch for Swiss court decisions and doctrine. Use when the exam
  question needs BGE/cantonal decisions, statute text, or commentary you cannot
  support from the prompt alone.
allowed-tools: opencaselaw_search_decisions opencaselaw_get_doctrine http_get
---

# OpenCaseLaw.ch — Swiss case law and doctrine lookup

Use the dedicated tools (`opencaselaw_search_decisions`, `opencaselaw_get_doctrine`) before answering LEXam questions that benefit from real Swiss authority.

## When to search

- Open-ended questions needing BGE/cantonal decisions or statute context
- MCQs where answer hinges on a specific Art. X OR / StGB provision
- Any claim about Swiss case law you cannot support from the prompt alone

## Tools

### `opencaselaw_search_decisions`

Search ~970k published Swiss court decisions.

Parameters:
- `query` (required): keywords, statute refs (e.g. `Art. 41 OR Schadenersatz`), party names
- `limit` (optional, default 5, max 20)
- `canton` (optional): e.g. `ZH`, `BE`

### `opencaselaw_get_doctrine`

Statute text plus ranked BGE citations and commentary excerpts for a legal reference.

Parameters:
- `query` (required): e.g. `Art. 41 OR`, `Art. 266g OR`

### `http_get` (fallback)

Only for allowlisted hosts (`mcp.opencaselaw.ch`, `opencaselaw.ch`). Prefer the structured tools above.

## Citation discipline

- Quote holdings briefly; cite decision IDs / canonical URLs returned by the API
- Do not invent BGE numbers or articles not present in tool output
- Swiss exam answers still need `Final Answer: ###X###` for MCQs

## REST reference (for humans)

```bash
curl -s -X POST https://mcp.opencaselaw.ch/api/search_decisions \
  -H "Content-Type: application/json" \
  -d '{"query":"Art. 41 OR Schadenersatz","limit":5}'
```

OpenAPI: https://mcp.opencaselaw.ch/api/openapi.json
