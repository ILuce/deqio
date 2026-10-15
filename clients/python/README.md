# deqio-client

Dependency-free Python SDK for a running [Deqio](https://github.com/ILuce/deqio) decision server. It covers the
public decision routes — `/v1/noul`, `/v1/choice` (`/v1/decision`), `/v1/shared`, `/v1/score`, `/v1/multi`,
`/v1/act`, `/v1/soam`, `/v1/systemone` — and returns the server's JSON responses unchanged, provenance included
(`provenance.provider`, `remote_request_id`, `cost_estimate`). The response shapes are documented by the server's
OpenAPI 3.1 document (`GET /openapi.json`, `deqio openapi`).

```bash
pip install deqio-client        # or: uv add deqio-client
```

```python
from deqio_client import DeqioClient, DeqioError

client = DeqioClient("http://127.0.0.1:8787")

yes_no = client.noul("Checkout works after retry, but payment latency is elevated.", "Is checkout degraded?")
print(yes_no["decision"], yes_no["probabilities"])

pick = client.choice(
    "A customer reports that the API returns HTTP 500 after the latest deployment.",
    "Classify the report.",
    {"bug": "An existing function no longer works", "feature": "A request for new functionality"},
)
print(pick["decision"], pick["provenance"]["provider"]["name"])

severity = client.score(
    "Checkout works after retry, but payment latency is elevated for some customers.",
    {"instructions": "Rate the operational severity.", "criteria": ["low", "medium", "high", "critical"]},
)
print(severity["answers"]["decision"])

try:
    client.choice("state", "Which?", [])
except ValueError as error:          # refused before any request
    print(error)
try:
    client.noul("state", "Is it?", input_policy={"require_complete": True}, contract=True)
except DeqioError as error:          # the server's own rejection: status + detail
    print(error.status, error.detail)
```

Every method mirrors one route and one request shape; nothing is retried, cached or defaulted silently. A server
that cannot be reached raises `DeqioConnectionError`. The package version follows the Deqio release it was built
with (`deqio_client.__version__`).
