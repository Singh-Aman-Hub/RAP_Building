# Lumen Cloud API Reference

Base URL: `https://api.lumen.example/v2`. All requests must use HTTPS and carry an API key in the `Authorization` header.

## Authentication

Send the key as a bearer token. Keys are created in the console and shown only once. A key that is not used for 90 days is revoked automatically.

```python
import requests

API_KEY = "lk_live_xxx"          # never commit real keys

def list_projects(page_size=50):
    """Return one page of projects for the current organisation."""
    resp = requests.get(
        "https://api.lumen.example/v2/projects",
        headers={"Authorization": f"Bearer {API_KEY}"},
        params={"page_size": page_size},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["items"]
```

The maximum `page_size` is 200. Larger values are rejected with error code 422.

## Rate limits

Limits depend on your plan and are counted per API key and per minute.

| Plan | Requests per minute | Burst | Monthly request quota |
|------|---------------------|-------|-----------------------|
| Free | 60 | 10 | 100,000 |
| Pro | 600 | 100 | 5,000,000 |
| Enterprise | 6,000 | 1,000 | unlimited |

When you exceed the limit the API answers with status 429 and a `Retry-After` header that tells you how many seconds to wait. Do not retry before that time.

## Errors

Every error response has a JSON body with `code` and `message`.

| HTTP status | Code | Meaning | What to do |
|-------------|------|---------|------------|
| 400 | bad_request | The body is not valid JSON | Fix the request |
| 401 | unauthorized | Missing or revoked API key | Create a new key |
| 404 | not_found | The resource id does not exist | Check the id |
| 409 | conflict | A resource with this name already exists | Choose another name |
| 422 | invalid_field | A field has an invalid value, for example page_size above 200 | Read the message |
| 429 | rate_limited | Too many requests | Wait for Retry-After |
| 503 | unavailable | The platform is overloaded | Retry with backoff |

## Webhooks

Lumen can call your server when a deployment changes state. Webhook requests are signed with HMAC-SHA256. The signature is sent in the `X-Lumen-Signature` header.

```python
import hmac, hashlib

def verify(body: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # constant-time comparison avoids timing attacks
    return hmac.compare_digest(expected, signature)
```

Failed deliveries are retried five times with exponential backoff, starting after 30 seconds. After the fifth failure the webhook is disabled and the project owner gets an email.

## Pagination

List endpoints return at most `page_size` items and a `next_cursor` field. Pass the cursor back as the `cursor` parameter to get the next page. Cursors expire after 15 minutes.

## Deployments

Create a deployment with `POST /deployments`. The call returns immediately with status `queued`. A deployment moves through `queued`, `building`, `live` and, on failure, `failed`. The build timeout is 20 minutes. Use `GET /deployments/{id}` to poll the state every few seconds, or use a webhook instead.
