"""
A hand-written ASGI application. Run it with a real ASGI server:

    cd 09_api_development/02_wsgi_asgi_by_hand
    uvicorn asgi_app:app --port 8000
    curl -i http://127.0.0.1:8000/hi

`asgi_app:app` = "in the file asgi_app.py, use the object called app".
"""


async def app(scope, receive, send):
    """ASGI contract: an ASYNC callable taking (scope, receive, send).

    scope   : dict describing the connection (type, method, path, headers ...)
    receive : `await receive()` gets incoming events (e.g. request body chunks)
    send    : `await send({...})` pushes response events to the server
    Because it is async, one worker can juggle thousands of waiting requests (e.g. waiting on a DB).
    """
    if scope["type"] != "http":                          # ASGI also covers websockets/lifespan; we ignore them
        return
    body = f"Hello from ASGI, path={scope['path']}".encode()
    # A response is TWO events: first the status+headers, then the body.
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": body})
