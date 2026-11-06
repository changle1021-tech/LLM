"""Record HTTP arrival timestamps before reading the request body."""

import hashlib
from pathlib import Path
import socket
import time
import uuid


def clock_id():
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = uuid.uuid4().hex
    identity = f"{socket.gethostname()}:{boot}"
    return hashlib.sha256(identity.encode()).hexdigest()


CLOCK_ID = clock_id()


class ArrivalMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        arrival_mono = time.monotonic_ns()
        arrival_unix = time.time_ns()

        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"x-llmsim-request-id"
            ),
            None,
        )

        if not request_id:
            return await self.app(scope, receive, send)

        async def stamped_send(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if not key.lower().startswith(b"x-llmsim-")
                ]
                headers.extend([
                    (b"x-llmsim-request-id", request_id),
                    (
                        b"x-llmsim-arrival-monotonic-ns",
                        str(arrival_mono).encode(),
                    ),
                    (
                        b"x-llmsim-arrival-unix-ns",
                        str(arrival_unix).encode(),
                    ),
                    (b"x-llmsim-clock-id", CLOCK_ID.encode()),
                    (b"x-llmsim-receipt-version", b"1"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, stamped_send)
