"""Shared threaded aiohttp test server helper (runs its own loop+thread)."""

import asyncio
import threading


class ThreadedAiohttpServer:
    """Starts an aiohttp web.Application in a private loop/thread.

    Usage: app class exposing .r = web.Application(); .url after start.
    """

    def __init__(self, app):
        from aiohttp import web_runner

        self.obj = app
        self.app = app.r
        self.loop = asyncio.new_event_loop()
        self.url = None
        self.started = threading.Event()

        def run():
            async def _start():
                runner = web_runner.AppRunner(self.app)
                await runner.setup()
                site = web_runner.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                port = runner.addresses[0][1]
                return f"http://127.0.0.1:{port}"

            self.url = self.loop.run_until_complete(_start())
            self.started.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        if not self.started.wait(timeout=5):
            raise RuntimeError("test server failed to start")

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
