"""Provider registry: stores factories, not instances.

register(kind, name, factory) → create(kind, name, **kwargs) builds a
fresh instance per call, letting callers inject config-derived parameters.
"""

import threading

_lock = threading.Lock()
_factories: dict[str, dict[str, callable]] = {}


def register(kind: str, name: str, factory: callable) -> None:
    with _lock:
        _factories.setdefault(kind, {})[name] = factory


def create(kind: str, name: str, **kwargs):
    factory = get_factory(kind, name)
    return factory(**kwargs)


def available(kind: str) -> list:
    with _lock:
        return sorted(_factories.get(kind, {}))


def get_factory(kind: str, name: str):
    with _lock:
        bucket = _factories.get(kind, {})
        if name not in bucket:
            raise KeyError(f"no provider {kind}/{name!r} registered")
        return bucket[name]
