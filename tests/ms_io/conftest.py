"""Shared fixtures and fake reader helpers for falcon.ms_io tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable


def make_fake_reader(
    entries: Iterable[Any],
    *,
    sources: list[Any] | None = None,
    kwargs_log: list[dict[str, Any]] | None = None,
    name: Any = None,
    error: BaseException | None = None,
) -> type[Any]:
    """Build a Pyteomics-like context-manager reader stub."""
    items = list(entries)

    class _FakeReader:
        def __init__(self, source: Any, *_args: Any, **kwargs: Any) -> None:
            if sources is not None:
                sources.append(source)
            if kwargs_log is not None:
                kwargs_log.append(kwargs)
            if name is not None:
                self.name = name

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def __iter__(self) -> Any:
            yield from items
            if error is not None:
                raise error

    return _FakeReader


@pytest.fixture(scope="session")
def fake_reader_factory():
    """Expose ``make_fake_reader`` as a session fixture."""
    return make_fake_reader
