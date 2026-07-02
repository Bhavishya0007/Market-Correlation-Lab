"""Pytest configuration.

Adds src/ to sys.path so tests import the modules directly, and installs
minimal stubs for streamlit and yfinance when they aren't importable.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _install_streamlit_stub() -> None:
    st = types.ModuleType("streamlit")

    def cache_data(*d_args, **d_kwargs):
        if d_args and callable(d_args[0]) and not d_kwargs:
            fn = d_args[0]
            fn.clear = lambda: None
            return fn

        def decorator(fn):
            fn.clear = lambda: None
            return fn
        return decorator

    class _Secrets(dict):
        def __getitem__(self, key):
            raise KeyError(key)

    st.cache_data = cache_data
    st.secrets = _Secrets()
    sys.modules["streamlit"] = st


def _install_yfinance_stub() -> None:
    yf = types.ModuleType("yfinance")

    def download(*args, **kwargs):
        raise RuntimeError("yfinance stub: network calls not available in tests")

    yf.download = download
    sys.modules["yfinance"] = yf


try:
    import streamlit  # noqa: F401
except ImportError:
    _install_streamlit_stub()

try:
    import yfinance  # noqa: F401
except ImportError:
    _install_yfinance_stub()
