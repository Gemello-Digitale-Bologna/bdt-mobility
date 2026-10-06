"""Lightweight i18n for the simulation dashboard."""

from __future__ import annotations

import streamlit as st

from . import en, it

LANGUAGES = {"it": it, "en": en}
DEFAULT_LANG = "it"


def get_lang() -> str:
    """Return current language code from session state."""
    return st.session_state.get("__lang__", DEFAULT_LANG)


def set_lang(lang: str) -> None:
    """Set current language and trigger rerun."""
    st.session_state["__lang__"] = lang
    st.rerun()


def t(key: str) -> str:
    """Translate a dot-separated key to the current language.

    Falls back to the key itself if the translation is missing.
    """
    lang_module = LANGUAGES[get_lang()]
    return getattr(lang_module, key, key)
