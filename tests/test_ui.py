"""Headless Streamlit UI tests (streamlit.testing AppTest) with a fake-LLM backend."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import turn_script
from sql_analyst.ui import client as client_mod

APP = str(Path(__file__).resolve().parents[1] / "src" / "sql_analyst" / "ui" / "app.py")


@pytest.fixture
def app_with(fake_service, monkeypatch):
    def _make(script):
        st.cache_resource.clear()
        local = client_mod.LocalClient(fake_service(script))
        monkeypatch.setattr(client_mod, "get_client", lambda: local)
        at = AppTest.from_file(APP, default_timeout=30)
        return at.run()

    yield _make
    st.cache_resource.clear()


def all_text(at) -> str:
    parts = (
        [m.value for m in at.markdown] + [i.value for i in at.info] + [w.value for w in at.warning]
    )
    return "\n".join(str(p) for p in parts)


def test_page_loads_with_examples_and_schema(app_with):
    at = app_with({})
    assert not at.exception
    assert at.title[0].value == "Ask your data"
    assert any("countries" in b.label for b in at.button)
    assert "Artist" in all_text(at)


def test_question_shows_answer_sql_and_data(app_with):
    at = app_with(turn_script())
    at.chat_input[0].set_value("How many artists?").run()
    assert not at.exception
    text = all_text(at)
    assert "How many artists?" in text
    assert "There are 3 artists." in text
    assert any("SELECT COUNT(*)" in c.value for c in at.code)
    assert len(at.dataframe) == 1


def test_approval_flow_in_ui(app_with):
    at = app_with(turn_script(sql="SELECT * FROM Artist, Album", answer="Here are the rows."))
    at.chat_input[0].set_value("show everything").run()
    assert "needs your approval" in all_text(at)
    assert at.chat_input[0].disabled  # can't ask a new question while one awaits approval

    at.button(key="approve").click().run()
    assert not at.exception
    assert "Here are the rows." in all_text(at)
    assert not at.chat_input[0].disabled


def test_reject_in_ui(app_with):
    at = app_with(turn_script(sql="SELECT * FROM Artist, Album"))
    at.chat_input[0].set_value("show everything").run()
    at.button(key="reject").click().run()
    assert "didn't run that query" in all_text(at)


def test_new_conversation_clears_messages(app_with):
    at = app_with(turn_script())
    at.chat_input[0].set_value("How many artists?").run()
    old_thread = at.session_state.thread_id
    next(b for b in at.button if b.label == "New conversation").click().run()
    assert at.session_state.messages == []
    assert at.session_state.thread_id != old_thread


def test_session_question_limit(app_with, monkeypatch):
    monkeypatch.setenv("UI_MAX_QUESTIONS_PER_SESSION", "1")
    from sql_analyst.config import get_settings

    get_settings.cache_clear()
    at = app_with(turn_script())
    at.chat_input[0].set_value("How many artists?").run()
    assert "question limit" in all_text(at)
    assert at.chat_input[0].disabled
    # "New conversation" must not reset the cap
    next(b for b in at.button if b.label == "New conversation").click().run()
    assert at.chat_input[0].disabled
