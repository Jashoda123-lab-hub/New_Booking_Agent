"""Shared fixtures: a fresh SQLite file per test, a pinned clock and a fake LLM.

The fake LLM stands in for the model's *understanding* step with fixed,
realistic outputs, so the tests exercise the graph, tools and database
deterministically and without an API key.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import Extraction, ViewingAgent  # noqa: E402
from app.db import init_db, make_engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.tools import BookingTools  # noqa: E402

# Saturday 21 Nov 2026, 09:00. Seed slots: 22–28 Nov at 10:00, 11:00, 16:00, 17:00.
NOW = datetime(2026, 11, 21, 9, 0)


class FakeLLM:
    """Maps a message to the Extraction a good LLM would return for it."""

    def __init__(self):
        self.responses = {}
        self.contexts = []  # what the agent told the LLM about the conversation, per call

    def when(self, message: str, **fields) -> str:
        self.responses[message] = Extraction(**fields)
        return message

    def extract(self, message, now, context=""):
        self.contexts.append(context)
        if message not in self.responses:
            raise AssertionError(f"FakeLLM has no response for: {message!r}")
        return self.responses[message]

    def draft(self, message, facts, template):
        return template


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(eng, seed_from=NOW)
    yield eng
    eng.dispose()


@pytest.fixture
def tools(engine):
    return BookingTools(sessionmaker(bind=engine))


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def agent(tools, llm):
    return ViewingAgent(tools, llm, clock=lambda: NOW)


@pytest.fixture
def client(agent):
    return TestClient(create_app(agent=agent))
