"""Background activation and pause controls, with no external services."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.services import pipeline, scheduler


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'agent.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    now = datetime.now(timezone.utc)
    loop = scheduler.AgentLoop()
    # Isolate the scheduled scan from unrelated periodic work.
    loop._last_inbox[None] = now
    loop._last_link_check[None] = now
    loop._last_fit[None] = now
    loop._last_idle_feed[None] = now
    scan = Mock(return_value=SimpleNamespace(status="failed", stats={}))
    send = Mock()
    monkeypatch.setattr(pipeline, "scan", scan)
    monkeypatch.setattr(pipeline, "send_batch", send)
    monkeypatch.setattr(pipeline, "llm_providers", lambda *_: [])
    monkeypatch.setattr(scheduler, "settings", SimpleNamespace(llm_enabled=False))
    with sessions() as db:
        row = pipeline.get_settings_row(db)
        row.chatgpt_browser_enabled = False
        db.commit()
        yield db, sessions, loop, now, scan, send
    engine.dispose()


def test_deactivate_and_reactivate_background_work(agent_env):
    db, sessions, loop, now, scan, send = agent_env
    row = pipeline.get_settings_row(db)
    row.enabled = False
    db.commit()

    loop._tick_user(db, None, now)
    scan.assert_not_called()
    send.assert_not_called()

    # Settings persist independently of the scheduler's session.
    with sessions() as other:
        saved = pipeline.get_settings_row(other)
        saved.enabled = True
        other.commit()
    db.expire_all()
    loop._tick_user(db, None, now)
    scan.assert_called_once()


def test_deactivating_during_scan_prevents_followup_send(agent_env):
    db, sessions, loop, now, scan, send = agent_env
    row = pipeline.get_settings_row(db)
    row.enabled = True
    db.commit()

    def pause_during_scan(*args, **kwargs):
        # Simulate the dashboard changing settings while a scan is in flight.
        with sessions() as other:
            saved = pipeline.get_settings_row(other)
            saved.enabled = False
            other.commit()
        return SimpleNamespace(status="completed", stats={"relevant": 1})

    scan.side_effect = pause_during_scan
    loop._tick_user(db, None, now)
    scan.assert_called_once()
    send.assert_not_called()
    assert row.enabled is False
