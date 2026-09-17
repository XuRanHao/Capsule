"""Connection helpers for the durable LangGraph checkpoint store."""

import asyncio
import sys

from sqlalchemy.engine import make_url

from capsule.config import Settings


def _configure_windows_event_loop_policy() -> None:
    """psycopg async connections require Selector rather than Windows Proactor."""

    if sys.platform != "win32":
        return
    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    proactor_policy = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    current_policy = asyncio.get_event_loop_policy()
    if (
        selector_policy is not None
        and proactor_policy is not None
        and isinstance(current_policy, proactor_policy)
    ):
        asyncio.set_event_loop_policy(selector_policy())


_configure_windows_event_loop_policy()


def postgres_checkpoint_url(settings: Settings) -> str:
    """Convert the SQLAlchemy async URL into the psycopg URL LangGraph needs."""

    url = make_url(settings.database_url)
    if not url.drivername.startswith("postgresql"):
        raise ValueError("LangGraph checkpoints require a PostgreSQL database URL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)
