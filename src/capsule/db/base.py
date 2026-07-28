from collections.abc import Callable
from uuid import uuid4

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def id_factory(prefix: str) -> Callable[[], str]:
    def create_id() -> str:
        return f"{prefix}_{uuid4().hex}"

    return create_id
