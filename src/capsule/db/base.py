from collections.abc import Callable

from sqlalchemy.orm import DeclarativeBase
from ulid import ULID


class Base(DeclarativeBase):
    pass


def id_factory(prefix: str) -> Callable[[], str]:
    def create_id() -> str:
        return f"{prefix}_{ULID()}"

    return create_id
