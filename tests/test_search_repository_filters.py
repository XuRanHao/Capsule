import pytest

from capsule.search.repositories import PostgresAssetSearchRepository


class _Rows:
    def all(self) -> list[object]:
        return []


class _Transaction:
    async def __aenter__(self) -> "_Transaction":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _Session:
    def __init__(self) -> None:
        self.statements: list[object] = []

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def begin(self) -> _Transaction:
        return _Transaction()

    async def execute(self, statement: object) -> _Rows:
        self.statements.append(statement)
        return _Rows()


class _Database:
    def __init__(self, session: _Session) -> None:
        self._session = session

    def session(self) -> _Session:
        return self._session


@pytest.mark.asyncio
async def test_hydration_excludes_parent_assets() -> None:
    session = _Session()
    repository = PostgresAssetSearchRepository(_Database(session))  # type: ignore[arg-type]

    records = await repository.get_by_ids(
        workspace_id="workspace_demo",
        asset_ids=["parent_asset", "child_asset"],
        embedding_ids=(),
    )

    assert records == {}
    assert len(session.statements) == 1
    compiled = session.statements[0].compile()  # type: ignore[union-attr]
    assert "index_role" in str(compiled)
    assert "parent" in compiled.params.values()
