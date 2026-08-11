import pytest

from capsule.search.repositories import (
    PostgresAssetSearchRepository,
    _text_relevance,
    _text_search_terms,
)


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


def test_local_text_terms_support_chinese_without_external_tokenizer() -> None:
    terms = _text_search_terms("想找小孩追着风筝跑的画面，最好是在空旷草地")

    assert "小孩" in terms
    assert "风筝" in terms
    assert "草地" in terms


def test_local_text_relevance_covers_filename_path_raw_text_and_description() -> None:
    terms = _text_search_terms("蓝紫色黄昏")
    scores = {
        "filename": _text_relevance(
            query_text="蓝紫色黄昏",
            terms=terms,
            file_name="蓝紫色黄昏.png",
            relative_path=None,
            raw_content=None,
            asset_description=None,
        ),
        "path": _text_relevance(
            query_text="蓝紫色黄昏",
            terms=terms,
            file_name=None,
            relative_path="概念图/蓝紫色黄昏/reference.png",
            raw_content=None,
            asset_description=None,
        ),
        "raw": _text_relevance(
            query_text="蓝紫色黄昏",
            terms=terms,
            file_name=None,
            relative_path=None,
            raw_content="场景发生在蓝紫色黄昏，远处城市灯光亮起。",
            asset_description=None,
        ),
        "description": _text_relevance(
            query_text="蓝紫色黄昏",
            terms=terms,
            file_name=None,
            relative_path=None,
            raw_content=None,
            asset_description="一幅蓝紫色黄昏下的动画城市景观。",
        ),
    }

    assert all(score > 0 for score in scores.values())
    assert scores["filename"] > scores["path"] > scores["raw"] > scores["description"]
