from sqlalchemy import ForeignKeyConstraint

from capsule.db.base import Base
from capsule.db.models import (
    GraphAsset,
    GraphAssetBinding,
    LogicalEntity,
    LogicalEntityRelation,
    NarrativeGraph,
)


def _composite_foreign_keys(table, local_columns: set[str]) -> list[ForeignKeyConstraint]:
    return [
        constraint
        for constraint in table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
        and {column.name for column in constraint.columns} == local_columns
    ]


def test_narrative_graph_tables_replace_retired_relation_tables() -> None:
    expected = {
        NarrativeGraph.__tablename__,
        GraphAsset.__tablename__,
        LogicalEntity.__tablename__,
        GraphAssetBinding.__tablename__,
        LogicalEntityRelation.__tablename__,
    }
    retired = {
        "relation_entities",
        "relation_entity_sources",
        "entity_entity_relations",
        "asset_entity_relations",
    }

    assert expected <= set(Base.metadata.tables)
    assert retired.isdisjoint(Base.metadata.tables)


def test_graph_asset_binding_cannot_cross_graph_or_workspace() -> None:
    constraints = _composite_foreign_keys(
        GraphAssetBinding.__table__, {"graph_id", "asset_id"}
    )
    constraints += _composite_foreign_keys(
        GraphAssetBinding.__table__, {"graph_id", "entity_id"}
    )

    assert len(constraints) == 2
    targets = {
        tuple(element.target_fullname for element in constraint.elements)
        for constraint in constraints
    }
    assert targets == {
        ("graph_assets.graph_id", "graph_assets.asset_id"),
        ("logical_entities.graph_id", "logical_entities.entity_id"),
    }


def test_entity_relation_edges_are_scoped_to_one_graph() -> None:
    constraints = [
        constraint
        for constraint in LogicalEntityRelation.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]

    assert len(constraints) == 2
    assert all(
        {column.name for column in constraint.columns}
        == {"graph_id", "source_entity_id"}
        or {column.name for column in constraint.columns}
        == {"graph_id", "target_entity_id"}
        for constraint in constraints
    )
