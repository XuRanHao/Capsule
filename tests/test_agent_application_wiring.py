from capsule.agent.runtime import AgentRuntime
from capsule.api.app import _configure_default_agent_runtime
from capsule.config import Settings
from capsule.db.agent_memory import AgentConversationRepository
from capsule.db.repositories import RelationGraphRepository
from capsule.db.session import Database


def test_default_application_runtime_registers_all_graph_tools() -> None:
    database = Database(Settings())
    runtime = AgentRuntime()

    _configure_default_agent_runtime(
        runtime=runtime,
        database=database,
        conversation_repository=AgentConversationRepository(database),
        graph_repository=RelationGraphRepository(database),
        settings=Settings(),
        memory_event_publisher=None,
    )

    assert {item["name"] for item in runtime._tools.catalog()} == {
        "load_current_graph_context",
        "get_entity_detail",
        "list_entity_relations",
        "list_recent_tool_operations",
        "create_entity",
        "merge_entities",
        "split_entity",
        "delete_entity",
        "move_asset_to_entity",
        "create_parent_relation",
        "remove_relation",
    }
