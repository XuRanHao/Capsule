"""HTTP entry point for the model-independent conversational Agent runtime."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status

from capsule.agent.contracts import (
    AgentMessageResponse,
    AgentRequest,
    AgentResponse,
    AgentThreadCreateRequest,
    AgentThreadRenameRequest,
    AgentThreadResponse,
)
from capsule.agent.runtime import AgentRuntime
from capsule.db.agent_memory import (
    AgentConversationRepository,
    AgentMessageRecord,
    AgentThreadRecord,
    AgentThreadStateError,
)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/invoke", response_model=AgentResponse)
async def invoke_agent(payload: AgentRequest, request: Request) -> AgentResponse:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if not isinstance(runtime, AgentRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "agent_runtime_not_ready", "message": "Agent runtime is not ready"},
        )
    try:
        return await runtime.invoke(payload)
    except AgentThreadStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/threads", response_model=AgentThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_thread(
    payload: AgentThreadCreateRequest,
    request: Request,
) -> AgentThreadResponse:
    repository = _conversation_repository(request)
    try:
        thread = await repository.create_thread(
            user_id=payload.user_id,
            workspace_id=payload.workspace_id,
            title=payload.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _thread_response(thread)


@router.get("/threads", response_model=list[AgentThreadResponse])
async def list_agent_threads(
    request: Request,
    user_id: str = Query(min_length=1, max_length=128),
    workspace_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=50, ge=1, le=200),
    thread_status: Literal["active", "archived", "deleted"] | None = Query(
        default=None,
        alias="status",
    ),
    query: str | None = Query(default=None, max_length=255),
    include_deleted: bool = Query(default=False),
) -> list[AgentThreadResponse]:
    threads = await _conversation_repository(request).list_threads(
        user_id=user_id,
        workspace_id=workspace_id,
        limit=limit,
        status=thread_status,
        query=query,
        include_deleted=include_deleted,
    )
    return [_thread_response(thread) for thread in threads]


@router.patch("/threads/{thread_id}", response_model=AgentThreadResponse)
async def rename_agent_thread(
    thread_id: str,
    payload: AgentThreadRenameRequest,
    request: Request,
) -> AgentThreadResponse:
    try:
        thread = await _conversation_repository(request).rename_thread(
            thread_id=thread_id,
            user_id=payload.user_id,
            workspace_id=payload.workspace_id,
            title=payload.title,
        )
    except AgentThreadStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _thread_response(thread)


@router.post("/threads/{thread_id}/archive", response_model=AgentThreadResponse)
async def archive_agent_thread(
    thread_id: str,
    request: Request,
    user_id: str = Query(min_length=1, max_length=128),
    workspace_id: str = Query(min_length=1, max_length=128),
) -> AgentThreadResponse:
    try:
        thread = await _conversation_repository(request).archive_thread(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        await _agent_runtime(request).discard_state(thread_id)
    except AgentThreadStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _thread_response(thread)


@router.post("/threads/{thread_id}/restore", response_model=AgentThreadResponse)
async def restore_agent_thread(
    thread_id: str,
    request: Request,
    user_id: str = Query(min_length=1, max_length=128),
    workspace_id: str = Query(min_length=1, max_length=128),
) -> AgentThreadResponse:
    try:
        thread = await _conversation_repository(request).restore_thread(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _thread_response(thread)


@router.delete("/threads/{thread_id}", response_model=AgentThreadResponse)
async def soft_delete_agent_thread(
    thread_id: str,
    request: Request,
    user_id: str = Query(min_length=1, max_length=128),
    workspace_id: str = Query(min_length=1, max_length=128),
) -> AgentThreadResponse:
    try:
        thread = await _conversation_repository(request).soft_delete_thread(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        await _agent_runtime(request).discard_state(thread_id)
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _thread_response(thread)


@router.get("/threads/{thread_id}/messages", response_model=list[AgentMessageResponse])
async def list_agent_messages(
    thread_id: str,
    request: Request,
    user_id: str = Query(min_length=1, max_length=128),
    workspace_id: str = Query(min_length=1, max_length=128),
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
) -> list[AgentMessageResponse]:
    repository = _conversation_repository(request)
    try:
        messages = await repository.list_messages(
            thread_id=thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
            after_sequence=after_sequence,
            limit=limit,
        )
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return [_message_response(message) for message in messages]


def _conversation_repository(request: Request) -> AgentConversationRepository:
    repository = getattr(request.app.state, "agent_conversation_repository", None)
    if not isinstance(repository, AgentConversationRepository):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "agent_conversations_not_ready",
                "message": "Agent conversations are not ready",
            },
        )
    return repository


def _agent_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if not isinstance(runtime, AgentRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "agent_runtime_not_ready",
                "message": "Agent runtime is not ready",
            },
        )
    return runtime


def _thread_response(thread: AgentThreadRecord) -> AgentThreadResponse:
    return AgentThreadResponse(
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        workspace_id=thread.workspace_id,
        title=thread.title,
        status=thread.status,
        summary=thread.summary,
        summary_topic=thread.summary_topic,
        memory_revision=thread.memory_revision,
        last_message_at=(thread.last_message_at.isoformat() if thread.last_message_at else None),
        deleted_at=(thread.deleted_at.isoformat() if thread.deleted_at else None),
    )


def _message_response(message: AgentMessageRecord) -> AgentMessageResponse:
    return AgentMessageResponse(
        message_id=message.message_id,
        sequence=message.sequence,
        role=message.role,
        content=message.content,
        name=message.name,
        turn_id=message.turn_id,
        request_id=message.request_id,
        created_at=message.created_at.isoformat(),
    )
