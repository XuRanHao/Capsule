"""HTTP entry point for the model-independent conversational Agent runtime."""

from fastapi import APIRouter, HTTPException, Request, status

from capsule.agent.contracts import AgentRequest, AgentResponse

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/invoke", response_model=AgentResponse)
async def invoke_agent(payload: AgentRequest, request: Request) -> AgentResponse:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "agent_runtime_not_ready", "message": "Agent runtime is not ready"},
        )
    return await runtime.invoke(payload)
