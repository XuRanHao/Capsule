from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from capsule.parsers import discover_files


class PipelineNotReadyError(RuntimeError):
    pass


class PipelinePlan(BaseModel):
    workspace_id: str
    input_path: str
    file_count: int
    total_bytes: int
    counts_by_extension: dict[str, int] = Field(default_factory=dict)


class PipelineRunner:
    """Top-level orchestration boundary for the staged implementation."""

    def build_plan(self, input_path: Path, workspace_id: str) -> PipelinePlan:
        files = discover_files(input_path)
        counts = Counter(item.extension for item in files)
        return PipelinePlan(
            workspace_id=workspace_id,
            input_path=str(input_path.expanduser().resolve()),
            file_count=len(files),
            total_bytes=sum(item.size_bytes for item in files),
            counts_by_extension=dict(sorted(counts.items())),
        )

    async def run(self, input_path: Path, workspace_id: str) -> None:
        plan = self.build_plan(input_path, workspace_id)
        raise PipelineNotReadyError(
            "pipeline adapters are not wired yet; "
            f"validated {plan.file_count} supported files for {workspace_id}"
        )
