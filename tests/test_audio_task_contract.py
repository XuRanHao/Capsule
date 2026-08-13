import pytest

from capsule.pipeline.media_task_dispatch import MediaTaskProcessor
from capsule.pipeline.processing_task_registry import validate_message_contract
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskLease,
    ProcessingTaskMessage,
    ProcessingTaskResult,
    ResourceClass,
)


def test_audio_uses_existing_mps_media_route() -> None:
    message = ProcessingTaskMessage(
        job_id="job-audio",
        workspace_id="workspace-audio",
        source_file_id="source-audio",
        generation=1,
        source_uri="file:///imports/sample.wav",
        task_kind=ProcessingTaskKind.AUDIO,
        resource_class=ResourceClass.MPS_VIDEO,
        route_key="mps_video",
    )

    registration = validate_message_contract(message, expected_route_key="mps_video")

    assert registration.task_kind is ProcessingTaskKind.AUDIO
    assert registration.resource_class is ResourceClass.MPS_VIDEO


@pytest.mark.asyncio
async def test_shared_media_processor_dispatches_audio_by_task_kind() -> None:
    called: list[ProcessingTaskKind] = []

    class Processor:
        async def process(self, message, lease, report_progress):  # type: ignore[no-untyped-def]
            del lease, report_progress
            called.append(message.task_kind)
            return ProcessingTaskResult(result_ref="audio-task://done")

    message = ProcessingTaskMessage(
        job_id="job-audio",
        workspace_id="workspace-audio",
        source_file_id="source-audio",
        generation=1,
        task_kind=ProcessingTaskKind.AUDIO,
    )
    lease = ProcessingTaskLease(
        task_id=message.task_id or "",
        source_file_id=message.source_file_id,
        source_generation=message.generation,
        attempt=1,
        worker_id="worker",
        result_version=1,
        task_kind=ProcessingTaskKind.AUDIO,
        resource_class=ResourceClass.MPS_VIDEO,
        route_key="mps_video",
        processor_version=1,
    )
    processor = MediaTaskProcessor({ProcessingTaskKind.AUDIO: Processor()})

    result = await processor.process(message, lease, lambda _progress: None)  # type: ignore[arg-type]

    assert result.result_ref == "audio-task://done"
    assert called == [ProcessingTaskKind.AUDIO]
