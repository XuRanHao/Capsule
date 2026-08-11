"""Shared contracts for materialized and timeline-only video Segment Assets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from capsule.enums import AssetType, ProcessingStatus
from capsule.schemas import AssetCreate

VideoOutputMode = Literal["materialized", "logical"]
LOGICAL_VIDEO_OUTPUT_MODE = "logical"
MATERIALIZED_VIDEO_OUTPUT_MODE = "materialized"


def is_logical_video_asset(
    *,
    asset_type: AssetType | str,
    file_info: Mapping[str, Any],
) -> bool:
    """Identify the explicit no-derived-media marker from durable metadata."""
    return (
        str(asset_type) == AssetType.VIDEO_SEGMENT.value
        and file_info.get("video_output_mode") == LOGICAL_VIDEO_OUTPUT_MODE
    )


def logical_video_asset(asset: AssetCreate) -> AssetCreate:
    """Remove persistent media payloads while keeping the Asset enrichable."""
    return asset.model_copy(
        update={
            "file_info": {
                **asset.file_info,
                "video_output_mode": LOGICAL_VIDEO_OUTPUT_MODE,
            },
            "derived_file_uri": None,
            "preview_uri": None,
            "transient_keyframe_jpegs": [],
            "processing_status": ProcessingStatus.PENDING,
        }
    )
