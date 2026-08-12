"""Copy a small, realistic nested sample from the local Thriller Park project."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_FILES: tuple[tuple[str, str], ...] = (
    (
        "医院篇/病人小男孩参考/正常三视/filename.png",
        "医院篇/角色设定/病人小男孩/正常形态/三视图/filename.png",
    ),
    (
        "医院篇/病人小男孩参考/正常三视/filename (1).png",
        "医院篇/角色设定/病人小男孩/正常形态/三视图/filename (1).png",
    ),
    (
        "医院篇/病人小男孩参考/黑化三视/1.png",
        "医院篇/角色设定/病人小男孩/黑化形态/三视图/1.png",
    ),
    (
        "医院篇/病人小男孩参考/黑化三视/2.png",
        "医院篇/角色设定/病人小男孩/黑化形态/三视图/2.png",
    ),
    (
        "医院篇/病人小男孩参考/黑化三视/3.png",
        "医院篇/角色设定/病人小男孩/黑化形态/三视图/3.png",
    ),
    (
        "医院篇/病房1（fbj醒来）/kling_20260108_图片消除笔__5158_1.png",
        "医院篇/场景设定/病房/病房1_fbj醒来/迭代稿/kling_20260108_图片消除笔__5158_1.png",
    ),
    (
        "医院篇/病房1（fbj醒来）/kling_20260108_图片消除笔__5176_0.png",
        "医院篇/场景设定/病房/病房1_fbj醒来/迭代稿/kling_20260108_图片消除笔__5176_0.png",
    ),
    (
        "医院篇/病房1（fbj醒来）/kling_20260108_图片消除笔__5189_0.png",
        "医院篇/场景设定/病房/病房1_fbj醒来/迭代稿/kling_20260108_图片消除笔__5189_0.png",
    ),
    (
        "医院篇/病房2（双床）/20260112-144948.png",
        "医院篇/场景设定/病房/病房2_双床/确认候选/20260112-144948.png",
    ),
    (
        "医院篇/病房2（双床）/20260112-103912.png",
        "医院篇/场景设定/病房/病房2_双床/确认候选/20260112-103912.png",
    ),
    (
        "医院篇/病房2（双床）/1月12日(2).png",
        "医院篇/场景设定/病房/病房2_双床/确认候选/1月12日(2).png",
    ),
    (
        "医院篇/走廊/4.png",
        "医院篇/场景设定/走廊/门与通道/确认候选/4.png",
    ),
    (
        "医院篇/走廊/5.png",
        "医院篇/场景设定/走廊/门与通道/确认候选/5.png",
    ),
    (
        "医院篇/走廊/7.png",
        "医院篇/场景设定/走廊/门与通道/确认候选/7.png",
    ),
    (
        "医院篇/尸体游乐园/确认/特写/2.png",
        "医院篇/场景设定/尸体游乐园/确认稿/特写/2.png",
    ),
    (
        "医院篇/尸体游乐园/确认/特写/4.png",
        "医院篇/场景设定/尸体游乐园/确认稿/特写/4.png",
    ),
    (
        "医院篇/尸体游乐园/确认/特写/6.png",
        "医院篇/场景设定/尸体游乐园/确认稿/特写/6.png",
    ),
    (
        "第四集/界面/a2.png",
        "第四集/界面/UI方案/确认候选/a2.png",
    ),
    (
        "第四集/界面/a3.png",
        "第四集/界面/UI方案/确认候选/a3.png",
    ),
    (
        "第四集/界面/20260119-122255.png",
        "第四集/界面/UI方案/确认候选/20260119-122255.png",
    ),
    (
        "医院篇/分镜/提示词.txt",
        "医院篇/分镜/提示词/提示词.txt",
    ),
    (
        "医院篇/1111111.txt",
        "医院篇/制作资料/临时文本/1111111.txt",
    ),
    (
        "乱七八糟/2月12日(3).png",
        "临时参考/未采用/杂项/2月12日(3).png",
    ),
    (
        "乱七八糟/局部手指蜡烛.png",
        "临时参考/未采用/杂项/局部手指蜡烛.png",
    ),
    (
        "乱七八糟/极简游戏UI图标集合.png",
        "临时参考/未采用/杂项/极简游戏UI图标集合.png",
    ),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/graph-demo/thriller-park"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    missing = [source for source, _ in _FILES if not (args.source / source).is_file()]
    if missing:
        raise FileNotFoundError(f"fixture sources missing: {missing}")
    copied_bytes = 0
    for source_relative, destination_relative in _FILES:
        source = args.source / source_relative
        destination = args.destination / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_bytes += destination.stat().st_size
    print(
        f"copied {len(_FILES)} files ({copied_bytes / 1024 / 1024:.1f} MiB) to {args.destination}"
    )


if __name__ == "__main__":
    main()
