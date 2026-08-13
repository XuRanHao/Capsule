"use client";

import { useEffect, useRef } from "react";
import type { MediaPlayback } from "../lib/api";

export default function SegmentAudioPlayer({
  playback,
}: {
  playback: MediaPlayback;
}) {
  const ref = useRef<HTMLAudioElement>(null);
  const startSeconds = (playback.start_ms || 0) / 1000;
  const endSeconds = playback.end_ms == null ? null : playback.end_ms / 1000;

  useEffect(() => {
    const audio = ref.current;
    if (!audio) return;
    const seek = () => {
      if (Math.abs(audio.currentTime - startSeconds) > 0.15) {
        audio.currentTime = startSeconds;
      }
    };
    const limit = () => {
      if (endSeconds != null && audio.currentTime >= endSeconds) {
        audio.pause();
        audio.currentTime = startSeconds;
      }
    };
    audio.addEventListener("loadedmetadata", seek);
    audio.addEventListener("timeupdate", limit);
    return () => {
      audio.removeEventListener("loadedmetadata", seek);
      audio.removeEventListener("timeupdate", limit);
    };
  }, [endSeconds, startSeconds]);

  return (
    <audio ref={ref} controls preload="metadata">
      <source src={playback.url} type={playback.mime_type} />
      当前浏览器无法播放该音频。
    </audio>
  );
}
