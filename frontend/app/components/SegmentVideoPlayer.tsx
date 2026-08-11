"use client";

import { useMemo, useRef, useState } from "react";
import type { VideoPlayback } from "../lib/api";

type SegmentVideoPlayerProps = {
  playback: VideoPlayback | null;
  legacyContentUrl: string | null;
  posterUrl: string | null;
  fallbackMimeType: string;
};

function formatTime(milliseconds: number) {
  const safeMilliseconds = Math.max(0, milliseconds);
  const totalSeconds = Math.floor(safeMilliseconds / 1_000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export default function SegmentVideoPlayer({
  playback,
  legacyContentUrl,
  posterUrl,
  fallbackMimeType,
}: SegmentVideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const effectivePlayback = useMemo<VideoPlayback | null>(() => {
    if (playback) return playback;
    if (!legacyContentUrl) return null;
    return {
      mode: "derived_clip",
      url: legacyContentUrl,
      mime_type: fallbackMimeType.includes("/")
        ? fallbackMimeType
        : "video/mp4",
      start_ms: 0,
      end_ms: null,
      duration_ms: null,
      browser_compatible: true,
      fallback_url: null,
    };
  }, [fallbackMimeType, legacyContentUrl, playback]);
  const [usingFallback, setUsingFallback] = useState(false);
  const [currentMs, setCurrentMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!effectivePlayback) return null;

  const activePlayback =
    usingFallback && effectivePlayback.fallback_url
      ? {
          ...effectivePlayback,
          mode: "transcoded_stream" as const,
          url: effectivePlayback.fallback_url,
          mime_type: "video/mp4",
          browser_compatible: true,
        }
      : effectivePlayback;
  const isSourceRange = activePlayback.mode === "source_range";
  const startMs = isSourceRange ? (activePlayback.start_ms ?? 0) : 0;
  const configuredDuration = activePlayback.duration_ms;
  const endMs = isSourceRange
    ? effectivePlayback.end_ms
    : configuredDuration;
  const segmentDurationMs =
    configuredDuration ??
    (endMs === null ? 0 : Math.max(0, endMs - startMs));

  if (!activePlayback.browser_compatible) {
    return (
      <div className="asset-video-unavailable" role="status">
        <strong>当前浏览器无法直接播放这个源视频</strong>
        <span>服务器尚未提供兼容的临时播放流。</span>
      </div>
    );
  }

  if (!isSourceRange) {
    return (
      <div className="segment-video-player">
        <video
          key={activePlayback.url}
          className="asset-video-player"
          controls
          preload="metadata"
          poster={posterUrl ?? undefined}
          onLoadedMetadata={() => setError(null)}
          onError={() => setError("视频加载失败，请检查源文件是否仍然可用。")}
        >
          <source
            src={activePlayback.url}
            type={activePlayback.mime_type || undefined}
          />
          当前浏览器不支持播放此视频片段。
        </video>
        {error && <p className="segment-video-error">{error}</p>}
      </div>
    );
  }

  const clampToSegment = (absoluteMilliseconds: number) => {
    const maximum = endMs ?? absoluteMilliseconds;
    return Math.min(Math.max(absoluteMilliseconds, startMs), maximum);
  };

  const seekRelative = (relativeMilliseconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    const absoluteMilliseconds = clampToSegment(startMs + relativeMilliseconds);
    video.currentTime = absoluteMilliseconds / 1_000;
    setCurrentMs(Math.max(0, absoluteMilliseconds - startMs));
  };

  const togglePlayback = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (!video.paused) {
      video.pause();
      return;
    }
    if (segmentDurationMs > 0 && currentMs >= segmentDurationMs - 100) {
      seekRelative(0);
    }
    try {
      await video.play();
      setError(null);
    } catch {
      setError("浏览器阻止了播放，请再次点击播放。");
    }
  };

  return (
    <div className="segment-video-player">
      <video
        key={activePlayback.url}
        ref={videoRef}
        className="asset-video-player"
        preload="metadata"
        poster={posterUrl ?? undefined}
        onLoadedMetadata={(event) => {
          const video = event.currentTarget;
          const sourceDurationMs = Number.isFinite(video.duration)
            ? video.duration * 1_000
            : startMs;
          video.currentTime = clampToSegment(
            Math.min(startMs, sourceDurationMs),
          ) / 1_000;
          setCurrentMs(0);
          setReady(true);
          setError(null);
        }}
        onTimeUpdate={(event) => {
          const video = event.currentTarget;
          const absoluteMilliseconds = video.currentTime * 1_000;
          if (endMs !== null && absoluteMilliseconds >= endMs - 80) {
            video.pause();
            video.currentTime = endMs / 1_000;
            setCurrentMs(segmentDurationMs);
            setPlaying(false);
            return;
          }
          if (absoluteMilliseconds < startMs - 80) {
            video.currentTime = startMs / 1_000;
            setCurrentMs(0);
            return;
          }
          setCurrentMs(Math.max(0, absoluteMilliseconds - startMs));
        }}
        onSeeking={(event) => {
          const video = event.currentTarget;
          const requestedMs = video.currentTime * 1_000;
          const clampedMs = clampToSegment(requestedMs);
          if (Math.abs(requestedMs - clampedMs) > 80) {
            video.currentTime = clampedMs / 1_000;
          }
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false);
          setCurrentMs(segmentDurationMs);
        }}
        onError={() => {
          if (effectivePlayback.fallback_url && !usingFallback) {
            setPlaying(false);
            setReady(false);
            setCurrentMs(0);
            setError("源视频编码不受支持，正在切换兼容播放流。");
            setUsingFallback(true);
            return;
          }
          setError("视频加载失败，请检查源文件是否仍然可用。");
        }}
      >
        <source
          src={activePlayback.url}
          type={activePlayback.mime_type || undefined}
        />
        当前浏览器不支持播放此视频片段。
      </video>
      <div className="segment-video-controls">
        <button type="button" disabled={!ready} onClick={() => void togglePlayback()}>
          {playing ? "暂停" : currentMs >= segmentDurationMs - 100 ? "重播" : "播放"}
        </button>
        <input
          aria-label="片段播放位置"
          type="range"
          min="0"
          max={Math.max(1, segmentDurationMs)}
          step="50"
          value={Math.min(currentMs, Math.max(1, segmentDurationMs))}
          disabled={!ready || segmentDurationMs <= 0}
          onChange={(event) => seekRelative(Number(event.currentTarget.value))}
        />
        <time>
          {formatTime(currentMs)} / {formatTime(segmentDurationMs)}
        </time>
      </div>
      {error && <p className="segment-video-error">{error}</p>}
    </div>
  );
}
