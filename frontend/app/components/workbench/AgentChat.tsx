"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { CREATED_BY, apiFetch, endpoint } from "../../lib/api";

type Message = { id: string; role: "user" | "assistant"; content: string };
type Props = { workspaceId: string; selectedGraphId: string | null };

type AgentThread = { thread_id: string; title: string };
type AgentResponse = { message: string | null; status: string; pending_action: unknown | null };
type StreamEvent = { event: string; data: Record<string, unknown> };

const STARTER: Message[] = [{
  id: "welcome",
  role: "assistant",
  content: "你好，我可以基于当前工作空间中的素材帮你查找线索、梳理关系，或继续追问图谱中的节点。",
}];

function parseSseFrames(chunk: string): { frames: StreamEvent[]; remaining: string } {
  const frames: StreamEvent[] = [];
  const parts = chunk.split("\n\n");
  const remaining = parts.pop() ?? "";
  for (const part of parts) {
    const event = part.match(/^event:\s*(.+)$/m)?.[1]?.trim();
    const data = part.match(/^data:\s*(.+)$/m)?.[1];
    if (!event || !data) continue;
    try {
      const parsed = JSON.parse(data) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        frames.push({ event, data: parsed as Record<string, unknown> });
      }
    } catch {
      // An incomplete or invalid server frame must not break a completed turn.
    }
  }
  return { frames, remaining };
}

export default function AgentChat({ workspaceId, selectedGraphId }: Props) {
  const [messages, setMessages] = useState<Message[]>(STARTER);
  const [input, setInput] = useState("");
  const [thread, setThread] = useState<AgentThread | null>(null);
  const [sending, setSending] = useState(false);
  const [streamStatus, setStreamStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [messages, sending]);
  async function ensureThread() {
    if (thread) return thread;
    const next = await apiFetch<AgentThread>("/api/v1/agent/threads", {
      method: "POST",
      body: JSON.stringify({ user_id: CREATED_BY, workspace_id: workspaceId, title: "工作台对话" }),
    });
    setThread(next);
    return next;
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = input.trim();
    if (!content || sending) return;
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content }]);
    setInput(""); setSending(true); setStreamStatus("正在规划回复…"); setError(null);
    try {
      const currentThread = await ensureThread();
      const response = await fetch(endpoint("/api/v1/agent/invoke/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          thread_id: currentThread.thread_id,
          user_id: CREATED_BY,
          workspace_id: workspaceId,
          graph_id: selectedGraphId ?? undefined,
          message: content,
        }),
      });
      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(payload?.detail || `请求失败（${response.status}）`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalResponse: AgentResponse | null = null;
      let streamError: string | null = null;
      const consume = (value: string) => {
        const parsed = parseSseFrames(buffer + value);
        buffer = parsed.remaining;
        for (const frame of parsed.frames) {
          if (frame.event === "phase" && typeof frame.data.message === "string") {
            setStreamStatus(frame.data.message);
          } else if (frame.event === "result") {
            finalResponse = frame.data as unknown as AgentResponse;
          } else if (frame.event === "error") {
            streamError = typeof frame.data.message === "string"
              ? frame.data.message
              : "对话请求失败，请稍后重试。";
          }
        }
      };
      while (true) {
        const { done, value } = await reader.read();
        if (value) consume(decoder.decode(value, { stream: !done }));
        if (done) break;
      }
      consume(decoder.decode());
      if (streamError) throw new Error(streamError);
      if (!finalResponse) throw new Error("对话服务未返回完成结果。");
      setMessages((current) => [...current, {
        id: crypto.randomUUID(), role: "assistant",
        content: finalResponse.message || (finalResponse.status === "awaiting_confirmation" ? "该操作需要你的确认后才能继续。" : "本次请求没有返回可展示的内容。"),
      }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "对话请求失败，请检查后端服务。");
    } finally { setSending(false); setStreamStatus(null); }
  }

  return (
    <aside className="workbench-panel agent-chat" aria-label="助手对话">
      <header className="panel-heading agent-heading">
        <div><span className="panel-kicker">CONVERSATION</span><h2>工作空间助手</h2></div>
        <span className="agent-status"><i />在线</span>
      </header>
      {selectedGraphId && <div className="agent-context">当前图谱 · {selectedGraphId.slice(0, 12)}</div>}
      <div className="chat-messages" aria-live="polite">
        {messages.map((message) => <article className={`chat-message ${message.role}`} key={message.id}><span>{message.role === "user" ? "你" : "AI"}</span><p>{message.content}</p></article>)}
        {sending && <article className="chat-message assistant pending"><span>AI</span><p>{streamStatus || "正在分析当前工作空间…"}</p></article>}
        <div ref={endRef} />
      </div>
      {error && <p className="chat-error">{error}</p>}
      <div className="chat-suggestions" aria-label="快捷提问">
        {["概览当前工作空间", "这批素材有哪些关联？", "找出最近导入的重点"].map((suggestion) => (
          <button type="button" key={suggestion} onClick={() => setInput(suggestion)}>{suggestion}</button>
        ))}
      </div>
      <form className="chat-composer" onSubmit={submit}>
        <label className="sr-only" htmlFor="agent-message">输入问题</label>
        <textarea id="agent-message" value={input} onChange={(event) => setInput(event.target.value)} placeholder="问问工作空间中的素材…" rows={2} disabled={sending} />
        <button type="submit" aria-label="发送消息" disabled={!input.trim() || sending}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 4 16 8-16 8 3-8z" fill="currentColor" /></svg>
        </button>
      </form>
    </aside>
  );
}
