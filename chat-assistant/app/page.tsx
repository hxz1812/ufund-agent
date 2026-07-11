"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
};

type ConnectionState = "checking" | "connected" | "missing";

const WELCOME_MESSAGE: Message = {
  id: 0,
  role: "assistant",
  content:
    "Hello — I’m your UFund assistant. Ask me a question and I’ll send it to your local research script.",
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [draft, setDraft] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const nextId = useRef(1);
  const endOfMessages = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => setConnection(data?.scriptFound ? "connected" : "missing"))
      .catch(() => setConnection("missing"));
  }, []);

  useEffect(() => {
    endOfMessages.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  function resetChat() {
    setMessages([WELCOME_MESSAGE]);
    setDraft("");
    setIsThinking(false);
    requestAnimationFrame(() => textarea.current?.focus());
  }

  function resizeTextarea() {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();
    const question = draft.trim();
    if (!question || isThinking) return;

    const userMessage: Message = {
      id: nextId.current++,
      role: "user",
      content: question,
    };

    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setIsThinking(true);
    if (textarea.current) textarea.current.style.height = "auto";

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || "The script could not complete this request.");
      }

      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content:
            error instanceof Error
              ? error.message
              : "I couldn’t reach the local script. Please try again.",
          isError: true,
        },
      ]);
    } finally {
      setIsThinking(false);
      requestAnimationFrame(() => textarea.current?.focus());
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function copyMessage(message: Message) {
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId(null), 1600);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Chat controls">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">U</div>
          <div>
            <div className="brand-name">UFund</div>
            <div className="brand-caption">Knowledge Assistant</div>
          </div>
        </div>

        <button className="new-chat-button" type="button" onClick={resetChat}>
          <span aria-hidden="true">＋</span>
          New conversation
        </button>

        <div className="sidebar-note">
          <div className={`connection-dot ${connection}`} aria-hidden="true" />
          <div>
            <span className="connection-title">
              {connection === "checking"
                ? "Checking script"
                : connection === "connected"
                  ? "Script connected"
                  : "Script not found"}
            </span>
            <span className="connection-detail">
              {connection === "missing"
                ? "Check the configured path"
                : "Each question runs independently"}
            </span>
          </div>
        </div>

        <div className="privacy-note">
          <span className="privacy-icon" aria-hidden="true">◆</span>
          Questions stay on this computer and are sent only to your local script.
        </div>
      </aside>

      <section className="chat-panel" aria-label="UFund chat assistant">
        <header className="mobile-header">
          <div className="brand-mark small" aria-hidden="true">U</div>
          <div>
            <div className="brand-name">UFund</div>
            <div className="mobile-status">
              <span className={`connection-dot ${connection}`} aria-hidden="true" />
              {connection === "connected" ? "Connected" : "Local assistant"}
            </div>
          </div>
          <button className="mobile-new-chat" type="button" onClick={resetChat} aria-label="New conversation">
            ＋
          </button>
        </header>

        <div className="conversation" aria-live="polite">
          <div className="conversation-inner">
            <div className="date-divider"><span>Today</span></div>

            {messages.map((message) => (
              <article
                className={`message-row ${message.role}`}
                key={message.id}
                data-testid={`${message.role}-message`}
              >
                {message.role === "assistant" && (
                  <div className="assistant-avatar" aria-hidden="true">U</div>
                )}
                <div className={`message-content ${message.isError ? "error" : ""}`}>
                  <div className="message-label">
                    {message.role === "assistant" ? "UFund Assistant" : "You"}
                  </div>
                  <div className="message-bubble">
                    {message.content.split("\n").map((line, index) => (
                      <span key={index}>
                        {line}
                        {index < message.content.split("\n").length - 1 && <br />}
                      </span>
                    ))}
                  </div>
                  {message.role === "assistant" && message.id !== 0 && (
                    <button className="copy-button" type="button" onClick={() => copyMessage(message)}>
                      {copiedId === message.id ? "Copied" : "Copy response"}
                    </button>
                  )}
                </div>
              </article>
            ))}

            {isThinking && (
              <article className="message-row assistant" data-testid="thinking-indicator">
                <div className="assistant-avatar" aria-hidden="true">U</div>
                <div className="message-content">
                  <div className="message-label">UFund Assistant</div>
                  <div className="thinking-bubble" aria-label="Assistant is working">
                    <span /><span /><span />
                    <em>Running your local script</em>
                  </div>
                </div>
              </article>
            )}
            <div ref={endOfMessages} />
          </div>
        </div>

        <footer className="composer-area">
          <form className="composer" onSubmit={sendMessage}>
            <label className="sr-only" htmlFor="chat-input">Ask UFund a question</label>
            <textarea
              id="chat-input"
              ref={textarea}
              value={draft}
              rows={1}
              placeholder="Ask about your UFund documents…"
              onChange={(event) => {
                setDraft(event.target.value);
                resizeTextarea();
              }}
              onKeyDown={handleKeyDown}
              disabled={isThinking}
              autoFocus
            />
            <button
              className="send-button"
              type="submit"
              aria-label="Send message"
              disabled={!draft.trim() || isThinking}
            >
              <span aria-hidden="true">↑</span>
            </button>
          </form>
          <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
        </footer>
      </section>
    </main>
  );
}
