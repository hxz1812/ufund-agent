"use client";

import {
  FormEvent,
  KeyboardEvent as ReactKeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";

type SourceDocument = {
  evidenceId: number;
  fileName: string | null;
  sourceSystem: string | null;
  relativePath: string | null;
  fileType: string | null;
  sheetName: string | null;
  rowStart: number | null;
  rowEnd: number | null;
};

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
  documentsUsed?: SourceDocument[];
  warning?: string | null;
  stepCount?: number;
  traceFile?: string | null;
};

type ConnectionState =
  | "checking"
  | "connected"
  | "missing"
  | "index-missing"
  | "reindexing";

type RunSettings = {
  llmProvider: "local" | "openai";
  openaiModel: string;
  maxTokens: number;
  temperature: number;
  nCtx: number;
  maxSteps: number;
  trace: boolean;
};

type ReindexState = "idle" | "running" | "success" | "error";

const DEFAULT_SETTINGS: RunSettings = {
  llmProvider: "local",
  openaiModel: "gpt-3.5-turbo",
  maxTokens: 512,
  temperature: 0,
  nCtx: 2048,
  maxSteps: 5,
  trace: false,
};

const WELCOME_MESSAGE: Message = {
  id: 0,
  role: "assistant",
  content:
    "Hello — I’m your UFund assistant. Ask me a question and I’ll search your indexed company documents.",
};

function boundedSetting(value: unknown, fallback: number, minimum: number, maximum: number) {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed)
    ? Math.min(maximum, Math.max(minimum, parsed))
    : fallback;
}

function normalizeSettings(value: unknown): RunSettings {
  const saved = value && typeof value === "object" ? value as Partial<RunSettings> : {};
  return {
    llmProvider: saved.llmProvider === "openai" ? "openai" : "local",
    openaiModel:
      typeof saved.openaiModel === "string" && saved.openaiModel.trim()
        ? saved.openaiModel.slice(0, 100)
        : DEFAULT_SETTINGS.openaiModel,
    maxTokens: Math.round(boundedSetting(saved.maxTokens, 512, 64, 4096)),
    temperature: boundedSetting(saved.temperature, 0, 0, 2),
    nCtx: Math.round(boundedSetting(saved.nCtx, 2048, 512, 32768)),
    maxSteps: Math.round(boundedSetting(saved.maxSteps, 5, 1, 10)),
    trace: saved.trace === true,
  };
}

async function readConnectionState(): Promise<ConnectionState> {
  try {
    const response = await fetch("/api/health");
    const data = response.ok ? await response.json() : null;
    if (data?.reindexRunning) return "reindexing";
    if (!data?.scriptFound) return "missing";
    if (data?.indexFound === false) return "index-missing";
    return "connected";
  } catch {
    return "missing";
  }
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [draft, setDraft] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsReady, setSettingsReady] = useState(false);
  const [runSettings, setRunSettings] = useState<RunSettings>(DEFAULT_SETTINGS);
  const [exportDrive, setExportDrive] = useState(false);
  const [reindexState, setReindexState] = useState<ReindexState>("idle");
  const [reindexMessage, setReindexMessage] = useState("");
  const nextId = useRef(1);
  const endOfMessages = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void readConnectionState().then(setConnection);
  }, []);

  useEffect(() => {
    try {
      const savedSettings = window.localStorage.getItem("ufund-run-settings");
      if (savedSettings) setRunSettings(normalizeSettings(JSON.parse(savedSettings)));
    } catch {
      setRunSettings(DEFAULT_SETTINGS);
    } finally {
      setSettingsReady(true);
    }
  }, []);

  useEffect(() => {
    if (!settingsReady) return;
    window.localStorage.setItem("ufund-run-settings", JSON.stringify(runSettings));
  }, [runSettings, settingsReady]);

  useEffect(() => {
    endOfMessages.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  useEffect(() => {
    if (!settingsOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSettingsOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [settingsOpen]);

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
    if (!question || isThinking || reindexState === "running") return;

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
        body: JSON.stringify({ message: question, options: runSettings }),
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
          documentsUsed: Array.isArray(data.documentsUsed) ? data.documentsUsed : [],
          warning: typeof data.warning === "string" ? data.warning : null,
          stepCount: typeof data.stepCount === "number" ? data.stepCount : 0,
          traceFile: typeof data.traceFile === "string" ? data.traceFile : null,
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

  async function runReindex() {
    if (reindexState === "running") return;
    setReindexState("running");
    setReindexMessage(
      exportDrive
        ? "Refreshing Drive files and rebuilding the index…"
        : "Rebuilding the index from current exported files…",
    );
    setConnection("reindexing");

    try {
      const response = await fetch("/api/reindex", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exportDrive }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error || "Reindexing did not complete.");

      setReindexState("success");
      setReindexMessage(data.message || "The document index is ready.");
      setConnection("connected");
    } catch (error) {
      setReindexState("error");
      setReindexMessage(
        error instanceof Error ? error.message : "The document index could not be rebuilt.",
      );
      setConnection(await readConnectionState());
    }
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
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

  const providerLabel =
    runSettings.llmProvider === "local" ? "Local model" : runSettings.openaiModel;

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

        <button className="settings-button" type="button" onClick={() => setSettingsOpen(true)}>
          <span aria-hidden="true">⚙</span>
          <span>Run settings</span>
          <em>{runSettings.llmProvider === "local" ? "Local" : "OpenAI"}</em>
        </button>

        <div className="sidebar-note">
          <div className={`connection-dot ${connection}`} aria-hidden="true" />
          <div>
            <span className="connection-title">
              {connection === "checking"
                ? "Checking script"
                : connection === "reindexing"
                  ? "Reindexing documents"
                  : connection === "connected"
                    ? "Script connected"
                    : connection === "index-missing"
                      ? "Index not ready"
                      : "Script not found"}
            </span>
            <span className="connection-detail">
              {connection === "missing"
                ? "Check the configured path"
                : connection === "reindexing"
                  ? "Questions resume when complete"
                  : connection === "index-missing"
                    ? "Run app.py --reindex first"
                    : "Index ready · independent questions"}
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
              {connection === "connected"
                ? "Index ready"
                : connection === "reindexing"
                  ? "Reindexing"
                  : connection === "index-missing"
                    ? "Index required"
                    : "Local assistant"}
            </div>
          </div>
          <button className="mobile-new-chat" type="button" onClick={resetChat} aria-label="New conversation">
            ＋
          </button>
          <button className="mobile-settings" type="button" onClick={() => setSettingsOpen(true)} aria-label="Run settings">
            ⚙
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
                    <>
                      <div className="response-actions">
                        <button className="copy-button" type="button" onClick={() => copyMessage(message)}>
                          {copiedId === message.id ? "Copied" : "Copy response"}
                        </button>
                        {!!message.stepCount && (
                          <span className="step-count">
                            {message.stepCount} agent {message.stepCount === 1 ? "step" : "steps"}
                          </span>
                        )}
                        {message.traceFile && (
                          <span className="trace-status" title={message.traceFile}>Trace saved</span>
                        )}
                      </div>

                      {!!message.documentsUsed?.length && (
                        <details className="sources-panel">
                          <summary>
                            <span aria-hidden="true">▤</span>
                            {message.documentsUsed.length} {message.documentsUsed.length === 1 ? "source" : "sources"} used
                          </summary>
                          <div className="source-list">
                            {message.documentsUsed.map((document, index) => {
                              const rowLabel = document.rowStart
                                ? document.rowEnd && document.rowEnd !== document.rowStart
                                  ? `Rows ${document.rowStart}–${document.rowEnd}`
                                  : `Row ${document.rowStart}`
                                : null;
                              return (
                                <div className="source-card" key={`${document.evidenceId}-${index}`}>
                                  <span className="source-id">[{document.evidenceId}]</span>
                                  <div className="source-copy">
                                    <strong>{document.fileName || document.relativePath || "Company document"}</strong>
                                    <span>
                                      {[document.sourceSystem, document.sheetName, rowLabel]
                                        .filter(Boolean)
                                        .join(" · ") || "Indexed document"}
                                    </span>
                                    {document.relativePath && document.relativePath !== document.fileName && (
                                      <code>{document.relativePath}</code>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </details>
                      )}

                      {message.warning && (
                        <div className="response-warning">
                          <strong>Agent note:</strong> {message.warning}
                        </div>
                      )}
                    </>
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
                    <em>Searching your indexed documents</em>
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
              placeholder={reindexState === "running" ? "Reindexing documents…" : "Ask about your UFund documents…"}
              onChange={(event) => {
                setDraft(event.target.value);
                resizeTextarea();
              }}
              onKeyDown={handleKeyDown}
              disabled={isThinking || reindexState === "running"}
              autoFocus
            />
            <button
              className="send-button"
              type="submit"
              aria-label="Send message"
              disabled={!draft.trim() || isThinking || reindexState === "running"}
            >
              <span aria-hidden="true">↑</span>
            </button>
          </form>
          <p className="composer-hint">
            {providerLabel}{runSettings.trace ? " · Trace on" : ""}
            {" · Enter to send · Shift + Enter for a new line"}
          </p>
        </footer>
      </section>

      {settingsOpen && (
        <>
          <button
            className="settings-overlay"
            type="button"
            aria-label="Close run settings"
            onClick={() => setSettingsOpen(false)}
          />
          <aside className="settings-drawer" role="dialog" aria-modal="true" aria-labelledby="settings-title">
            <header className="settings-header">
              <div>
                <span className="settings-kicker">app.py options</span>
                <h2 id="settings-title">Run settings</h2>
              </div>
              <button type="button" onClick={() => setSettingsOpen(false)} aria-label="Close run settings">×</button>
            </header>

            <div className="settings-body">
              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <h3>Answer model</h3>
                    <p>Applied to each new, independent question.</p>
                  </div>
                  <button type="button" className="reset-settings" onClick={() => setRunSettings({ ...DEFAULT_SETTINGS })}>
                    Reset
                  </button>
                </div>

                <label className="setting-field">
                  <span>LLM provider</span>
                  <select
                    value={runSettings.llmProvider}
                    onChange={(event) => setRunSettings((current) => ({
                      ...current,
                      llmProvider: event.target.value as RunSettings["llmProvider"],
                    }))}
                  >
                    <option value="local">Local Llama model</option>
                    <option value="openai">OpenAI</option>
                  </select>
                </label>

                {runSettings.llmProvider === "openai" && (
                  <label className="setting-field">
                    <span>OpenAI model</span>
                    <input
                      type="text"
                      value={runSettings.openaiModel}
                      placeholder="gpt-3.5-turbo"
                      maxLength={100}
                      onChange={(event) => setRunSettings((current) => ({
                        ...current,
                        openaiModel: event.target.value,
                      }))}
                    />
                  </label>
                )}

                <label className="setting-switch">
                  <div>
                    <strong>Save trace</strong>
                    <span>Writes agent actions and tool outputs to local_traces.</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={runSettings.trace}
                    onChange={(event) => setRunSettings((current) => ({
                      ...current,
                      trace: event.target.checked,
                    }))}
                  />
                </label>

                <details className="advanced-settings">
                  <summary>Advanced generation settings</summary>
                  <div className="advanced-grid">
                    <label className="setting-field compact">
                      <span>Max tokens</span>
                      <input
                        type="number"
                        min={64}
                        max={4096}
                        step={64}
                        value={runSettings.maxTokens}
                        onChange={(event) => setRunSettings((current) => ({
                          ...current,
                          maxTokens: Number(event.target.value),
                        }))}
                      />
                    </label>
                    <label className="setting-field compact">
                      <span>Max steps</span>
                      <input
                        type="number"
                        min={1}
                        max={10}
                        value={runSettings.maxSteps}
                        onChange={(event) => setRunSettings((current) => ({
                          ...current,
                          maxSteps: Number(event.target.value),
                        }))}
                      />
                    </label>
                    <label className="setting-field compact">
                      <span>Context size</span>
                      <input
                        type="number"
                        min={512}
                        max={32768}
                        step={512}
                        value={runSettings.nCtx}
                        onChange={(event) => setRunSettings((current) => ({
                          ...current,
                          nCtx: Number(event.target.value),
                        }))}
                      />
                    </label>
                    <label className="setting-field compact temperature-field">
                      <span>Temperature <output>{runSettings.temperature.toFixed(1)}</output></span>
                      <input
                        type="range"
                        min={0}
                        max={2}
                        step={0.1}
                        value={runSettings.temperature}
                        onChange={(event) => setRunSettings((current) => ({
                          ...current,
                          temperature: Number(event.target.value),
                        }))}
                      />
                    </label>
                  </div>
                </details>
              </section>

              <section className="settings-section reindex-section">
                <div className="settings-section-heading">
                  <div>
                    <h3>Document index</h3>
                    <p>Rebuild the persistent Chroma index used by the agent.</p>
                  </div>
                  <span className={`index-pill ${connection === "connected" ? "ready" : ""}`}>
                    {reindexState === "running" ? "Working" : connection === "connected" ? "Ready" : "Needs attention"}
                  </span>
                </div>

                <label className="setting-switch drive-switch">
                  <div>
                    <strong>Refresh Google Drive first</strong>
                    <span>Adds --export-drive to the reindex run.</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={exportDrive}
                    disabled={reindexState === "running"}
                    onChange={(event) => setExportDrive(event.target.checked)}
                  />
                </label>

                <button
                  className="reindex-button"
                  type="button"
                  disabled={reindexState === "running" || isThinking}
                  onClick={runReindex}
                >
                  {reindexState === "running" ? "Reindexing…" : "Rebuild document index"}
                </button>

                {reindexMessage && (
                  <p className={`reindex-status ${reindexState}`} role="status">{reindexMessage}</p>
                )}
              </section>
            </div>
          </aside>
        </>
      )}
    </main>
  );
}
