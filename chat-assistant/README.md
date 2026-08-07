# UFund Knowledge Assistant

A private, local chat interface that sends each question independently to the
existing UFund Python script. The original script is never imported or modified;
the local development server launches it as a subprocess with its `--prompt`
argument.

## Configuration

The default script path is:

```text
~/Desktop/ufund-agent/app.py
```

To use a different script or Python environment, copy `.env.example` to `.env`
and update either value:

```bash
UFUND_SCRIPT_PATH=/absolute/path/to/app.py
UFUND_PYTHON_BIN=/absolute/path/to/python3
```

`UFUND_PYTHON_BIN` should point to the Python interpreter where the script's own
dependencies are installed.

## Run locally

Requires Node.js 22 or newer.

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Script response contract

The bridge understands the structured result returned by the current agent:

```json
{
  "answer": "Answer text with [1] citations",
  "documents_used": [{ "evidence_id": 1, "file_name": "example.pdf" }],
  "steps": [],
  "error": null
}
```

Answers appear as normal chat messages. `documents_used` becomes an expandable
source list, `steps` is shown as a compact activity count, and a returned
`error` appears as an agent note. Plain-text output remains supported for
backward compatibility.

The health check also verifies that `chroma_db` exists beside `app.py`. If it
does not, the interface explains that the script needs to be reindexed.

If the script exits with an error, the chat displays the final error line. Each
message is a separate script run; previous chat messages are not passed along.

## Script options in the website

Open **Run settings** in the sidebar to control the CLI options supported by
`app.py`:

- local or OpenAI LLM provider (`--llm-provider`)
- OpenAI model name (`--openai-model`)
- maximum tokens, temperature, context size, and maximum agent steps
- local trace files (`--trace`)

These preferences are saved only in the current browser. The adapter validates
and bounds numeric values before launching the Python process.

The same panel can start `--reindex`. Enable **Refresh Google Drive first** to
also pass `--export-drive`. Chat requests are temporarily paused while the
index is being rebuilt so they do not read a partially updated database.
