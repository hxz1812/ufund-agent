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

The bridge supports these outputs, in priority order:

1. A JSON line containing an `answer` field: `{"answer":"..."}`
2. Text following an `ANSWER:` marker
3. Any other non-empty standard output as plain text
4. A friendly placeholder when the script exits successfully without output

If the script exits with an error, the chat displays the final error line. Each
message is a separate script run; previous chat messages are not passed along.
