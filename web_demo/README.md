# Cerebros Web Demo

This directory contains a **static HTML/JS frontend** demonstrating the Cerebros assistant functionalities through three main pages. It connects directly to the running FastAPI backend at `http://localhost:8080`.

## 🌐 Pages

### 1. Dashboard (`index.html`)
Displays the summary of available assistants and metrics via:
```
GET http://localhost:8080/api/status
```

### 2. Upload Wizard (`new.html`)
Allows users to upload training data (CSV/JSON) to the backend:
```
POST http://localhost:8080/api/upload
```

### 3. Chat Interface (`assistants.html`)
Provides a conversational chat interface with streaming responses from:
```
POST http://localhost:8080/api/assistants/:id/chat
```

## ⚙️ Launching Locally

To open the demo:
1. Ensure the FastAPI backend is running on `http://localhost:8080`.
2. Open any of the HTML files in your browser directly (`index.html`, `new.html`, or `assistants.html`).
3. Navigation between pages uses relative URLs and works offline since all content is static.

## 📁 Contents

```
web_demo/
 ├── index.html        # Dashboard overview
 ├── new.html          # Data upload page
 ├── assistants.html   # Chat interface
 ├── index.css         # Shared styles (copied from UIREFERENCE)
 └── README.md         # This file
```

# Cerebros NotGPT Web Demo (React Integration)

This unified version uses a **React + Vite + TypeScript** frontend under `web_demo/react_app/`, integrated with the FastAPI backend at `http://localhost:8080`.

## 🚀 Running Instructions

### 1. Install dependencies
```bash
cd web_demo/react_app
npm install
```

### 2. Build the production bundle
```bash
npm run build
```

### 3. Serve with FastAPI
```bash
cd ..
python3 server.py
```

This will serve the app at **http://localhost:3000**, while API requests are proxied to **http://localhost:8080**.

## 🔗 Routes
| Path | Description |
|------|-------------|
| `/` | Root dashboard |
| `/new` | PromptTraining UI |
| `/assistants/:id` | Chat/assistant page |
| `/train` | MultiStageWizard training view |

## ⚙️ API Integration
| Endpoint | Purpose |
|-----------|----------|
| `/api/upload` | Upload files for model training |
| `/assistants/:id/query` | Query assistant for chat |
| `/api/status` | System status endpoint |

Once built, `server.py` automatically detects `react_app/build` and serves it as the root.
If absent, it falls back to static HTML (`index.html`, `new.html`, `assistants.html`).
