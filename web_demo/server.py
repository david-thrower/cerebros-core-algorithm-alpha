from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os

app = FastAPI(title="Cerebros NotGPT Demo Server")

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine directories for legacy HTML and React app
static_dir = os.path.join(os.path.dirname(__file__), "")
react_build_dir = os.path.join(static_dir, "react_app", "build")

# Mount React build if present, otherwise redirect legacy routes into it
if os.path.exists(react_build_dir):
    app.mount("/static", StaticFiles(directory=react_build_dir, html=True), name="react")
else:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serve React SPA root index always."""
    build_index = os.path.join(react_build_dir, "index.html")
    if os.path.exists(build_index):
        return FileResponse(build_index)
    # If React build missing, fallback legacy
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/new")
async def new_page():
    """Serve upload page."""
    build_index = os.path.join(react_build_dir, "index.html")
    if os.path.exists(build_index):
        return FileResponse(build_index)
    return FileResponse(os.path.join(static_dir, "new.html"))

@app.get("/assistants/{assistant_id}")
async def assistant_chat(assistant_id: str):
    """Serve chat interface page."""
    build_index = os.path.join(react_build_dir, "index.html")
    if os.path.exists(build_index):
        return FileResponse(build_index)
    return FileResponse(os.path.join(static_dir, "assistants.html"))

@app.get("/{full_path:path}")
async def serve_react_routes(full_path: str):
    """Serve React routes for SPA navigation and fallback."""
    build_index = os.path.join(react_build_dir, "index.html")
    if os.path.exists(build_index):
        return FileResponse(build_index)
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/backend-status")
async def backend_status():
    import requests
    try:
        resp = requests.get("http://localhost:8080/health")
        return {"backend": "reachable", "status_code": resp.status_code}
    except Exception as e:
        return {"backend": "unreachable", "error": str(e)}

if __name__ == "__main__":
    # Adjust Uvicorn target to match module path when run from project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    uvicorn.run("web_demo.server:app", host="0.0.0.0", port=3000, log_level="info")