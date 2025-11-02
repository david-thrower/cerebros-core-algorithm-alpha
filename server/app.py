#!/usr/bin/env python3
"""
CEREBROS NotGPT - FastAPI Inference Server
Serves trained assistant models via REST API

Usage:
python3 server/app.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
NFS_PATH = Path(os.environ.get("CEREBROS_NFS_PATH", "priv/nfs"))
HOST = os.environ.get("CEREBROS_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("CEREBROS_API_PORT", "8080"))

# Global model cache
loaded_models: Dict[str, Dict] = {}

# Training job tracking
active_training_jobs: Dict[str, Dict] = {}  # assistant_id -> {process, status, logs}
training_websockets: Dict[str, List[WebSocket]] = {}  # assistant_id -> [websocket connections]


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class QueryResponse(BaseModel):
    response: str
    assistant_id: str
    timestamp: str
    metadata: Optional[Dict] = None


class AssistantStatus(BaseModel):
    assistant_id: str
    name: str
    status: str
    model_path: Optional[str] = None
    created_at: Optional[str] = None
    metrics: Optional[Dict] = None


class TrainingRequest(BaseModel):
    assistant_name: Optional[str] = None
    assistant_id: Optional[str] = None
    data_sources: Optional[List[str]] = None


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 CEREBROS NotGPT API Server Starting...")
    print(f"📁 NFS Path: {NFS_PATH}")
    print(f"🌐 Listening on {HOST}:{PORT}")
    yield
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="CEREBROS NotGPT API",
    description="Personalized AI Assistant Training & Inference API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def load_assistant_model(assistant_id: str) -> Dict:
    """Load assistant model and metadata"""
    assistant_path = NFS_PATH / assistant_id
    
    # Check if assistant exists
    if not assistant_path.exists():
        # Try agents subdirectory
        assistant_path = NFS_PATH / "agents" / assistant_id
        if not assistant_path.exists():
            raise HTTPException(status_code=404, detail=f"Assistant '{assistant_id}' not found")
    
    # Load model metadata
    metadata_path = assistant_path / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "agent_id": assistant_id,
            "status": "ready",
            "created_at": datetime.now().isoformat()
        }
    
    # Load checkpoint (in production, this would load the actual model)
    checkpoint_path = assistant_path / "checkpoints" / "stage_5_checkpoint.keras"
    if not checkpoint_path.exists():
        checkpoint_path = None
    
    model_data = {
        "assistant_id": assistant_id,
        "metadata": metadata,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "loaded_at": datetime.now().isoformat()
    }
    
    # Cache the model
    loaded_models[assistant_id] = model_data
    
    return model_data


def generate_response(assistant_id: str, query: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """Generate response from assistant model"""
    # In production, this would use the actual trained model
    # For demo, we'll generate a contextual response
    
    model_data = loaded_models.get(assistant_id)
    if not model_data:
        model_data = load_assistant_model(assistant_id)
    
    # Simulated response generation
    responses = [
        f"Based on my training, here's what I understand about your query: {query[:50]}...",
        f"Let me help you with that. {query[:30]}... is an interesting topic.",
        f"I've analyzed your question about {query[:40]}... and here's my perspective:",
        f"From what I've learned, the answer to '{query[:50]}...' involves several key points.",
    ]
    
    import random
    base_response = random.choice(responses)
    
    # Add some context based on the query
    if "how" in query.lower():
        base_response += " The process involves understanding the context, analyzing the requirements, and providing a structured approach."
    elif "what" in query.lower():
        base_response += " This concept relates to our previous discussions and the data I've been trained on."
    elif "why" in query.lower():
        base_response += " The reasoning behind this is based on patterns I've identified in my training data."
    else:
        base_response += " I'm here to help you understand this better based on my personalized training."
    
    return base_response


async def stream_response(assistant_id: str, query: str, temperature: float, max_tokens: int):
    """Stream response tokens"""
    response = generate_response(assistant_id, query, temperature, max_tokens)
    
    # Simulate streaming by yielding chunks
    words = response.split()
    for i, word in enumerate(words):
        chunk = {
            "token": word + " " if i < len(words) - 1 else word,
            "done": i == len(words) - 1
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)  # Simulate processing time


# API Routes
@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "service": "CEREBROS NotGPT API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "POST /assistants/{assistant_id}/query",
            "status": "GET /assistants/{assistant_id}/status",
            "list": "GET /assistants",
            "train": "POST /assistants/train"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/assistants")
async def list_assistants():
    """List all available assistants"""
    assistants = []
    
    # Check both root and agents subdirectory
    for check_path in [NFS_PATH, NFS_PATH / "agents"]:
        if not check_path.exists():
            continue
        
        for assistant_path in check_path.iterdir():
            if not assistant_path.is_dir():
                continue
            
            assistant_id = assistant_path.name
            
            # Skip system directories
            if assistant_id in ["uploads", "datasets", "database"]:
                continue
            
            # Load metadata if available
            metadata_path = assistant_path / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "agent_id": assistant_id,
                    "agent_name": assistant_id.title(),
                    "status": "unknown"
                }
            
            assistants.append({
                "assistant_id": metadata.get("agent_id", assistant_id),
                "name": metadata.get("agent_name", assistant_id.title()),
                "status": metadata.get("status", "unknown"),
                "created_at": metadata.get("created_at"),
                "deployment_ready": metadata.get("deployment_ready", False)
            })
    
    return {"assistants": assistants, "count": len(assistants)}


@app.get("/assistants/{assistant_id}/status")
async def get_assistant_status(assistant_id: str):
    """Get assistant status and metadata"""
    try:
        model_data = load_assistant_model(assistant_id)
        metadata = model_data["metadata"]
        
        return AssistantStatus(
            assistant_id=assistant_id,
            name=metadata.get("agent_name", assistant_id.title()),
            status=metadata.get("status", "unknown"),
            model_path=model_data.get("checkpoint_path"),
            created_at=metadata.get("created_at"),
            metrics=metadata.get("metrics")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assistants/{assistant_id}/query")
async def query_assistant(assistant_id: str, request: QueryRequest):
    """Query an assistant and get a response"""
    try:
        # Load model if not cached
        if assistant_id not in loaded_models:
            load_assistant_model(assistant_id)
        
        # Stream response if requested
        if request.stream:
            return StreamingResponse(
                stream_response(assistant_id, request.query, request.temperature, request.max_tokens),
                media_type="text/event-stream"
            )
        
        # Generate response
        response_text = generate_response(
            assistant_id, 
            request.query, 
            request.temperature, 
            request.max_tokens
        )
        
        return QueryResponse(
            response=response_text,
            assistant_id=assistant_id,
            timestamp=datetime.now().isoformat(),
            metadata={
                "query_length": len(request.query),
                "response_length": len(response_text),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    assistant_id: Optional[str] = None
):
    """Upload training data file (CSV or JSON)"""
    if not assistant_id:
        assistant_id = f"assistant_{int(time.time())}"
    
    # Validate file type
    allowed_extensions = ['.csv', '.json']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")
    
    # Create upload directory
    upload_dir = NFS_PATH / "agents" / assistant_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_path = upload_dir / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "status": "success",
            "assistant_id": assistant_id,
            "filename": file.filename,
            "path": str(file_path),
            "size_bytes": len(content),
            "message": f"File uploaded successfully to {assistant_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/process-stage")
async def process_stage(
    file: UploadFile = File(...),
    assistant_id: str = Form(None),
    stage: str = Form("1")
):
    """Process uploaded file: chunk text to 512 chars, save as training CSV"""
    import csv
    import re
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"📦 Received upload: assistant_id='{assistant_id}', stage={stage}, file={file.filename}")
    
    # Generate assistant_id if not provided or empty
    if not assistant_id or assistant_id.strip() == "":
        assistant_id = f"assistant_{int(time.time())}"
        logger.info(f"🆔 Generated new ID: {assistant_id}")
    else:
        assistant_id = assistant_id.strip()
        logger.info(f"✅ Using provided ID: {assistant_id}")
    
    # Create directories
    datasets_dir = NFS_PATH / "agents" / assistant_id / "datasets"
    uploads_dir = NFS_PATH / "agents" / assistant_id / "uploads"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original file
    upload_path = uploads_dir / f"stage{stage}_{file.filename}"
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)
    
    # Extract text content
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        # Try latin-1 as fallback
        text_content = content.decode('latin-1', errors='ignore')
    
    # Clean text: remove extra whitespace
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    
    # Chunk text into 512-character segments
    chunk_size = 512
    chunks = []
    for i in range(0, len(text_content), chunk_size):
        chunk = text_content[i:i + chunk_size]
        if len(chunk.strip()) > 50:  # Skip very small chunks
            chunks.append(chunk.strip())
    
    # Create training CSV
    csv_path = datasets_dir / f"training_stage{stage}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prompt', 'reasoning', 'response'])
        
        for idx, chunk in enumerate(chunks):
            # Create synthetic training examples from chunks
            prompt = f"Context from uploaded document (stage {stage}, chunk {idx+1})"
            reasoning = f"This content represents the user's style and knowledge from their uploaded materials"
            response = chunk
            writer.writerow([prompt, reasoning, response])
    
    return {
        "status": "success",
        "assistant_id": assistant_id,
        "stage": int(stage),
        "csv_file": str(csv_path),
        "chunks_created": len(chunks),
        "original_file": str(upload_path)
    }


@app.post("/assistants/train")
async def train_assistant(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training a new assistant"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Received training request: assistant_id={request.assistant_id}, assistant_name={request.assistant_name}")
    
    # Use provided assistant_id or generate from name
    if request.assistant_id:
        assistant_id = request.assistant_id
    elif request.assistant_name:
        assistant_id = request.assistant_name.lower().replace(' ', '_')
    else:
        assistant_id = f"assistant_{int(time.time())}"
    
    logger.info(f"Using assistant_id: {assistant_id}")
    
    # In production, this would start the training pipeline
    # For demo, we'll simulate the process
    
    async def training_task():
        """Background training task with live streaming"""
        import subprocess
        import logging
        import shutil
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training for assistant: {assistant_id}")
        
        # Check if CSV files exist from wizard uploads
        datasets_dir = NFS_PATH / "agents" / assistant_id / "datasets"
        csv_files = sorted(datasets_dir.glob("training_stage*.csv")) if datasets_dir.exists() else []
        
        if not csv_files:
            error_msg = f"No training data found. Please upload files in the wizard before starting training."
            logger.error(f"No CSV files found in {datasets_dir}, cannot train")
            active_training_jobs[assistant_id] = {
                "status": "failed",
                "error": error_msg,
                "logs": [f"❌ Error: {error_msg}", f"   Expected location: {datasets_dir}", "   Make sure to upload files in steps 1-4 of the wizard before training."]
            }
            # Notify WebSocket clients of failure
            if assistant_id in training_websockets:
                for ws in training_websockets[assistant_id]:
                    try:
                        await ws.send_json({"type": "complete", "status": "failed", "error": error_msg})
                    except:
                        pass
            return
            
        logger.info(f"Found {len(csv_files)} CSV files for training: {[f.name for f in csv_files]}")
        
        # Create processed directory for multi_stage_trainer
        processed_dir = NFS_PATH / "agents" / assistant_id / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy CSV files to processed directory with expected names
        stage_mapping = {
            "training_stage1.csv": "stage1_base.csv",
            "training_stage2.csv": "stage2_domain.csv", 
            "training_stage3.csv": "stage3_knowledge.csv",
            "training_stage4.csv": "stage4_style.csv"
        }
        
        for csv_file in csv_files:
            if csv_file.name in stage_mapping:
                dest_name = stage_mapping[csv_file.name]
                dest_path = processed_dir / dest_name
                shutil.copy2(csv_file, dest_path)
                logger.info(f"Copied {csv_file.name} -> {dest_name}")
        
        # Run training with live output streaming
        try:
            logger.info(f"Launching multi_stage_trainer.py for {assistant_id}")
            
            # Prepare environment with CPU-only mode to avoid CUDA conflicts
            training_env = os.environ.copy()
            training_env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
            training_env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
            training_env["LD_LIBRARY_PATH"] = ""  # Clear library path to prevent CUDA loading
            training_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
            training_env["TF_CPP_MIN_VLOG_LEVEL"] = "0"
            
            # Use Popen for streaming output with virtual environment Python
            venv_python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python3")
            python_executable = venv_python if Path(venv_python).exists() else sys.executable
            
            process = subprocess.Popen(
                [python_executable, "-u", "multi_stage_trainer.py",
                 assistant_id,
                 request.assistant_name or assistant_id,
                 str(NFS_PATH)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=training_env,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(Path(__file__).parent.parent)
            )
            
            # Track the job
            active_training_jobs[assistant_id] = {
                "status": "training",
                "process": process,
                "logs": [],
                "started_at": datetime.now().isoformat()
            }
            
            # Stream output line by line
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"[{assistant_id}] {line}")
                    active_training_jobs[assistant_id]["logs"].append(line)
                    
                    # Broadcast to connected WebSockets
                    if assistant_id in training_websockets:
                        disconnected = []
                        for ws in training_websockets[assistant_id]:
                            try:
                                await ws.send_json({"type": "log", "data": line})
                            except:
                                disconnected.append(ws)
                        # Clean up disconnected clients
                        for ws in disconnected:
                            training_websockets[assistant_id].remove(ws)
            
            # Wait for process to complete
            return_code = process.wait(timeout=3600)
            
            logger.info(f"Training completed with exit code: {return_code}")
                
            # Update assistant status
            if return_code == 0:
                logger.info(f"Training succeeded for {assistant_id}")
                active_training_jobs[assistant_id]["status"] = "completed"
                
                # Create metadata file
                metadata = {
                    "assistant_id": assistant_id,
                    "name": request.assistant_name or assistant_id,
                    "status": "ready",
                    "trained_at": datetime.now().isoformat(),
                    "training_files": len(csv_files)
                }
                metadata_path = NFS_PATH / "agents" / assistant_id / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                # Notify WebSocket clients of completion
                if assistant_id in training_websockets:
                    for ws in training_websockets[assistant_id]:
                        try:
                            await ws.send_json({"type": "complete", "status": "success"})
                        except:
                            pass
            else:
                logger.error(f"Training failed with exit code {return_code}")
                active_training_jobs[assistant_id]["status"] = "failed"
                active_training_jobs[assistant_id]["error"] = f"Exit code {return_code}"
                
                # Notify WebSocket clients of failure
                if assistant_id in training_websockets:
                    for ws in training_websockets[assistant_id]:
                        try:
                            await ws.send_json({"type": "complete", "status": "failed", "error": f"Exit code {return_code}"})
                        except:
                            pass
                
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out after 1 hour")
            active_training_jobs[assistant_id]["status"] = "failed"
            active_training_jobs[assistant_id]["error"] = "Timeout after 1 hour"
        except Exception as e:
            logger.error(f"Training failed with exception: {str(e)}", exc_info=True)
            active_training_jobs[assistant_id]["status"] = "failed"
            active_training_jobs[assistant_id]["error"] = str(e)
    
    # Add to background tasks
    background_tasks.add_task(training_task)
    
    return {
        "assistant_id": assistant_id,
        "name": request.assistant_name,
        "status": "training_started",
        "message": "Training pipeline started in background"
    }


@app.get("/training/status/{assistant_id}")
async def get_training_status(assistant_id: str):
    """Get training job status and logs"""
    if assistant_id not in active_training_jobs:
        raise HTTPException(status_code=404, detail=f"No training job found for assistant '{assistant_id}'")
    
    job = active_training_jobs[assistant_id]
    return {
        "assistant_id": assistant_id,
        "status": job["status"],
        "started_at": job.get("started_at"),
        "log_lines": len(job.get("logs", [])),
        "error": job.get("error")
    }


@app.get("/training/logs/{assistant_id}")
async def get_training_logs(assistant_id: str, offset: int = 0):
    """Get training logs for an assistant"""
    if assistant_id not in active_training_jobs:
        raise HTTPException(status_code=404, detail=f"No training job found for assistant '{assistant_id}'")
    
    logs = active_training_jobs[assistant_id].get("logs", [])
    return {
        "assistant_id": assistant_id,
        "logs": logs[offset:],
        "total_lines": len(logs),
        "status": active_training_jobs[assistant_id]["status"]
    }


@app.websocket("/ws/training/{assistant_id}")
async def training_websocket(websocket: WebSocket, assistant_id: str):
    """WebSocket endpoint for live training logs"""
    await websocket.accept()
    
    # Register this WebSocket
    if assistant_id not in training_websockets:
        training_websockets[assistant_id] = []
    training_websockets[assistant_id].append(websocket)
    
    try:
        # Send existing logs if job is active
        if assistant_id in active_training_jobs:
            job = active_training_jobs[assistant_id]
            await websocket.send_json({
                "type": "init",
                "status": job["status"],
                "logs": job.get("logs", [])
            })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back for ping/pong
                await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup on disconnect
        if assistant_id in training_websockets:
            if websocket in training_websockets[assistant_id]:
                training_websockets[assistant_id].remove(websocket)
            if not training_websockets[assistant_id]:
                del training_websockets[assistant_id]


@app.delete("/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """Delete an assistant and its data"""
    assistant_path = NFS_PATH / "agents" / assistant_id
    
    if not assistant_path.exists():
        raise HTTPException(status_code=404, detail=f"Assistant '{assistant_id}' not found")
    
    # In production, this would properly clean up
    import shutil
    shutil.rmtree(assistant_path)
    
    # Remove from cache
    if assistant_id in loaded_models:
        del loaded_models[assistant_id]
    
    return {"message": f"Assistant '{assistant_id}' deleted successfully"}


def main():
    """Run the server"""
    print("\n" + "=" * 60)
    print("🚀 CEREBROS NotGPT API Server")
    print("=" * 60)
    print(f"📡 Starting server on http://{HOST}:{PORT}")
    print(f"📖 API docs at http://{HOST}:{PORT}/docs")
    print(f"📁 Data directory: {NFS_PATH}")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()