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

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
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
PORT = int(os.environ.get("CEREBROS_API_PORT", "8000"))

# Global model cache
loaded_models: Dict[str, Dict] = {}


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
    assistant_id: str = None,
    stage: str = "1"
):
    """Process uploaded file: chunk text to 512 chars, save as training CSV"""
    import csv
    import re
    
    if not assistant_id:
        assistant_id = f"assistant_{int(time.time())}"
    
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
    # Use provided assistant_id or generate from name
    if request.assistant_id:
        assistant_id = request.assistant_id
    elif request.assistant_name:
        assistant_id = request.assistant_name.lower().replace(' ', '_')
    else:
        assistant_id = f"assistant_{int(time.time())}"
    
    # In production, this would start the training pipeline
    # For demo, we'll simulate the process
    
    def training_task():
        """Background training task"""
        import subprocess
        
        # Run data processing
        subprocess.run([
            "python3", "scripts/process_user_samples.py",
            "--assistant_id", assistant_id
        ])
        
        # Run training
        subprocess.run([
            "python3", "multi_stage_trainer.py",
            assistant_id,
            request.assistant_name,
            str(NFS_PATH)
        ])
    
    # Add to background tasks
    background_tasks.add_task(training_task)
    
    return {
        "assistant_id": assistant_id,
        "name": request.assistant_name,
        "status": "training_started",
        "message": "Training pipeline started in background"
    }


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