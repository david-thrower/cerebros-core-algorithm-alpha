#!/usr/bin/env python3
"""
Cerebros Training API Server
Flask API for managing multi-stage training pipeline
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import threading
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Training state storage
training_sessions = {}
NFS_PATH = Path("priv/nfs/agents")


def run_training_pipeline(agent_id: str, agent_name: str):
    """Run the training pipeline in a subprocess"""
    try:
        result = subprocess.run(
            [
                "python3",
                "cerebros-core-algorithm-alpha/multi_stage_trainer.py",
                agent_id,
                agent_name
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        training_sessions[agent_id]['status'] = 'completed' if result.returncode == 0 else 'failed'
        training_sessions[agent_id]['output'] = result.stdout
        training_sessions[agent_id]['error'] = result.stderr
        training_sessions[agent_id]['completed_at'] = datetime.now().isoformat()
        
        # Parse results from output
        if result.returncode == 0:
            try:
                # Extract JSON from output
                output_lines = result.stdout.split('\n')
                json_start = None
                for i, line in enumerate(output_lines):
                    if 'FINAL RESULTS:' in line:
                        json_start = i + 1
                        break
                
                if json_start:
                    json_text = '\n'.join(output_lines[json_start:])
                    json_text = json_text.split('=' * 80)[0].strip()
                    results = json.loads(json_text)
                    training_sessions[agent_id]['results'] = results
            except Exception as e:
                print(f"Error parsing results: {e}")
        
    except subprocess.TimeoutExpired:
        training_sessions[agent_id]['status'] = 'failed'
        training_sessions[agent_id]['error'] = 'Training timeout'
    except Exception as e:
        training_sessions[agent_id]['status'] = 'failed'
        training_sessions[agent_id]['error'] = str(e)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "cerebros-training-api"})


@app.route('/api/agents', methods=['POST'])
def create_agent():
    """Create a new agent and initialize training"""
    data = request.json
    agent_name = data.get('name', 'Unnamed Assistant')
    agent_id = str(uuid.uuid4())
    
    # Create agent directory
    agent_path = NFS_PATH / agent_id
    agent_path.mkdir(parents=True, exist_ok=True)
    
    # Save agent metadata
    metadata = {
        "id": agent_id,
        "name": agent_name,
        "created_at": datetime.now().isoformat(),
        "status": "initialized"
    }
    
    with open(agent_path / "agent_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return jsonify({
        "agent_id": agent_id,
        "agent_name": agent_name,
        "status": "initialized"
    }), 201


@app.route('/api/agents/<agent_id>/documents', methods=['POST'])
def upload_documents(agent_id):
    """Upload training documents"""
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    document_type = request.form.get('type', 'work_product')
    
    agent_path = NFS_PATH / agent_id
    docs_path = agent_path / document_type
    docs_path.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        if file.filename:
            filepath = docs_path / file.filename
            file.save(filepath)
            uploaded_files.append(file.filename)
    
    return jsonify({
        "agent_id": agent_id,
        "uploaded": uploaded_files,
        "type": document_type
    }), 200


@app.route('/api/agents/<agent_id>/train', methods=['POST'])
def start_training(agent_id):
    """Start the multi-stage training pipeline"""
    data = request.json
    agent_name = data.get('agent_name', 'Assistant')
    
    # Check if already training
    if agent_id in training_sessions and training_sessions[agent_id]['status'] == 'training':
        return jsonify({"error": "Training already in progress"}), 409
    
    # Initialize training session
    training_sessions[agent_id] = {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "current_stage": 1,
        "total_stages": 5
    }
    
    # Start training in background thread
    thread = threading.Thread(
        target=run_training_pipeline,
        args=(agent_id, agent_name)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "agent_id": agent_id,
        "status": "training",
        "message": "Training pipeline started"
    }), 202


@app.route('/api/agents/<agent_id>/status', methods=['GET'])
def get_training_status(agent_id):
    """Get training status"""
    if agent_id not in training_sessions:
        return jsonify({"error": "Training session not found"}), 404
    
    return jsonify(training_sessions[agent_id]), 200


@app.route('/api/agents/<agent_id>/results', methods=['GET'])
def get_training_results(agent_id):
    """Get training results"""
    if agent_id not in training_sessions:
        return jsonify({"error": "Training session not found"}), 404
    
    session = training_sessions[agent_id]
    
    if session['status'] != 'completed':
        return jsonify({"error": "Training not completed"}), 400
    
    return jsonify({
        "agent_id": agent_id,
        "status": session['status'],
        "results": session.get('results', {}),
        "started_at": session['started_at'],
        "completed_at": session.get('completed_at')
    }), 200


@app.route('/api/agents', methods=['GET'])
def list_agents():
    """List all agents"""
    agents = []
    
    if NFS_PATH.exists():
        for agent_dir in NFS_PATH.iterdir():
            if agent_dir.is_dir():
                metadata_file = agent_dir / "agent_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        agents.append(json.load(f))
    
    return jsonify({"agents": agents}), 200


@app.route('/api/agents/<agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get agent details"""
    agent_path = NFS_PATH / agent_id
    metadata_file = agent_path / "agent_metadata.json"
    
    if not metadata_file.exists():
        return jsonify({"error": "Agent not found"}), 404
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check for model metadata
    model_file = agent_path / "model_metadata.json"
    if model_file.exists():
        with open(model_file, 'r') as f:
            metadata['model'] = json.load(f)
    
    return jsonify(metadata), 200


@app.route('/api/agents/<agent_id>/deploy', methods=['POST'])
def deploy_agent(agent_id):
    """Deploy agent (placeholder for actual deployment)"""
    agent_path = NFS_PATH / agent_id
    model_file = agent_path / "model_metadata.json"
    
    if not model_file.exists():
        return jsonify({"error": "Model not found"}), 404
    
    # In production, this would:
    # 1. Deploy REST API endpoint
    # 2. Deploy UI container
    # 3. Create database tables
    
    return jsonify({
        "agent_id": agent_id,
        "status": "deployed",
        "endpoint": f"http://localhost:5000/api/agents/{agent_id}/chat"
    }), 200


@app.route('/api/agents/<agent_id>/chat', methods=['POST'])
def chat(agent_id):
    """Chat with deployed agent (placeholder)"""
    data = request.json
    message = data.get('message', '')
    
    # Placeholder response
    return jsonify({
        "agent_id": agent_id,
        "response": f"Echo: {message}",
        "model": "cerebros-stage-5"
    }), 200


if __name__ == '__main__':
    # Ensure NFS directory exists
    NFS_PATH.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Cerebros Training API Server")
    print("=" * 60)
    print(f"NFS Path: {NFS_PATH}")
    print("Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
