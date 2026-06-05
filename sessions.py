"""
sessions.py — Session Management for PX Explorer
================================================
Handles saving/loading chat histories to disk and JSON import/export.
"""

import os
import json
import uuid
from typing import List, Dict, Any

SESSION_DIR = "sessions"

def ensure_session_dir():
    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)

def save_session(session_id: str, history: List[Dict[str, Any]], model_id: str = None):
    """Save session history to a JSON file."""
    ensure_session_dir()
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    data = {
        "session_id": session_id,
        "model_id": model_id,
        "history": history,
        "updated_at": str(os.times()[4]) # simplified timestamp
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def load_session(session_id: str) -> Dict[str, Any]:
    """Load session data from a JSON file."""
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return {"session_id": session_id, "history": []}
    with open(path, "r") as f:
        return json.load(f)

def list_sessions() -> List[str]:
    """List all available session IDs."""
    ensure_session_dir()
    files = os.listdir(SESSION_DIR)
    return [f.replace(".json", "") for f in files if f.endswith(".json")]

def get_new_session_id() -> str:
    return str(uuid.uuid4().hex[:8])
