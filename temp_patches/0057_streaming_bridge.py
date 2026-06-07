import httpx
import json
import os
import sys
import time
import argparse

# Configuration
API_URL = "https://localhost:7860/v1/chat/completions"
SESSION_DIR = "/run/media/julian/ML4/ollama-work/all_space/sessions"

def load_local_session(session_id):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return {"session_id": session_id, "history": []}
    with open(path, "r") as f:
        return json.load(f)

def save_local_session(session_id, history):
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    data = load_local_session(session_id)
    data["history"] = history
    data["updated_at"] = str(time.time())
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default="aab82b16", help="Session ID to load or create")
    parser.add_argument("--message", type=str, help="The message to send (if omitted, will ask)")
    parser.add_argument("--preset", type=str, default="SUBJECTIVE", help="The PX preset (e.g. SUBJECTIVE, RESONANCE_CITY)")
    parser.add_argument("--model", type=str, default="gemma3-1b-it", help="Model ID")
    args = parser.parse_args()

    session_id = args.session
    print("="*60)
    print(f" LIVE SPACE INTERFACE - SESSION: {session_id} ")
    print(f" MODE: {args.preset} | MODEL: {args.model}")
    print("="*60)
    
    session_data = load_local_session(session_id)
    history = session_data.get("history", [])
    
    # Show last context
    if history:
        last_msg = history[-1]
        print(f"\n[LETZTER KONTEXT - {last_msg['role'].upper()}]:")
        content = last_msg['content']
        if isinstance(content, list):
             text = "".join([b.get("text", "") for b in content if b.get("type") == "text"])
             print(text[:300] + "..." if len(text) > 300 else text)
        else:
             print(content[:300] + "..." if len(content) > 300 else content)
    else:
        print("\n[NEUE SESSION GESTARTET]")

    new_user_msg = args.message
    if not new_user_msg:
        print("\n" + "-"*60)
        new_user_msg = input("Eingabe (oder leer lassen für Standard-Fortsetzung): ").strip()
        if not new_user_msg:
             new_user_msg = "Bitte setze deine Gedanken fort."

    print("\n" + "-"*20 + " MEIN INPUT (GEMINI CLI) " + "-"*20)
    print(new_user_msg)
    print("-" * 60)
    print("\n[WARTE AUF ANTWORT VON ALL_SPACE...]\n")

    # Prepare API payload
    api_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            text = "".join([b.get("text", "") for b in content if b.get("type") == "text"])
            content = text
        api_messages.append({"role": role, "content": content})
    
    api_messages.append({"role": "user", "content": new_user_msg})
    
    payload = {
        "model": args.model,
        "messages": api_messages,
        "px_subjective": True,
        "px_config_preset": args.preset,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True
    }

    full_response = ""
    print("[MODELL ANTWORT]:")
    try:
        with httpx.stream("POST", API_URL, json=payload, verify=False, timeout=None) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                try:
                    print(response.read().decode())
                except:
                    pass
                return

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            content = delta["content"]
                            full_response += content
                            sys.stdout.write(content)
                            sys.stdout.flush()
                    except:
                        pass
    except Exception as e:
        print(f"\n[FEHLER]: {e}")
        return

    print("\n\n" + "="*60)
    
    # Save session
    new_history = history + [
        {"role": "user", "content": new_user_msg},
        {"role": "assistant", "content": full_response}
    ]
    save_local_session(session_id, new_history)
    print(f"Update: Session {session_id} gespeichert.")

if __name__ == "__main__":
    main()
