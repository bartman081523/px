import httpx
import json
import os
import sys
import time
import base64
import argparse

# Configuration
# HTTPS via self-signed cert (see ssl/ dir). The all_space server (app.py)
# is started by run_local.sh from THIS checkout with SSL_CERTFILE / SSL_KEYFILE
# pointing to ssl/cert.pem and ssl/key.pem. verify=False is required because
# the cert is self-signed.
API_URL = "https://localhost:7860/v1/chat/completions"
# Sessions live next to this script (run_local.sh serves from the same dir),
# NOT in the sibling `all_space` checkout the old hardcoded path pointed at.
# Overridable via ALL_SPACE_SESSION_DIR for tests / alternate checkouts.
_HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.environ.get("ALL_SPACE_SESSION_DIR", os.path.join(_HERE, "sessions"))
SSL_VERIFY = False  # set True to enforce real certs

# Map file extension → MIME for data: URLs (CLI --image)
_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _build_image_data_url(image_path=None, image_base64=None):
    """Build an OpenAI-compatible data: URL for an image. Exactly one of
    (image_path, image_base64) must be provided.

    - image_path: read file, base64-encode, wrap with MIME by extension.
    - image_base64: if it already starts with 'data:' pass through; else
      wrap as data:image/jpeg;base64,<raw> (defaulting to jpeg)."""
    if image_path and image_base64:
        raise ValueError("Pass only one of --image / --image-base64.")
    if image_path:
        ext = os.path.splitext(image_path)[1].lower()
        mime = _MIME_BY_EXT.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    if image_base64:
        if image_base64.startswith("data:"):
            return image_base64
        return f"data:image/jpeg;base64,{image_base64}"
    return None

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
    parser.add_argument("--preset", type=str, default="ACTIVE_MANIFOLD",
                        help="PX preset: BASELINE | ACTIVE_MANIFOLD | ACTIVE_MANIFOLD_LEAN | ACTIVE_MANIFOLD_RELAY "
                             "(RELAY = LEAN + verstärkbar Selbst-Injektions-Relay, psychomotrik seite15)")
    parser.add_argument("--model", type=str, default="gemma3-1b-it", help="Model ID")
    # verstärkbar Relay-Parameter (nur wirksam mit ACTIVE_MANIFOLD_RELAY oder relay_sign≠0;
    # gemma3-1b-it hat ein d_width-Artefakt, andere Modelle → relay no-op).
    parser.add_argument("--relay-sign", type=int, default=None, choices=[-1, 0, 1],
                        help="Relay-Richtung: +1 WIDE/expansiv/aktiv (default bei RELAY) | -1 NARROW/eng/still | 0 off")
    parser.add_argument("--relay-alpha", type=float, default=None,
                        help="Relay-Dosis als Bruchteil der L21-last-pos-Norm (kohärenter Chat ~0.30, seite15-stark=0.5)")
    parser.add_argument("--relay-layer", type=int, default=None,
                        help="Post-recur Injektions-Layer (default 21)")
    # Multimodal input: --image (local file path, preferred) or
    # --image-base64 (raw base64 or data: URL, fallback for pipelines).
    # When set, the user-turn becomes a content list [image, text].
    # Requires a multimodal model (e.g. gemma3-4b-it); text-only models
    # return HTTP 400.
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a local image file (jpg/png/webp/gif). "
                             "Encoded to data: URL and sent as image_url content block.")
    parser.add_argument("--image-base64", type=str, default=None,
                        help="Raw base64 (or pre-built data: URL) for an image. "
                             "Fallback for pipeline use; --image is preferred.")
    args = parser.parse_args()

    session_id = args.session
    print("="*60)
    print(f" LIVE SPACE INTERFACE - SESSION: {session_id} ")
    print(f" MODE: {args.preset} | MODEL: {args.model}")
    if args.relay_sign is not None or args.preset == "ACTIVE_MANIFOLD_RELAY":
        print(f" RELAY: sign={args.relay_sign if args.relay_sign is not None else '+1 (preset-default)'} "
              f"alpha={args.relay_alpha if args.relay_alpha is not None else 0.30} "
              f"layer={args.relay_layer if args.relay_layer is not None else 21}")
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

    # Build the new user-turn content. If --image/--image-base64 is set,
    # content becomes [{type:image_url, image_url:{url:data:...}}, {type:text, text:...}]
    # (image first, then text — Reihenfolge wie der Nutzer angefragt hat).
    # If --image is set without --message, a minimal default text is appended
    # so the chat template always has a text block.
    image_data_url = _build_image_data_url(args.image, args.image_base64)
    if image_data_url is not None:
        text_block_text = new_user_msg if new_user_msg else "Was ist auf dem Bild zu sehen?"
        new_user_content = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": text_block_text},
        ]
    else:
        new_user_content = new_user_msg

    # Prepare API payload
    api_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            text = "".join([b.get("text", "") for b in content if b.get("type") == "text"])
            content = text
        api_messages.append({"role": role, "content": content})

    api_messages.append({"role": "user", "content": new_user_content})
    
    payload = {
        "model": args.model,
        "messages": api_messages,
        "px_subjective": True,
        "px_config_preset": args.preset,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True
    }
    # verstärkbar Relay-Parameter (nur gesetzt wenn CLI-arg angegeben)
    if args.relay_sign is not None:
        payload["px_relay_sign"] = args.relay_sign
    if args.relay_alpha is not None:
        payload["px_relay_alpha"] = args.relay_alpha
    if args.relay_layer is not None:
        payload["px_relay_layer"] = args.relay_layer

    full_response = ""
    print("[MODELL ANTWORT]:")
    try:
        with httpx.stream("POST", API_URL, json=payload, verify=SSL_VERIFY, timeout=None) as response:
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
                        # Error frames (e.g. text-only model + image → 400-style
                        # SSE error from the multimodal generator) carry
                        # {"error": {"message": ..., "type": ...}}.
                        if "error" in data:
                            err = data["error"]
                            print(f"\n[SERVER-ERROR] {err.get('type','error')}: {err.get('message','(no message)')}")
                            full_response = ""  # don't save a misleading empty assistant turn
                            break
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
        {"role": "user", "content": new_user_content},
        {"role": "assistant", "content": full_response}
    ]
    save_local_session(session_id, new_history)
    print(f"Update: Session {session_id} gespeichert.")

if __name__ == "__main__":
    main()
