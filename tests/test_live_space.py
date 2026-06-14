import os
import sys
import json
import time
from gradio_client import Client

def test_live_history():
    print("Connecting to live space at https://localhost:7860/gradio/...")
    try:
        client = Client("https://localhost:7860/gradio/", ssl_verify=False)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    session_id = "live-test-session-999"
    # Ensure session doesn't exist yet
    path = os.path.join("sessions", f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)

    print(f"Sending first message to session {session_id}...")
    # predict(message, model, px_preset, temp, tp, mt, rp, gamma, session_id_state, api_name="/chat_wrapper")
    res1 = client.predict(
        "First message",         # param_0 (message)
        "gemma3-270m-it",        # param_2 (model)
        "ACTIVE_MANIFOLD",       # param_3 (preset)
        0.7, 0.95, 128, 1.15, 0.08, # params 4-8
        session_id,              # param_9 (session_id_state)
        api_name="/chat_wrapper"
    )
    print(f"Response 1 received: {res1[:50]}...")

    # Wait a bit for async file saving
    time.sleep(1)

    if not os.path.exists(path):
        print(f"ERROR: Session file {path} was not created!")
        return

    with open(path, "r") as f:
        data = json.load(f)
        history = data.get("history", [])
        print(f"History length after msg 1: {len(history)}")
        if len(history) < 2:
            print("ERROR: History too short after message 1")
            return

    print("Sending second message to the SAME session...")
    res2 = client.predict(
        "Second message",
        "gemma3-270m-it",
        "ACTIVE_MANIFOLD",
        0.7, 0.95, 128, 1.15, 0.08,
        session_id,
        api_name="/chat_wrapper"
    )
    print(f"Response 2 received: {res2[:50]}...")

    time.sleep(1)

    with open(path, "r") as f:
        data = json.load(f)
        history = data.get("history", [])
        print(f"Final history length: {len(history)}")
        for i, msg in enumerate(history):
            print(f"  [{i}] {msg['role']}: {msg['content'][:30]}...")

        # We expect 4 messages: User(1), Assistant(1), User(2), Assistant(2)
        # OR 3 if merged (but User 1 and User 2 are separated by Assistant 1, so no merge)
        if len(history) == 4:
            print("\n[SUCCESS] Live test passed: History correctly appended.")
        else:
            print(f"\n[FAILURE] History length is {len(history)}, expected 4.")

if __name__ == "__main__":
    test_live_history()
