from gradio_client import Client, handle_file
import os
import time
import requests

def test_client():
    url = "http://127.0.0.1:7860"
    print(f"Connecting to {url}...")
    
    # Wait for server to be really ready
    max_retries = 5
    for i in range(max_retries):
        try:
            requests.get(url)
            print("Server is responding to GET.")
            break
        except:
            print(f"Waiting for server... ({i+1}/{max_retries})")
            time.sleep(5)
    
    client = Client(url)
    
    # 1. Prepare dummy image
    img_path = "/run/media/julian/ML4/ollama-work/all_space/sessions/temp_media/test_client.png"
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))
    from PIL import Image
    Image.new("RGB", (100, 100), color="blue").save(img_path)

    print(f"Sending image {img_path} to Gradio server...")
    
    try:
        # In Gradio 6, the MultimodalTextbox value is expected.
        # Format: {"text": "...", "files": [path]}
        result = client.predict(
            {"text": "Test image upload.", "files": [handle_file(img_path)]},
            [], # history
            api_name="/user_message"
        )
        print("user_message result:", result)
        
        # history is the second element in the returned tuple from user_message
        updated_history = result[1]
        
        print("Calling bot_response with history:", updated_history)
        job = client.submit(
            updated_history,
            "gemma3-270m-it", # model_id
            "BASELINE",      # px_preset
            0.7,             # temp
            0.9,             # tp
            20,              # mt (Short for test)
            1.1,             # rp
            0.08,            # gamma
            False,           # visual_screen
            False,           # visual_feedback
            False,           # infinite_context
            "client_test_sid", # session_id
            api_name="/bot_response"
        )
        
        print("Waiting for response...")
        final_history = None
        for outputs in job:
            final_history = outputs
            print(".", end="", flush=True)
        
        print("\nFinal response received!")
        if final_history:
             print("Last message:", final_history[-1]["content"])
        else:
             print("No history returned.")

    except Exception as e:
        print("\nCRASH DETECTED by Client:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_client()
