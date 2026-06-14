import gradio as gr
import inspect
import asyncio

# 1. Define a generator function (similar to chat_fn)
def my_generator(message, history):
    for i in range(3):
        yield f"Step {i} for {message}"

# 2. Define a lambda that calls it (similar to how it's done in chat_tab.py)
# This lambda returns a generator OBJECT, but is not a generator FUNCTION.
my_lambda = lambda message, history: my_generator(message, history)

# 3. Define a wrapper that uses yield from
def my_wrapper(message, history):
    yield from my_generator(message, history)

async def simulate_gradio_issue():
    print(f"Is my_generator a generator function? {inspect.isgeneratorfunction(my_generator)}")
    print(f"Is my_lambda a generator function? {inspect.isgeneratorfunction(my_lambda)}")
    print(f"Is my_wrapper a generator function? {inspect.isgeneratorfunction(my_wrapper)}")

    print("\n--- Testing with lambda (expected to fail) ---")
    try:
        response = my_lambda("test", [])
        print(f"Response type: {type(response)}")
        response.get("files", [])
    except AttributeError as e:
        print(f"Caught expected AttributeError: {e}")

    print("\n--- Testing with wrapper (expected to work) ---")
    try:
        if inspect.isgeneratorfunction(my_wrapper):
            print("my_wrapper IS recognized as a generator function.")
            # Gradio would now iterate over it:
            response_gen = my_wrapper("test", [])
            print(f"Iterating over generator...")
            for val in response_gen:
                print(f"Yielded: {val}")
            print("Success!")
        else:
            print("my_wrapper is NOT recognized as a generator function.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_gradio_issue())
