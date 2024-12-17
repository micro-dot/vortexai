from flask import Flask
import gradio as gr
from huggingface_hub import InferenceClient
app = Flask(__name__)
# Initialize the client with the fine-tuned model
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")  # Update if using another model

# Function to validate inputs
def validate_inputs(max_tokens, temperature, top_p):
    if not (1 <= max_tokens <= 32768):
        raise ValueError("Max tokens must be between 1 and 32768.")
    if not (0.1 <= temperature <= 4.0):
        raise ValueError("Temperature must be between 0.1 and 4.0.")
    if not (0.1 <= top_p <= 1.0):
        raise ValueError("Top-p must be between 0.1 and 1.0.")

# Response generation
def respond(message, history, system_message, max_tokens, temperature, top_p):
    validate_inputs(max_tokens, temperature, top_p)
    
    # Prepare messages for the model
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:  # User's message
            messages.append({"role": "user", "content": val[0]})
        if val[1]:  # Assistant's response
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    
    # Generate response with streaming
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Updated system message
system_message = """
You are an advanced AI assistant specialized in coding tasks. 
- You deliver precise, error-free code in multiple programming languages.
- Analyze queries for logical accuracy and provide optimized solutions.
- Ensure clarity, brevity, and adherence to programming standards.

Guidelines:
1. Prioritize accurate, functional code.
2. Provide explanations only when necessary for understanding.
3. Handle tasks ethically, respecting user intent and legal constraints.

Thank you for using this system. Please proceed with your query.
"""

# Gradio Interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value=system_message, label="System message", lines=10),
        gr.Slider(minimum=1, maximum=32768, value=17012, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
