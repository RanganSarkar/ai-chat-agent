from transformers import pipeline
import gradio as gr

# Use text-generation pipeline instead
generator = pipeline("text-generation", model="microsoft/DialoGPT-small")

def respond(message):
    # Add prompt style formatting (DialoGPT expects dialogue format)
    input_text = f"User: {message}\nAI:"
    response = generator(input_text, max_length=100, pad_token_id=50256)[0]['generated_text']
    
    # Only return the AI's part of the response
    return response.split("AI:")[-1].strip()

# Gradio UI
demo = gr.Interface(fn=respond, inputs="text", outputs="text", title="AI Chat Agent")

if __name__ == "__main__":
    demo.launch()
