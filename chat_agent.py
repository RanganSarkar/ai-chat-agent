from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import gradio as gr

# Load Flan-T5 Base model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Gradio chatbot logic
def respond(prompt):
    result = generator(prompt, max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    return result.strip()

# Gradio UI
demo = gr.Interface(
    fn=respond,
    inputs="text",
    outputs="text",
    title="Flan-T5 Chat Agent",
    description="Fast and smart instruction-tuned chatbot that runs on 8GB RAM / 4GB VRAM.",
)

if __name__ == "__main__":
    demo.launch()
