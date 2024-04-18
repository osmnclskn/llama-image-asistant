import base64
from io import BytesIO

import pyttsx3
import streamlit as st
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image

# LLaVA Model
model_mmproj_file = "llava-v1.5-7b-mmproj-f16.gguf"
model_file = "llava-v1.5-7b-Q4_K.gguf"


def load_chat_handler():
    """Load LLAVA chat handler"""
    return Llava15ChatHandler(clip_model_path=model_mmproj_file)


def load_model():
    """Load model"""
    chat_handler = load_chat_handler()
    return Llama(
        model_path=model_file,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1,  # Set to 0 if you don't have a GPU,
        verbose=True,
        logits_all=True,
    )


def st_describe(model, prompt, image):
    """Describe image with a prompt in browser"""
    with st.spinner("Describing the image..."):
        response = model_inference(model, prompt, image)
    st.text(response)
    text_to_speech(response)


def image_b64encode(img):
    """Convert image to a base64 format"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def model_inference(model, request, image):
    """Ask model a question"""
    image_b64 = image_b64encode(image)
    out_stream = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who perfectly describes images.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": request},
                ],
            },
        ],
        stream=True,
        temperature=0.2,
    )

    # Get characters from stream
    output = ""
    for r in out_stream:
        data = r["choices"][0]["delta"]
        if "content" in data:
            output += data["content"]
    return output


def text_to_speech(text):
    """Convert text to speech"""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    """Main app"""
    st.title("LLAVA AI Assistant")

    with st.spinner("Loading the model, please wait..."):
        model = load_model()

    img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file_buffer:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prompt = st.text_input("Enter your prompt", "Please describe the image.")
        if st.button("Describe"):
            st_describe(model, prompt, image)
        if st.button("Repeat"):
            st_describe(model, prompt, image)


if __name__ == "__main__":
    main()
