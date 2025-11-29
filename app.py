# app.py
import os
import datetime
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image as PILImage, ImageDraw, ImageFont
import gradio as gr

# Optional: load .env automatically if you store token there
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if not installed, no problem
    pass

# ---------- Configuration ----------
# Preferred model (requires Hugging Face token)
MODEL_ID_AUTH = "runwayml/stable-diffusion-v1-5"
# Public fallback model (no token required)
MODEL_ID_NO_AUTH = "stabilityai/stable-diffusion-2-1-base"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def get_hf_token():
    """
    Return Hugging Face token if set in environment or .env file.
    Example: setx HUGGINGFACE_TOKEN "your_token_here"
    """
    return os.getenv("HUGGINGFACE_TOKEN")


def load_pipeline(preferred_model=MODEL_ID_AUTH, fallback_model=MODEL_ID_NO_AUTH):
    """
    Try loading preferred_model with token (if available).
    If that fails (no token or loading error), try fallback_model without auth.
    """
    token = get_hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # First attempt: preferred model with token
    if token:
        try:
            print(f"Attempting to load preferred model '{preferred_model}' with Hugging Face token...")
            pipe = StableDiffusionPipeline.from_pretrained(
                preferred_model,
                torch_dtype=torch_dtype,
                use_auth_token=token
            )
            pipe = pipe.to(device)
            if device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            print(f"✅ Loaded '{preferred_model}' successfully (with token).")
            return pipe
        except Exception as e:
            print(f"❌ Failed to load '{preferred_model}' with token: {e}")
            print("👉 Falling back to public model...")

    # Second attempt: fallback public model
    try:
        print(f"Attempting to load fallback public model '{fallback_model}' (no token required)...")
        pipe = StableDiffusionPipeline.from_pretrained(
            fallback_model,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        pipe = pipe.to(device)
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        print(f"✅ Loaded '{fallback_model}' successfully (no token).")
        return pipe
    except Exception as e:
        raise RuntimeError(
            "❌ Unable to load any model.\n"
            "If you intended to use a token-protected model, set your token first:\n"
            "   setx HUGGINGFACE_TOKEN \"your_token_here\"\n\n"
            f"Error details: {e}"
        )

# Lazy-load pipeline global
PIPE = None

def add_text_overlay(pil_image: PILImage.Image, text: str) -> PILImage.Image:
    if pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(pil_image)

    font_size = max(18, pil_image.width // 25)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w, img_h = pil_image.size
    x = (img_w - text_w) / 2
    y = img_h - text_h - 10

    padding = 8
    rect_coords = [(x - padding, y - padding), (x + text_w + padding, y + text_h + padding)]
    overlay = PILImage.new('RGBA', pil_image.size, (0,0,0,0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(rect_coords, fill=(0,0,0,160))
    pil_image = PILImage.alpha_composite(pil_image, overlay)

    draw = ImageDraw.Draw(pil_image)
    draw.text((x, y), text, font=font, fill=(255,255,255,255))

    return pil_image.convert("RGB")

def generate_image(prompt: str, filename: str = None, guidance_scale: float = 7.5, num_inference_steps: int = 50, height: int = None, width: int = None):
    global PIPE
    if PIPE is None:
        PIPE = load_pipeline(MODEL_ID_AUTH, MODEL_ID_NO_AUTH)

    if not prompt or prompt.strip() == "":
        return None, "Please enter a prompt."

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not filename:
        safe_name = "".join(c for c in prompt[:40] if c.isalnum() or c in (" ", "_")).replace(" ", "_")
        filename = f"{safe_name}_{timestamp}.png"
    if not filename.lower().endswith(".png"):
        filename += ".png"
    out_path = os.path.join(OUTPUT_DIR, filename)

    gen_kwargs = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": int(num_inference_steps),
    }
    if height:
        gen_kwargs["height"] = int(height)
    if width:
        gen_kwargs["width"] = int(width)

    try:
        print("Generating image — this can take a while depending on your hardware...")
        result = PIPE(**gen_kwargs)
        image = result.images[0]
        image_with_text = add_text_overlay(image, prompt)
        image_with_text.save(out_path)
        print(f"Saved to {out_path}")
        return out_path, "Image generated successfully."
    except Exception as e:
        print("Generation error:", e)
        return None, f"Generation error: {e}"

# ---------- Gradio UI ----------
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Stable Diffusion (local) — VS Code friendly")
        with gr.Row():
            prompt_input = gr.Textbox(label="Image Description (prompt)", lines=2, placeholder="A fantasy landscape, painting...")
            filename_input = gr.Textbox(label="Output filename (optional, .png appended if missing)", placeholder="my_image")
        with gr.Row():
            steps = gr.Slider(10, 150, value=50, step=1, label="Steps (num_inference_steps)")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
        generate_btn = gr.Button("Generate")
        output_img = gr.Image(label="Generated image (saved to outputs/)", interactive=False)
        status_txt = gr.Textbox(label="Status", interactive=False)

        def on_generate(prompt, filename, steps, guidance):
            out_path, status = generate_image(prompt, filename, guidance_scale=guidance, num_inference_steps=steps)
            return out_path if out_path else None, status

        generate_btn.click(on_generate, inputs=[prompt_input, filename_input, steps, guidance], outputs=[output_img, status_txt])

    demo.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
