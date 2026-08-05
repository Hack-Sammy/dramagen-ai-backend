# ================================================
# DRAMAGEN AI - BACKEND
# Runs on Render.com for free
# ================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import re
import json
import gc
import uuid
import tempfile
import requests
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from groq import Groq
import io
import base64
import time

app = Flask(__name__)

# Fix CORS - Allow requests from anywhere
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Handle OPTIONS preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

# Add CORS headers to every response
@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# API Keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY   = os.environ.get("HF_API_KEY", "")

# Hugging Face image model
HF_API_URL = "https://api-inference.huggingface.co/models/Lykon/dreamshaper-8"

# Temp storage for videos
TEMP_DIR = tempfile.mkdtemp()

print("Backend started.")
print("GROQ key set:", bool(GROQ_API_KEY))
print("HF key set:", bool(HF_API_KEY))

# ================================================
# ART STYLES
# ================================================

STYLES = {
    "pixar": {
        "prefix": "pixar style, disney 3d animation, vibrant colors, expressive character, smooth 3d render, cinematic lighting, highly detailed, movie still, professional animation",
        "negative": "realistic, photograph, blurry, ugly, distorted, bad anatomy, watermark, text, low quality, sketch, flat"
    },
    "anime": {
        "prefix": "anime style, studio ghibli, beautiful anime illustration, emotional scene, detailed background, vibrant colors, high quality",
        "negative": "realistic, photo, 3d render, blurry, ugly, watermark, text, low quality"
    },
    "comic": {
        "prefix": "comic book style, bold outlines, dramatic colors, professional comic illustration, dynamic scene, detailed",
        "negative": "realistic, photo, blurry, watermark, text, low quality, 3d render"
    },
    "watercolor": {
        "prefix": "watercolor illustration, soft artistic painting, beautiful emotional scene, storybook illustration, detailed",
        "negative": "realistic, photo, 3d, blurry, watermark, text, low quality"
    }
}

# ================================================
# STORY SPLITTER
# ================================================

def split_with_groq(story, num_scenes):
    if not GROQ_API_KEY:
        print("No Groq key. Using fallback splitter.")
        return fallback_split(story, num_scenes)
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are a professional storyboard artist.
Break this drama story into exactly {num_scenes} visual scenes.
Each scene must be a vivid visual description suitable for an AI image generator.
Each description must include: location, character emotion, action, lighting.
Keep the main character visually consistent across all scenes.
Story: {story}
Return ONLY a valid JSON array of exactly {num_scenes} strings.
No explanation. No extra text. Just the JSON array."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

        text = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            scenes = json.loads(match.group(0))
            if isinstance(scenes, list) and len(scenes) > 0:
                print(f"Groq returned {len(scenes)} scenes.")
                return scenes[:num_scenes]

    except Exception as e:
        print(f"Groq error: {e}")

    return fallback_split(story, num_scenes)

def fallback_split(story, num_scenes):
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return ["A dramatic emotional scene in a love story"] * num_scenes
    group_size = max(1, len(sentences) // num_scenes)
    scenes = []
    for i in range(0, len(sentences), group_size):
        scenes.append(" ".join(sentences[i:i+group_size]))
        if len(scenes) == num_scenes:
            break
    while len(scenes) < num_scenes:
        scenes.append(scenes[-1])
    return scenes[:num_scenes]

# ================================================
# IMAGE GENERATOR
# ================================================

def generate_image(prompt, style_key):
    style = STYLES.get(style_key, STYLES["pixar"])
    full_prompt = style["prefix"] + ", " + prompt

    headers = {"Authorization": "Bearer " + HF_API_KEY}
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "negative_prompt": style["negative"],
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 896,
            "height": 512
        }
    }

    for attempt in range(5):
        try:
            print(f"Generating image attempt {attempt + 1}...")
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                print("Image generated successfully.")
                return image

            elif response.status_code == 503:
                wait_time = 20 + (attempt * 10)
                print(f"Model loading. Waiting {wait_time} seconds...")
                time.sleep(wait_time)

            elif response.status_code == 429:
                print("Rate limited. Waiting 30 seconds...")
                time.sleep(30)

            else:
                print(f"HF API error {response.status_code}: {response.text[:200]}")
                break

        except Exception as e:
            print(f"Request error: {e}")
            time.sleep(10)

    print("All attempts failed. Using placeholder.")
    img = Image.new("RGB", (896, 512), color=(20, 20, 40))
    return img

# ================================================
# VIDEO BUILDER
# ================================================

def build_video(images, seconds_per_scene, job_id):
    output_path = os.path.join(TEMP_DIR, job_id + ".mp4")
    fps = 24
    frames_per_scene = fps * seconds_per_scene

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p"
    )

    for i, image in enumerate(images):
        print(f"Writing scene {i+1} to video...")
        frame = np.array(image.resize((896, 512)))
        for _ in range(frames_per_scene):
            writer.append_data(frame)

    writer.close()
    print(f"Video saved: {output_path}")
    return output_path

# ================================================
# API ROUTES
# ================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "DramaGen AI Backend is running",
        "version": "1.0"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        story         = data.get("story", "").strip()
        char_name     = data.get("char_name", "the main character").strip()
        char_look     = data.get("char_look", "expressive cartoon character").strip()
        style_key     = data.get("style", "pixar")
        num_scenes    = int(data.get("num_scenes", 6))
        sec_per_scene = int(data.get("sec_per_scene", 10))

        if not story or len(story) < 20:
            return jsonify({"error": "Story is too short"}), 400

        if num_scenes < 1: num_scenes = 6
        if num_scenes > 8: num_scenes = 8
        if sec_per_scene < 5: sec_per_scene = 5
        if sec_per_scene > 15: sec_per_scene = 15

        job_id = str(uuid.uuid4())[:8]
        print(f"\n[{job_id}] New generation request")
        print(f"[{job_id}] Scenes: {num_scenes} | Style: {style_key}")

        print(f"[{job_id}] Splitting story...")
        scenes = split_with_groq(story, num_scenes)
        print(f"[{job_id}] Got {len(scenes)} scenes")

        images = []
        char_anchor = (
            "main character " + char_name +
            " who is " + char_look +
            ", consistent character appearance"
        )

        for i, scene in enumerate(scenes):
            print(f"[{job_id}] Image {i+1}/{len(scenes)}: {scene[:60]}...")
            prompt = scene + ", " + char_anchor + ", cinematic, emotional drama"
            image = generate_image(prompt, style_key)
            images.append(image)

        print(f"[{job_id}] Building video...")
        video_path = build_video(images, sec_per_scene, job_id)

        scene_images_b64 = []
        for img in images:
            buffer = io.BytesIO()
            img_resized = img.resize((448, 256))
            img_resized.save(buffer, format="JPEG", quality=80)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            scene_images_b64.append("data:image/jpeg;base64," + b64)

        total_duration = len(scenes) * sec_per_scene
        print(f"[{job_id}] Complete. Duration: {total_duration}s")

        return jsonify({
            "success": True,
            "job_id": job_id,
            "scenes": scenes,
            "scene_images": scene_images_b64,
            "video_url": "/download/" + job_id,
            "total_duration": total_duration,
            "total_scenes": len(scenes)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    job_id = re.sub(r'[^a-zA-Z0-9\-]', '', job_id)
    video_path = os.path.join(TEMP_DIR, job_id + ".mp4")

    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    return send_file(
        video_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name="DramaGen_AI_Video.mp4"
    )

# ================================================
# START SERVER
# ================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
