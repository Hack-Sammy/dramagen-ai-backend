# ================================================
# DRAMAGEN AI - BACKEND
# Simple and reliable version
# ================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import re
import json
import uuid
import tempfile
import requests
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import io
import base64
import time

app = Flask(__name__)

# Allow ALL origins - fixes CORS completely
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok", "message": "DramaGen AI is running"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "DramaGen AI Backend"})

@app.route("/test", methods=["GET"])
def test():
    groq_key = os.environ.get("GROQ_API_KEY", "")
    hf_key = os.environ.get("HF_API_KEY", "")
    return jsonify({
        "status": "ok",
        "groq_key_set": bool(groq_key),
        "hf_key_set": bool(hf_key),
        "groq_key_length": len(groq_key),
        "hf_key_length": len(hf_key)
    })

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/Lykon/dreamshaper-8"

TEMP_DIR = tempfile.mkdtemp()
print("TEMP_DIR:", TEMP_DIR)
print("GROQ key set:", bool(GROQ_API_KEY))
print("HF key set:", bool(HF_API_KEY))

# ================================================
# STYLES
# ================================================

STYLES = {
    "pixar": {
        "prefix": "pixar style, disney 3d animation, vibrant colors, expressive character, smooth 3d render, cinematic lighting, highly detailed",
        "negative": "realistic, photograph, blurry, ugly, distorted, watermark, text, low quality"
    },
    "anime": {
        "prefix": "anime style, studio ghibli, beautiful illustration, emotional, detailed, vibrant colors",
        "negative": "realistic, photo, 3d render, blurry, watermark, text"
    },
    "comic": {
        "prefix": "comic book style, bold outlines, dramatic colors, professional illustration, dynamic",
        "negative": "realistic, photo, blurry, watermark, text, low quality"
    },
    "watercolor": {
        "prefix": "watercolor illustration, soft painting, beautiful, emotional, storybook art",
        "negative": "realistic, photo, 3d, blurry, watermark, text"
    }
}

# ================================================
# STORY SPLITTER
# ================================================

def split_story(story, num_scenes):
    # Try Groq first
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            prompt = (
                "You are a storyboard artist. "
                "Split this drama story into exactly " + str(num_scenes) + " visual scenes. "
                "Each scene is a detailed image description with setting, emotion, action, lighting. "
                "Story: " + story + " "
                "Return ONLY a JSON array of " + str(num_scenes) + " strings. Nothing else."
            )
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
                    print("Groq split successful: " + str(len(scenes)) + " scenes")
                    return scenes[:num_scenes]
        except Exception as e:
            print("Groq error: " + str(e))

    # Fallback: simple sentence split
    print("Using fallback splitter")
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return ["A dramatic emotional scene"] * num_scenes
    group_size = max(1, len(sentences) // num_scenes)
    scenes = []
    for i in range(0, len(sentences), group_size):
        scenes.append(" ".join(sentences[i:i + group_size]))
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
    print("Generating: " + prompt[:60])

    headers = {"Authorization": "Bearer " + HF_API_KEY}
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "negative_prompt": style["negative"],
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "width": 768,
            "height": 432
        }
    }

    for attempt in range(5):
        try:
            print("Attempt " + str(attempt + 1) + "...")
            resp = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            print("Status: " + str(resp.status_code))

            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content))
                print("Image OK: " + str(img.size))
                return img

            elif resp.status_code == 503:
                print("Model loading, waiting 30s...")
                time.sleep(30)

            elif resp.status_code == 429:
                print("Rate limited, waiting 20s...")
                time.sleep(20)

            else:
                print("Error: " + resp.text[:100])
                break

        except Exception as e:
            print("Request failed: " + str(e))
            time.sleep(10)

    # Return dark placeholder
    print("Using placeholder image")
    img = Image.new("RGB", (768, 432), color=(20, 20, 40))
    return img

# ================================================
# VIDEO BUILDER
# ================================================

def build_video(images, seconds_per_scene, job_id):
    path = os.path.join(TEMP_DIR, job_id + ".mp4")
    fps = 24
    spf = fps * seconds_per_scene
    print("Building video: " + str(len(images)) + " scenes x " + str(seconds_per_scene) + "s")

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p"
    )

    for i, img in enumerate(images):
        frame = np.array(img.resize((768, 432)))
        for _ in range(spf):
            writer.append_data(frame)
        print("Scene " + str(i + 1) + " written")

    writer.close()
    print("Video saved: " + path)
    return path

# ================================================
# GENERATE ENDPOINT
# ================================================

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    print("\n=== NEW REQUEST ===")

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data"}), 400

        story = str(data.get("story", "")).strip()
        char_name = str(data.get("char_name", "Maya")).strip()
        char_look = str(data.get("char_look", "cartoon character")).strip()
        style_key = str(data.get("style", "pixar"))
        num_scenes = min(8, max(1, int(data.get("num_scenes", 6))))
        sec_per_scene = min(15, max(5, int(data.get("sec_per_scene", 10))))

        print("Story length: " + str(len(story)))
        print("Scenes: " + str(num_scenes))
        print("Style: " + style_key)

        if len(story) < 20:
            return jsonify({"error": "Story too short"}), 400

        job_id = str(uuid.uuid4())[:8]
        print("Job ID: " + job_id)

        # Split story
        print("Splitting story...")
        scenes = split_story(story, num_scenes)
        print("Scenes: " + str(len(scenes)))

        # Generate images
        images = []
        char_desc = (
            "main character " + char_name +
            " who is " + char_look
        )

        for i, scene in enumerate(scenes):
            print("Image " + str(i + 1) + "/" + str(len(scenes)))
            prompt = scene + ", " + char_desc
            img = generate_image(prompt, style_key)
            images.append(img)

        # Build video
        print("Building video...")
        video_path = build_video(images, sec_per_scene, job_id)

        # Convert to base64 for preview
        previews = []
        for img in images:
            buf = io.BytesIO()
            img.resize((384, 216)).save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode()
            previews.append("data:image/jpeg;base64," + b64)

        total = len(scenes) * sec_per_scene
        print("Done! Duration: " + str(total) + "s")

        return jsonify({
            "success": True,
            "job_id": job_id,
            "scenes": scenes,
            "scene_images": previews,
            "video_url": "/download/" + job_id,
            "total_duration": total,
            "total_scenes": len(scenes)
        })

    except Exception as e:
        print("ERROR: " + str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================================
# DOWNLOAD ENDPOINT
# ================================================

@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    job_id = re.sub(r'[^a-zA-Z0-9\-]', '', job_id)
    path = os.path.join(TEMP_DIR, job_id + ".mp4")
    if not os.path.exists(path):
        return jsonify({"error": "Video not found"}), 404
    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name="DramaGen_AI_Video.mp4"
    )

# ================================================
# RUN
# ================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
