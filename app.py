# ================================================
# DRAMAGEN AI - BACKEND
# Fixed CORS version
# ================================================

from flask import Flask, request, jsonify, send_file, make_response
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

# ================================================
# CORS - Handle everything manually
# ================================================

def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

@app.after_request
def after_request(response):
    return add_cors(response)

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = make_response()
        response.status_code = 200
        return add_cors(response)

# ================================================
# API KEYS
# ================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/Lykon/dreamshaper-8"
TEMP_DIR = tempfile.mkdtemp()

print("=== DRAMAGEN AI BACKEND STARTED ===")
print("GROQ key set: " + str(bool(GROQ_API_KEY)))
print("HF key set: " + str(bool(HF_API_KEY)))
print("TEMP DIR: " + TEMP_DIR)

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
# ROUTES
# ================================================

@app.route("/", methods=["GET", "OPTIONS"])
def home():
    return jsonify({
        "status": "ok",
        "message": "DramaGen AI Backend is running"
    })

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok"})

@app.route("/test", methods=["GET", "OPTIONS"])
def test():
    return jsonify({
        "status": "ok",
        "groq_key_set": bool(GROQ_API_KEY),
        "hf_key_set": bool(HF_API_KEY),
        "groq_key_length": len(GROQ_API_KEY),
        "hf_key_length": len(HF_API_KEY)
    })

# ================================================
# STORY SPLITTER
# ================================================

def split_story(story, num_scenes):
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            prompt = (
                "You are a storyboard artist. "
                "Split this drama story into exactly " + str(num_scenes) + " visual scenes. "
                "Each scene is a detailed image description including setting, emotion, action and lighting. "
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
                    print("Groq success: " + str(len(scenes)) + " scenes")
                    return scenes[:num_scenes]
        except Exception as e:
            print("Groq error: " + str(e))

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
    print("Generating image: " + prompt[:60])

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
            print("Attempt " + str(attempt + 1))
            resp = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            print("HTTP Status: " + str(resp.status_code))

            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content))
                print("Image OK")
                return img
            elif resp.status_code == 503:
                wait = 30 + (attempt * 10)
                print("Model loading. Waiting " + str(wait) + "s")
                time.sleep(wait)
            elif resp.status_code == 429:
                print("Rate limited. Waiting 30s")
                time.sleep(30)
            else:
                print("HF Error: " + str(resp.status_code))
                print(resp.text[:200])
                break
        except Exception as e:
            print("Request error: " + str(e))
            time.sleep(10)

    print("All attempts failed. Using placeholder.")
    return Image.new("RGB", (768, 432), color=(20, 20, 40))

# ================================================
# VIDEO BUILDER
# ================================================

def build_video(images, seconds_per_scene, job_id):
    path = os.path.join(TEMP_DIR, job_id + ".mp4")
    fps = 24
    spf = fps * seconds_per_scene
    print("Building video: " + str(len(images)) + " scenes")

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
# GENERATE
# ================================================

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    print("\n=== GENERATE REQUEST ===")

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            print("No JSON data received")
            return jsonify({"error": "No data received"}), 400

        story = str(data.get("story", "")).strip()
        char_name = str(data.get("char_name", "Maya")).strip()
        char_look = str(data.get("char_look", "cartoon character")).strip()
        style_key = str(data.get("style", "pixar"))
        num_scenes = min(8, max(1, int(data.get("num_scenes", 6))))
        sec_per_scene = min(15, max(5, int(data.get("sec_per_scene", 10))))

        print("Story: " + str(len(story)) + " chars")
        print("Scenes: " + str(num_scenes))
        print("Style: " + style_key)

        if len(story) < 20:
            return jsonify({"error": "Story is too short"}), 400

        job_id = str(uuid.uuid4())[:8]
        print("Job: " + job_id)

        scenes = split_story(story, num_scenes)
        print("Got " + str(len(scenes)) + " scenes")

        images = []
        char_desc = "main character " + char_name + " who is " + char_look

        for i, scene in enumerate(scenes):
            print("Image " + str(i + 1) + "/" + str(len(scenes)))
            img = generate_image(scene + ", " + char_desc, style_key)
            images.append(img)

        video_path = build_video(images, sec_per_scene, job_id)

        previews = []
        for img in images:
            buf = io.BytesIO()
            img.resize((384, 216)).save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode()
            previews.append("data:image/jpeg;base64," + b64)

        total = len(scenes) * sec_per_scene
        print("Complete. Duration: " + str(total) + "s")

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
# DOWNLOAD
# ================================================

@app.route("/download/<job_id>", methods=["GET", "OPTIONS"])
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
