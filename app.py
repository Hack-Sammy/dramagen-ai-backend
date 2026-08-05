# ================================================
# DRAMAGEN AI
# Full stack - Frontend + Backend on Render
# ================================================

from flask import Flask, request, jsonify, send_file, render_template_string
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
CORS(app)

# ================================================
# API KEYS
# ================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY   = os.environ.get("HF_API_KEY", "")
HF_API_URL   = "https://api-inference.huggingface.co/models/Lykon/dreamshaper-8"
TEMP_DIR     = tempfile.mkdtemp()

print("=== DRAMAGEN AI STARTED ===")
print("GROQ key: " + str(bool(GROQ_API_KEY)))
print("HF key:   " + str(bool(HF_API_KEY)))

# ================================================
# THE WEBSITE HTML
# Served directly from Python
# Same domain = no CORS issues
# ================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DramaGen AI - Story to Cartoon Video</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a14;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, sans-serif;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            padding: 50px 20px;
            text-align: center;
            border-bottom: 1px solid #2a2a40;
        }
        .logo {
            font-size: 52px;
            font-weight: 900;
            background: linear-gradient(90deg, #f953c6, #7c4dff, #40c4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .tagline { color: #aaaacc; font-size: 18px; }
        .badges {
            margin-top: 15px;
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .badge {
            background: rgba(124,77,255,0.2);
            border: 1px solid #7c4dff;
            color: #b39dff;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            .logo { font-size: 36px; }
        }
        .panel {
            background: #13131f;
            border: 1px solid #2a2a40;
            border-radius: 20px;
            padding: 28px;
        }
        .panel-title {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #7c4dff;
            margin-bottom: 18px;
        }
        label {
            display: block;
            font-size: 13px;
            color: #aaaacc;
            margin-bottom: 6px;
            margin-top: 16px;
        }
        textarea, input[type="text"], select {
            width: 100%;
            background: #1e1e30;
            border: 1px solid #333355;
            border-radius: 10px;
            color: #ffffff;
            padding: 12px 14px;
            font-size: 14px;
            font-family: inherit;
            transition: border-color 0.2s;
            resize: vertical;
        }
        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: #7c4dff;
        }
        textarea { min-height: 160px; }
        .row-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .style-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 6px;
        }
        .style-btn {
            background: #1e1e30;
            border: 2px solid #333355;
            border-radius: 10px;
            color: #aaaacc;
            padding: 10px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s;
            text-align: center;
        }
        .style-btn:hover { border-color: #7c4dff; color: #fff; }
        .style-btn.active {
            border-color: #f953c6;
            background: rgba(249,83,198,0.15);
            color: #f953c6;
        }
        .slider-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #aaaacc;
            margin-bottom: 6px;
        }
        .slider-val { color: #f953c6; font-weight: 700; }
        input[type="range"] {
            width: 100%;
            accent-color: #7c4dff;
            cursor: pointer;
        }
        .generate-btn {
            width: 100%;
            background: linear-gradient(135deg, #f953c6, #7c4dff);
            border: none;
            border-radius: 14px;
            color: white;
            font-size: 18px;
            font-weight: 800;
            padding: 18px;
            cursor: pointer;
            margin-top: 22px;
            transition: all 0.3s;
            box-shadow: 0 8px 30px rgba(124,77,255,0.4);
        }
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(124,77,255,0.6);
        }
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .tips {
            background: #1a1a2e;
            border: 1px solid #2a2a40;
            border-radius: 12px;
            padding: 16px;
            margin-top: 18px;
            font-size: 12px;
            color: #666688;
            line-height: 1.8;
        }
        .status-bar {
            background: #1e1e30;
            border: 1px solid #333355;
            border-radius: 10px;
            padding: 14px;
            font-size: 13px;
            color: #aaaacc;
            margin-bottom: 18px;
            min-height: 48px;
            word-break: break-all;
        }
        .debug-box {
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 14px;
            font-size: 11px;
            color: #00ff00;
            margin-bottom: 18px;
            min-height: 80px;
            font-family: monospace;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 150px;
            overflow-y: auto;
        }
        .progress-wrap { display: none; margin-bottom: 18px; }
        .progress-label {
            font-size: 12px;
            color: #aaaacc;
            margin-bottom: 8px;
        }
        .progress-bar-outer {
            background: #1e1e30;
            border-radius: 20px;
            height: 8px;
            overflow: hidden;
        }
        .progress-bar-inner {
            height: 100%;
            background: linear-gradient(90deg, #f953c6, #7c4dff);
            border-radius: 20px;
            width: 0%;
            transition: width 0.5s ease;
        }
        .video-wrap {
            background: #1e1e30;
            border: 1px solid #333355;
            border-radius: 14px;
            overflow: hidden;
            aspect-ratio: 16/9;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 16px;
        }
        .video-placeholder {
            text-align: center;
            color: #444466;
        }
        .video-placeholder .icon { font-size: 48px; }
        .video-placeholder p { margin-top: 8px; font-size: 14px; }
        video { width: 100%; height: 100%; object-fit: contain; }
        .download-btn {
            display: none;
            width: 100%;
            background: linear-gradient(135deg, #00c853, #00796b);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 16px;
            font-weight: 700;
            padding: 14px;
            cursor: pointer;
            margin-bottom: 18px;
            transition: all 0.2s;
        }
        .download-btn:hover { opacity: 0.9; }
        .gallery-title {
            font-size: 13px;
            color: #666688;
            margin-bottom: 12px;
        }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .gallery-img {
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid #2a2a40;
        }
        .footer {
            text-align: center;
            color: #333355;
            padding: 30px 20px;
            font-size: 13px;
            line-height: 2;
        }
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>

<div class="header">
    <div class="logo">🎬 DramaGen AI</div>
    <div class="tagline">
        Turn any drama story into a Pixar-style cartoon video — Free, forever.
    </div>
    <div class="badges">
        <span class="badge">✨ No Signup</span>
        <span class="badge">🎨 4 Art Styles</span>
        <span class="badge">⚡ Powered by Llama 3</span>
        <span class="badge">📥 Free Download</span>
        <span class="badge">🎭 Drama and Romance</span>
    </div>
</div>

<div class="container">
    <div class="main-grid">

        <div class="panel">
            <div class="panel-title">✍️ Your Story</div>

            <label>Paste or type your drama story</label>
            <textarea id="story" placeholder="Example: Maya had been sitting by the same coffee shop window for three years. She never told anyone she was waiting for Daniel. On a rainy Thursday morning a letter arrived with no return address. Her hands trembled as she opened it. It was from Daniel saying he was sorry and never stopped loving her. She looked up and saw him standing in the rain outside the window."></textarea>

            <div class="row-2">
                <div>
                    <label>Main Character Name</label>
                    <input type="text" id="charName" value="Maya">
                </div>
                <div>
                    <label>Gender</label>
                    <select id="charGender">
                        <option value="female">Female</option>
                        <option value="male">Male</option>
                    </select>
                </div>
            </div>

            <label>Character Appearance</label>
            <input type="text" id="charLook"
                value="dark-skinned African woman, natural afro hair, warm brown eyes, elegant and slim">

            <label>Animation Style</label>
            <div class="style-grid">
                <div class="style-btn active" data-style="pixar" onclick="selectStyle(this)">
                    🎠 Pixar / Disney 3D
                </div>
                <div class="style-btn" data-style="anime" onclick="selectStyle(this)">
                    🌸 Anime
                </div>
                <div class="style-btn" data-style="comic" onclick="selectStyle(this)">
                    💥 Comic Book
                </div>
                <div class="style-btn" data-style="watercolor" onclick="selectStyle(this)">
                    🎨 Watercolor
                </div>
            </div>

            <div class="slider-group">
                <div>
                    <div class="slider-label">
                        <span>Number of Scenes</span>
                        <span class="slider-val" id="scenesVal">6</span>
                    </div>
                    <input type="range" id="numScenes" min="3" max="8" value="6"
                        oninput="document.getElementById('scenesVal').textContent=this.value">
                </div>
                <div>
                    <div class="slider-label">
                        <span>Seconds per Scene</span>
                        <span class="slider-val" id="secsVal">10</span>
                    </div>
                    <input type="range" id="secPerScene" min="5" max="15" value="10"
                        oninput="document.getElementById('secsVal').textContent=this.value">
                </div>
            </div>

            <button class="generate-btn" id="generateBtn" onclick="generateVideo()">
                🎬 Generate My Video
            </button>

            <div class="tips">
                💡 <strong style="color:#aaaacc">Tips:</strong><br>
                • Write at least 3 to 4 sentences<br>
                • 6 scenes x 10 seconds = 1 minute video<br>
                • Generation takes about 5 to 15 minutes<br>
                • After download add voice in CapCut free<br>
                • Add music using VEED.io free
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">🎥 Your Video</div>

            <div class="status-bar" id="statusBar">
                Your video will appear here after generation completes.
            </div>

            <div class="debug-box" id="debugBox">Ready. Click Generate to start.</div>

            <div class="progress-wrap" id="progressWrap">
                <div class="progress-label" id="progressLabel">Starting...</div>
                <div class="progress-bar-outer">
                    <div class="progress-bar-inner" id="progressBar"></div>
                </div>
            </div>

            <div class="video-wrap" id="videoWrap">
                <div class="video-placeholder">
                    <div class="icon">🎬</div>
                    <p>Your cartoon video will appear here</p>
                </div>
            </div>

            <button class="download-btn" id="downloadBtn" onclick="downloadVideo()">
                📥 Download Video MP4
            </button>

            <div class="gallery-title" id="galleryTitle"></div>
            <div class="gallery-grid" id="galleryGrid"></div>
        </div>

    </div>
</div>

<div class="footer">
    🎬 DramaGen AI — Turn any story into a cartoon drama video for free<br>
    Perfect for True Tales Chronicles and YouTube drama channels<br>
    No watermark. No signup. No limits.
</div>

<script>
    let selectedStyle = "pixar";
    let currentJobId = null;

    function log(msg) {
        const box = document.getElementById("debugBox");
        const time = new Date().toLocaleTimeString();
        box.textContent += "\\n[" + time + "] " + msg;
        box.scrollTop = box.scrollHeight;
    }

    function selectStyle(el) {
        document.querySelectorAll(".style-btn")
            .forEach(function(b) { b.classList.remove("active"); });
        el.classList.add("active");
        selectedStyle = el.dataset.style;
    }

    function setStatus(msg) {
        document.getElementById("statusBar").textContent = msg;
    }

    function setProgress(pct, label) {
        document.getElementById("progressWrap").style.display = "block";
        document.getElementById("progressBar").style.width = pct + "%";
        document.getElementById("progressLabel").textContent = label;
    }

    function downloadVideo() {
        if (!currentJobId) return;
        const a = document.createElement("a");
        a.href = "/download/" + currentJobId;
        a.download = "DramaGen_AI_Video.mp4";
        a.click();
    }

    function showGallery(images, scenes) {
        const grid = document.getElementById("galleryGrid");
        const title = document.getElementById("galleryTitle");
        grid.innerHTML = "";
        title.textContent = images.length + " Scenes Generated";
        images.forEach(function(src, i) {
            const img = document.createElement("img");
            img.src = src;
            img.className = "gallery-img";
            grid.appendChild(img);
        });
    }

    async function generateVideo() {
        const story = document.getElementById("story").value.trim();
        const charName = document.getElementById("charName").value.trim() || "Maya";
        const charLook = document.getElementById("charLook").value.trim();
        const numScenes = parseInt(document.getElementById("numScenes").value);
        const secPerScene = parseInt(document.getElementById("secPerScene").value);

        if (!story || story.length < 20) {
            setStatus("Please enter a longer story.");
            return;
        }

        const btn = document.getElementById("generateBtn");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Generating...';

        document.getElementById("videoWrap").innerHTML =
            '<div class="video-placeholder"><div class="icon">⏳</div><p>Generating...</p></div>';
        document.getElementById("downloadBtn").style.display = "none";
        document.getElementById("galleryGrid").innerHTML = "";
        document.getElementById("galleryTitle").textContent = "";

        log("Starting generation...");
        log("Scenes: " + numScenes + " | Style: " + selectedStyle);
        setProgress(5, "Sending to AI...");
        setStatus("Generating. This takes 5 to 15 minutes. Please wait...");

        let fakeProgress = 5;
        const interval = setInterval(function() {
            if (fakeProgress < 85) {
                fakeProgress += Math.random() * 2;
                const labels = [
                    "Splitting story into scenes...",
                    "Generating cartoon images...",
                    "Applying art style...",
                    "Building video...",
                    "Almost done..."
                ];
                const idx = Math.min(Math.floor(fakeProgress / 20), labels.length - 1);
                setProgress(Math.min(fakeProgress, 85), labels[idx]);
            }
        }, 5000);

        try {
            log("Calling /generate...");

            const response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    story: story,
                    char_name: charName,
                    char_look: charLook,
                    style: selectedStyle,
                    num_scenes: numScenes,
                    sec_per_scene: secPerScene
                })
            });

            clearInterval(interval);
            log("Response: " + response.status);

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || "Server error " + response.status);
            }

            const data = await response.json();
            log("Done! Job: " + data.job_id);
            log("Scenes: " + data.total_scenes);
            log("Duration: " + data.total_duration + "s");

            currentJobId = data.job_id;
            setProgress(100, "Done!");

            const mins = Math.floor(data.total_duration / 60);
            const secs = data.total_duration % 60;
            setStatus("Done! " + data.total_scenes + " scenes | " + mins + "m " + secs + "s");

            document.getElementById("videoWrap").innerHTML =
                '<video controls autoplay loop src="/download/' + currentJobId + '"></video>';
            document.getElementById("downloadBtn").style.display = "block";

            if (data.scene_images && data.scene_images.length > 0) {
                showGallery(data.scene_images, data.scenes);
            }

        } catch (error) {
            clearInterval(interval);
            log("ERROR: " + error.message);
            setStatus("Error: " + error.message);
            setProgress(0, "Failed");
            document.getElementById("videoWrap").innerHTML =
                '<div class="video-placeholder"><div class="icon">❌</div><p>' +
                error.message + '</p></div>';
        } finally {
            btn.disabled = false;
            btn.innerHTML = "🎬 Generate My Video";
        }
    }
</script>
</body>
</html>
"""

# ================================================
# SERVE THE WEBSITE
# ================================================

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "ok",
        "groq_key_set": bool(GROQ_API_KEY),
        "hf_key_set": bool(HF_API_KEY)
    })

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
    if GROQ_API_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            prompt = (
                "You are a storyboard artist. "
                "Split this drama story into exactly " + str(num_scenes) + " visual scenes. "
                "Each scene is a vivid image description with setting, emotion, action and lighting. "
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
                    return scenes[:num_scenes]
        except Exception as e:
            print("Groq error: " + str(e))

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
            print("Attempt " + str(attempt + 1))
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
            print("Status: " + str(resp.status_code))
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content))
                print("Image OK")
                return img
            elif resp.status_code == 503:
                wait = 30 + (attempt * 10)
                print("Loading. Wait " + str(wait) + "s")
                time.sleep(wait)
            elif resp.status_code == 429:
                print("Rate limited. Wait 30s")
                time.sleep(30)
            else:
                print("Error: " + str(resp.status_code))
                break
        except Exception as e:
            print("Error: " + str(e))
            time.sleep(10)

    return Image.new("RGB", (768, 432), color=(20, 20, 40))

# ================================================
# VIDEO BUILDER
# ================================================

def build_video(images, seconds_per_scene, job_id):
    path = os.path.join(TEMP_DIR, job_id + ".mp4")
    fps = 24
    spf = fps * seconds_per_scene
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p")
    for i, img in enumerate(images):
        frame = np.array(img.resize((768, 432)))
        for _ in range(spf):
            writer.append_data(frame)
        print("Scene " + str(i + 1) + " written")
    writer.close()
    print("Video saved")
    return path

# ================================================
# GENERATE ENDPOINT
# ================================================

@app.route("/generate", methods=["POST"])
def generate():
    print("\n=== GENERATE ===")
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No data"}), 400

        story         = str(data.get("story", "")).strip()
        char_name     = str(data.get("char_name", "Maya")).strip()
        char_look     = str(data.get("char_look", "cartoon character")).strip()
        style_key     = str(data.get("style", "pixar"))
        num_scenes    = min(8, max(1, int(data.get("num_scenes", 6))))
        sec_per_scene = min(15, max(5, int(data.get("sec_per_scene", 10))))

        if len(story) < 20:
            return jsonify({"error": "Story too short"}), 400

        job_id = str(uuid.uuid4())[:8]
        print("Job: " + job_id)

        scenes = split_story(story, num_scenes)
        print("Scenes: " + str(len(scenes)))

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
        print("Done! " + str(total) + "s")

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
