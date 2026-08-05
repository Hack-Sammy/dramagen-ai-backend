from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import re
import json
import uuid
import requests
import io
import base64
import time
from PIL import Image
from groq import Groq

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_API_KEY   = os.environ.get("HF_API_KEY", "")
HF_API_URL   = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

print("GROQ: " + str(bool(GROQ_API_KEY)))
print("HF:   " + str(bool(HF_API_KEY)))

STYLES = {
    "pixar": "pixar style, disney 3d animation, vibrant colors, expressive character, smooth 3d render, cinematic lighting, highly detailed",
    "anime": "anime style, studio ghibli, beautiful illustration, emotional, detailed, vibrant colors",
    "comic": "comic book style, bold outlines, dramatic colors, professional illustration",
    "watercolor": "watercolor illustration, soft painting, beautiful, emotional, storybook art"
}

NEGATIVE = "realistic, photograph, blurry, ugly, distorted, watermark, text, low quality"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DramaGen AI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a14;color:#fff;font-family:'Segoe UI',sans-serif;min-height:100vh}
.header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);padding:40px 20px;text-align:center}
.logo{font-size:48px;font-weight:900;background:linear-gradient(90deg,#f953c6,#7c4dff,#40c4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.tagline{color:#aaaacc;font-size:16px}
.badges{margin-top:12px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap}
.badge{background:rgba(124,77,255,0.2);border:1px solid #7c4dff;color:#b39dff;padding:3px 12px;border-radius:20px;font-size:11px}
.container{max-width:1100px;margin:0 auto;padding:24px 16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){.grid{grid-template-columns:1fr}.logo{font-size:32px}}
.panel{background:#13131f;border:1px solid #2a2a40;border-radius:18px;padding:24px}
.pt{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#7c4dff;margin-bottom:16px}
label{display:block;font-size:12px;color:#aaaacc;margin-bottom:5px;margin-top:14px}
textarea,input,select{width:100%;background:#1e1e30;border:1px solid #333355;border-radius:8px;color:#fff;padding:10px 12px;font-size:13px;font-family:inherit;resize:vertical}
textarea{min-height:140px}
textarea:focus,input:focus,select:focus{outline:none;border-color:#7c4dff}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.styles{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:5px}
.sb{background:#1e1e30;border:2px solid #333355;border-radius:8px;color:#aaaacc;padding:9px;cursor:pointer;font-size:12px;font-weight:600;text-align:center;transition:all 0.2s}
.sb:hover{border-color:#7c4dff;color:#fff}
.sb.active{border-color:#f953c6;background:rgba(249,83,198,0.15);color:#f953c6}
.sliders{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
.sl{font-size:11px;color:#aaaacc;display:flex;justify-content:space-between;margin-bottom:5px}
.sv{color:#f953c6;font-weight:700}
input[type=range]{accent-color:#7c4dff;cursor:pointer}
.gbtn{width:100%;background:linear-gradient(135deg,#f953c6,#7c4dff);border:none;border-radius:12px;color:#fff;font-size:17px;font-weight:800;padding:16px;cursor:pointer;margin-top:20px;transition:all 0.3s;box-shadow:0 8px 30px rgba(124,77,255,0.4)}
.gbtn:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(124,77,255,0.6)}
.gbtn:disabled{opacity:0.6;cursor:not-allowed;transform:none}
.tips{background:#1a1a2e;border:1px solid #2a2a40;border-radius:10px;padding:14px;margin-top:16px;font-size:11px;color:#666688;line-height:1.9}
.status{background:#1e1e30;border:1px solid #333355;border-radius:8px;padding:12px;font-size:12px;color:#aaaacc;margin-bottom:14px;min-height:42px}
.prog{display:none;margin-bottom:14px}
.pl{font-size:11px;color:#aaaacc;margin-bottom:6px}
.po{background:#1e1e30;border-radius:20px;height:6px;overflow:hidden}
.pi{height:100%;background:linear-gradient(90deg,#f953c6,#7c4dff);border-radius:20px;width:0%;transition:width 0.5s ease}
.gallery{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:14px}
.gallery img{width:100%;border-radius:8px;border:1px solid #2a2a40;cursor:pointer}
.gallery img:hover{border-color:#7c4dff}
.dbtn{display:none;width:100%;background:linear-gradient(135deg,#00c853,#00796b);border:none;border-radius:10px;color:#fff;font-size:15px;font-weight:700;padding:13px;cursor:pointer;margin-top:14px}
.dbtn:hover{opacity:0.9}
.vwrap{background:#1e1e30;border:1px solid #333355;border-radius:12px;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;margin-bottom:12px;overflow:hidden}
.vph{text-align:center;color:#444466}
.vph .ic{font-size:42px}
.vph p{margin-top:6px;font-size:13px}
video{width:100%;height:100%;object-fit:contain}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;color:#333355;padding:24px;font-size:12px;line-height:2}
</style>
</head>
<body>
<div class="header">
  <div class="logo">🎬 DramaGen AI</div>
  <div class="tagline">Turn any drama story into a Pixar-style cartoon video — Free, forever.</div>
  <div class="badges">
    <span class="badge">✨ No Signup</span>
    <span class="badge">🎨 4 Art Styles</span>
    <span class="badge">⚡ Llama 3 Powered</span>
    <span class="badge">📥 Free Download</span>
    <span class="badge">🎭 Drama and Romance</span>
  </div>
</div>
<div class="container">
  <div class="grid">
    <div class="panel">
      <div class="pt">✍️ Your Story</div>
      <label>Paste your drama story here</label>
      <textarea id="story" placeholder="Example: Maya had been sitting by the same coffee shop window for three years. She never told anyone she was waiting for Daniel. On a rainy Thursday morning a letter arrived with no return address. Her hands trembled as she opened it. It was from Daniel saying he was sorry and never stopped loving her. She looked up and saw him standing in the rain outside."></textarea>
      <div class="row2">
        <div>
          <label>Character Name</label>
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
      <input type="text" id="charLook" value="dark-skinned African woman, natural afro hair, warm brown eyes, elegant and slim">
      <label>Art Style</label>
      <div class="styles">
        <div class="sb active" data-style="pixar" onclick="pickStyle(this)">🎠 Pixar / Disney</div>
        <div class="sb" data-style="anime" onclick="pickStyle(this)">🌸 Anime</div>
        <div class="sb" data-style="comic" onclick="pickStyle(this)">💥 Comic Book</div>
        <div class="sb" data-style="watercolor" onclick="pickStyle(this)">🎨 Watercolor</div>
      </div>
      <div class="sliders">
        <div>
          <div class="sl"><span>Scenes</span><span class="sv" id="sv1">6</span></div>
          <input type="range" id="numScenes" min="3" max="8" value="6" oninput="document.getElementById('sv1').textContent=this.value">
        </div>
        <div>
          <div class="sl"><span>Secs per Scene</span><span class="sv" id="sv2">10</span></div>
          <input type="range" id="secScene" min="5" max="15" value="10" oninput="document.getElementById('sv2').textContent=this.value">
        </div>
      </div>
      <button class="gbtn" id="gbtn" onclick="generate()">🎬 Generate My Video</button>
      <div class="tips">
        💡 <strong style="color:#aaaacc">Tips:</strong><br>
        • Write at least 3 to 4 clear sentences<br>
        • 6 scenes × 10 seconds = 1 minute video<br>
        • Generation takes 5 to 15 minutes<br>
        • Add voice in CapCut after downloading<br>
        • Add music in VEED.io after downloading
      </div>
    </div>
    <div class="panel">
      <div class="pt">🎥 Your Video</div>
      <div class="status" id="status">Your video will appear here after generation.</div>
      <div class="prog" id="prog">
        <div class="pl" id="pl">Starting...</div>
        <div class="po"><div class="pi" id="pi"></div></div>
      </div>
      <div class="vwrap" id="vwrap">
        <div class="vph"><div class="ic">🎬</div><p>Your cartoon video will appear here</p></div>
      </div>
      <button class="dbtn" id="dbtn" onclick="downloadVideo()">📥 Download Video MP4</button>
      <div class="gallery" id="gallery"></div>
    </div>
  </div>
</div>
<div class="footer">
  🎬 DramaGen AI — Turn any story into a cartoon drama video for free<br>
  Perfect for True Tales Chronicles and YouTube drama channels<br>
  No watermark. No signup. No limits.
</div>

<script>
let style = "pixar";
let generatedImages = [];
let secPerScene = 10;

function pickStyle(el) {
  document.querySelectorAll(".sb").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  style = el.dataset.style;
}

function setStatus(msg) { document.getElementById("status").textContent = msg; }

function setProgress(pct, label) {
  document.getElementById("prog").style.display = "block";
  document.getElementById("pi").style.width = pct + "%";
  document.getElementById("pl").textContent = label;
}

function downloadVideo() {
  if (generatedImages.length === 0) return;
  setStatus("Preparing download...");
  
  // Create a zip of all images for the user
  // They can use CapCut to make video from images
  generatedImages.forEach(function(src, i) {
    const a = document.createElement("a");
    a.href = src;
    a.download = "scene_" + String(i + 1).padStart(2, "0") + ".jpg";
    setTimeout(function() { a.click(); }, i * 500);
  });
  
  setStatus("All scene images downloaded! Import them into CapCut to make your video.");
}

async function generate() {
  const story = document.getElementById("story").value.trim();
  const charName = document.getElementById("charName").value.trim() || "Maya";
  const charLook = document.getElementById("charLook").value.trim();
  const numScenes = parseInt(document.getElementById("numScenes").value);
  secPerScene = parseInt(document.getElementById("secScene").value);

  if (!story || story.length < 20) {
    setStatus("Please enter a longer story.");
    return;
  }

  const btn = document.getElementById("gbtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating...';

  document.getElementById("vwrap").innerHTML =
    '<div class="vph"><div class="ic">⏳</div><p>Generating your scenes...</p></div>';
  document.getElementById("dbtn").style.display = "none";
  document.getElementById("gallery").innerHTML = "";
  generatedImages = [];

  setProgress(5, "Sending story to AI...");
  setStatus("Generating. This takes 5 to 15 minutes. Please wait...");

  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        story: story,
        char_name: charName,
        char_look: charLook,
        style: style,
        num_scenes: numScenes,
        sec_per_scene: secPerScene
      })
    });

    const text = await response.text();
    
    let data;
    try {
      data = JSON.parse(text);
    } catch(e) {
      console.log("Raw response:", text.substring(0, 500));
      throw new Error("Server returned invalid response. Check logs.");
    }

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Generation failed");
    }

    generatedImages = data.scene_images;
    setProgress(100, "Done!");

    const mins = Math.floor(data.total_duration / 60);
    const secs = data.total_duration % 60;
    setStatus("Done! " + data.total_scenes + " scenes | " + mins + "m " + secs + "s");

    // Show gallery
    const gallery = document.getElementById("gallery");
    gallery.innerHTML = "";
    data.scene_images.forEach(function(src, i) {
      const img = document.createElement("img");
      img.src = src;
      img.title = "Scene " + (i + 1) + ": " + (data.scenes[i] || "").substring(0, 60);
      gallery.appendChild(img);
    });

    // Show video placeholder with instructions
    document.getElementById("vwrap").innerHTML =
      '<div class="vph">' +
      '<div class="ic">✅</div>' +
      '<p style="color:#aaaacc;padding:20px">' +
      data.total_scenes + ' scenes generated!<br><br>' +
      'Click Download to save all scene images.<br>' +
      'Import into CapCut to assemble your video.' +
      '</p></div>';

    document.getElementById("dbtn").style.display = "block";
    document.getElementById("dbtn").textContent = "📥 Download All " + data.total_scenes + " Scene Images";

  } catch (error) {
    setStatus("Error: " + error.message);
    setProgress(0, "Failed");
    document.getElementById("vwrap").innerHTML =
      '<div class="vph"><div class="ic">❌</div><p>' + error.message + '</p></div>';
  } finally {
    btn.disabled = false;
    btn.innerHTML = "🎬 Generate My Video";
  }
}
</script>
</body>
</html>"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/test")
def test():
    return jsonify({
        "status": "ok",
        "groq": bool(GROQ_API_KEY),
        "hf": bool(HF_API_KEY)
    })

def split_story(story, num_scenes):
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt = (
                "Split this drama story into exactly " + str(num_scenes) + " visual scene descriptions. "
                "Each must describe setting, emotion, action, lighting vividly for an image AI. "
                "Story: " + story + " "
                "Return ONLY a JSON array of " + str(num_scenes) + " strings."
            )
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            text = res.choices[0].message.content.strip()
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                scenes = json.loads(match.group(0))
                if isinstance(scenes, list) and scenes:
                    return scenes[:num_scenes]
        except Exception as e:
            print("Groq error: " + str(e))

    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    sentences = [s for s in sentences if len(s) > 5]
    if not sentences:
        return ["A dramatic emotional scene"] * num_scenes
    size = max(1, len(sentences) // num_scenes)
    scenes = []
    for i in range(0, len(sentences), size):
        scenes.append(" ".join(sentences[i:i+size]))
        if len(scenes) == num_scenes:
            break
    while len(scenes) < num_scenes:
        scenes.append(scenes[-1])
    return scenes[:num_scenes]

def make_image(prompt, style_key):
    prefix = STYLES.get(style_key, STYLES["pixar"])
    full = prefix + ", " + prompt
    headers = {"Authorization": "Bearer " + HF_API_KEY}
    payload = {
        "inputs": full,
        "parameters": {
            "negative_prompt": NEGATIVE,
            "num_inference_steps": 8,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 288
        }
    }
    for attempt in range(3):
        try:
            print("Attempt " + str(attempt+1) + ": " + prompt[:50])
            r = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=25
            )
            print("HTTP " + str(r.status_code))
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                print("OK: " + str(img.size))
                return img
            elif r.status_code == 503:
                print("Model loading. Wait 15s")
                time.sleep(15)
            elif r.status_code == 429:
                print("Rate limit. Wait 15s")
                time.sleep(15)
            else:
                print("Error: " + r.text[:100])
                return Image.new("RGB", (512, 288), (20, 20, 40))
        except Exception as e:
            print("Err: " + str(e))
            return Image.new("RGB", (512, 288), (20, 20, 40))
    return Image.new("RGB", (512, 288), (20, 20, 40))

@app.route("/generate", methods=["POST"])
def generate():
    print("\n=== GENERATE ===")
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No data"}), 400

        story = str(data.get("story", "")).strip()
        char_name = str(data.get("char_name", "Maya")).strip()
        char_look = str(data.get("char_look", "cartoon character")).strip()
        style_key = str(data.get("style", "pixar"))
        num_scenes = min(3, max(1, int(data.get("num_scenes", 3))))
        sec_per_scene = min(15, max(5, int(data.get("sec_per_scene", 10))))

        if len(story) < 20:
            return jsonify({"error": "Story too short"}), 400

        job_id = str(uuid.uuid4())[:8]
        print("Job: " + job_id)

        scenes = split_story(story, num_scenes)
        print("Scenes: " + str(len(scenes)))

        images_b64 = []
        char_desc = char_name + " who is " + char_look

        for i, scene in enumerate(scenes):
            print("Image " + str(i+1) + "/" + str(len(scenes)))
            prompt = scene + ", main character " + char_desc
            img = make_image(prompt, style_key)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            images_b64.append("data:image/jpeg;base64," + b64)
            print("Image " + str(i+1) + " done")

        total = len(scenes) * sec_per_scene
        print("All done. Total: " + str(total) + "s")

        return jsonify({
            "success": True,
            "job_id": job_id,
            "scenes": scenes,
            "scene_images": images_b64,
            "total_duration": total,
            "total_scenes": len(scenes)
        })

    except Exception as e:
        print("ERROR: " + str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
