# /Users/rishi/Desktop/p2/app.py
import os
import io
import json
import hashlib
import secrets
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.utils import load_img, img_to_array

MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model.keras")
model = load_model(MODEL_PATH)

input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
h = input_shape[1] or 224
w = input_shape[2] or 224

num_classes = model.output_shape[-1]
class_names_env = os.environ.get("CLASS_NAMES")
if class_names_env:
    try:
        CLASS_NAMES = json.loads(class_names_env)
    except:
        CLASS_NAMES = [s.strip() for s in class_names_env.split(",")]
else:
    if num_classes == 4:
        CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]
    else:
        CLASS_NAMES = [f"class_{i}" for i in range(num_classes)]

INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Plant Pathology Detector</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font-family:system-ui,sans-serif;margin:2rem}header{margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center}.card{border:1px solid #ddd;border-radius:8px;padding:1rem;margin-top:1rem}.row{display:flex;gap:1rem;flex-wrap:wrap}.preview{max-width:320px;border:1px solid #eee;border-radius:8px}.bar{height:10px;background:#4CAF50;border-radius:4px}.bar-wrap{background:#eee;border-radius:4px;width:240px}.result{display:flex;align-items:center;gap:.5rem;margin:.25rem 0}button{padding:.5rem 1rem}.muted{color:#666;font-size:.9rem}a{color:#06c;text-decoration:none}</style></head><body><header><h1>Plant Pathology Detector</h1><a href="/logout">Logout</a></header><p class="muted">Upload a leaf image to classify diseases or confirm healthy tissue.</p><div class="card"><form id="form"><input type="file" id="file" accept="image/*" required><button type="submit">Analyze</button></form><div class="row"><img id="preview" class="preview" alt="Preview"><div id="output"></div></div></div><script>const f=document.getElementById('file'),p=document.getElementById('preview'),o=document.getElementById('output');f.addEventListener('change',()=>{const g=f.files[0];if(g)p.src=URL.createObjectURL(g)});document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();o.innerHTML='Processing...';const g=f.files[0];if(!g)return;const d=new FormData();d.append('file',g);try{const r=await fetch('/predict',{method:'POST',body:d});const j=await r.json();render(j)}catch(err){o.textContent='Error: '+err}});function render(x){const t=x.predictions||[],a=x.all||[],m=x.multi_label||[];o.innerHTML=`<h3>Top Predictions</h3>${t.map(y=>item(y)).join('')}<h3>All Scores</h3>${a.map(y=>item(y)).join('')}<h3>Multi-Label ≥ 0.5</h3>${m.length?m.map(y=>item(y)).join(''):'<div class="muted">None above threshold</div>'}`;}function item(y){const q=Math.round(y.score*100);return `<div class="result"><div style="width:120px">${y.label}</div><div class="bar-wrap"><div class="bar" style="width:${q}%"></div></div><div>${q}%</div></div>`}</script></body></html>"""

LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Login</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font-family:system-ui,sans-serif;margin:2rem}form{display:flex;flex-direction:column;gap:.75rem;max-width:320px}.card{border:1px solid #ddd;border-radius:8px;padding:1rem}input,button{padding:.5rem}button{cursor:pointer}a{color:#06c;text-decoration:none}</style></head><body><h1>Login</h1><div class="card"><form method="post" action="/login"><input name="username" placeholder="Username" required><input name="password" type="password" placeholder="Password" required><button type="submit">Login</button></form></div><p>No account? <a href="/signup">Sign up</a></p></body></html>"""

SIGNUP_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sign Up</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font-family:system-ui,sans-serif;margin:2rem}form{display:flex;flex-direction:column;gap:.75rem;max-width:320px}.card{border:1px solid #ddd;border-radius:8px;padding:1rem}input,button{padding:.5rem}button{cursor:pointer}a{color:#06c;text-decoration:none}</style></head><body><h1>Sign Up</h1><div class="card"><form method="post" action="/signup"><input name="username" placeholder="Username" required><input name="password" type="password" placeholder="Password" required><button type="submit">Create Account</button></form></div><p>Have an account? <a href="/login">Login</a></p></body></html>"""

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

USERS = {}
SESSIONS = {}

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def user_from_request(request: Request):
    t = request.cookies.get("session")
    return SESSIONS.get(t)

def preprocess_image(file_like):
    image = load_img(file_like, target_size=(h, w))
    arr = img_to_array(image)
    arr = arr / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/login", response_class=HTMLResponse)
def login_page():
    path = os.path.join(static_dir, "login.html")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return LOGIN_HTML

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username in USERS and USERS[username] == hash_pw(password):
        token = secrets.token_urlsafe(32)
        SESSIONS[token] = username
        r = RedirectResponse("/", status_code=302)
        r.set_cookie("session", token, httponly=True)
        return r
    return HTMLResponse(content="Invalid credentials", status_code=401)

@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    path = os.path.join(static_dir, "signup.html")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return SIGNUP_HTML

@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...)):
    if username in USERS:
        return HTMLResponse(content="User exists", status_code=400)
    USERS[username] = hash_pw(password)
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = username
    r = RedirectResponse("/", status_code=302)
    r.set_cookie("session", token, httponly=True)
    return r

@app.get("/logout")
def logout(request: Request):
    t = request.cookies.get("session")
    if t in SESSIONS:
        del SESSIONS[t]
    r = RedirectResponse("/login", status_code=302)
    r.delete_cookie("session")
    return r

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    if not user_from_request(request):
        return RedirectResponse("/login", status_code=302)
    path = os.path.join(static_dir, "index.html")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return INDEX_HTML

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if not user_from_request(request):
        return RedirectResponse("/login", status_code=302)
    try:
        contents = await file.read()
    except:
        raise HTTPException(status_code=400, detail="Invalid file")
    arr = preprocess_image(io.BytesIO(contents))
    preds = model.predict(arr)
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    probs = preds[0].tolist()
    top_indices = np.argsort(probs)[::-1][:3]
    predictions = [{"label": CLASS_NAMES[i], "score": float(probs[i])} for i in top_indices]
    multi = [{"label": CLASS_NAMES[i], "score": float(probs[i])} for i in range(len(probs)) if probs[i] >= 0.5]
    all_scores = [{"label": CLASS_NAMES[i], "score": float(probs[i])} for i in range(len(probs))]
    return {"predictions": predictions, "all": all_scores, "multi_label": multi}