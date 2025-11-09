import json
import asyncio
import base64
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
from datetime import datetime
import face_recognition
import time

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2-vision"
HOST = "0.0.0.0"
PORT = 8000
KNOWN_FACES_DIR = "known_faces"
FRAME_PROCESS_INTERVAL = 2.0

# [NEW] Detection Configuration
# The set of objects we strictly watch for
TARGET_OBJECTS = ["apples"]
# The URL of the "other process" that will do deeper inference
SECONDARY_PROCESS_URL = "http://localhost:9000/deep-inference"
# ---------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Facial Recognition Storage ---
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load all known faces from the known_faces directory."""
    global known_face_encodings, known_face_names
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created {KNOWN_FACES_DIR} directory.")
        return

    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"Loaded face for: {name}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(known_face_names)} known faces")

def recognize_faces_in_frame(image_b64: str) -> dict:
    """Recognize faces using standard CV library (not VLM)."""
    try:
        image_bytes = base64.b64decode(image_b64)
        image_np = np.array(Image.open(io.BytesIO(image_bytes)))
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        faces_found = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            confidence = 0.0
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]

            faces_found.append({
                "name": name,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "confidence": float(confidence)
            })

        return {"count": len(faces_found), "faces": faces_found}
    except Exception as e:
        print(f"Face recognition error: {e}")
        return {"count": 0, "faces": [], "error": str(e)}

# --- [UPDATED] Simpler VLM Object Detector ---
async def detect_objects(image_b64: str) -> list:
    """
    Asks for a general description and checks if target words are in it.
    More robust than strict JSON schema for small models.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    # Very open-ended prompt to get maximum recall
                    "prompt": "List everything you can see in this image. Be detailed.",
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.0} 
                }
            )
            if response.status_code == 200:
                full_text = response.json().get("response", "").lower()
                print(f"Raw VLM Output: {full_text}") # ADDED PRINT STATEMENT
                # Simple string matching
                detected = [obj for obj in TARGET_OBJECTS if obj.lower() in full_text]
                return detected
        except Exception as e:
            print(f"Detection error: {e}")
    return []

# --- Forwarder to Secondary Process ---
async def forward_frame(image_b64: str, detected_items: list, metadata: dict = None):
    """
    Sends the frame to another service for deeper analysis.
    This is 'fire-and-forget' - we don't wait long for a response.
    """
    print(f"--> FORWARDING frame with {detected_items} to secondary process...")
    payload = {
        "timestamp": datetime.now().isoformat(),
        "triggers": detected_items,
        "image": image_b64,
        "metadata": metadata or {}
    }
    # Using a short timeout so we don't hang the main scanner if secondary is down
    async with httpx.AsyncClient(timeout=1.0) as client:
        try:
             response = await client.post(SECONDARY_PROCESS_URL, json=payload)
             print(f"    > Secondary process responded: {response.status_code}")
        except httpx.RequestError as e:
            print(f"Warning: Could not forward frame to secondary: {e}")

async def query_ollama_full(image_b64: str, prompt: str, system_prompt: str) -> str:
    """Standard full VLM analysis for user queries."""
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME, "prompt": prompt, "system": system_prompt,
                    "images": [image_b64], "stream": False
                }
            )
            return response.json().get("response", "No response.")
        except Exception as e:
            return f"Error: {e}"

@app.get("/")
async def get_client():
    return FileResponse("client.html")

async def process_single_frame(websocket: WebSocket, image_data: str, message: dict):
    user_prompt = message.get("prompt")
    recognize_faces = message.get("recognize_faces", False)

    # 1. Always run Face ID if requested (it's fast and local)
    face_results = None
    if recognize_faces:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Face ID...")
        face_results = recognize_faces_in_frame(image_data)
        if face_results["count"] > 0:
             print(f"   > Faces found: {[f['name'] for f in face_results['faces']]}")

    response_text = ""

    # 2. DECISION LOGIC
    if user_prompt and user_prompt != "Analyze this frame according to the system instructions.":
        # CASE A: User asked a specific question (override scanner)
        print(f"🎤 User Query: {user_prompt}")
        response_text = await query_ollama_full(image_data, user_prompt, "You are a helpful visual assistant.")

    else:
        # CASE B: Background Scanning Mode (Default)
        # Step 1: Scan strictly for target objects
        print(f"🔍 Scanning for targets: {TARGET_OBJECTS}...")
        detected = await detect_objects(image_data)

        if detected:
            # TRIGGER!
            print(f"🚨 MATCH FOUND: {detected}")
            response_text = f"ALERT: Detected {', '.join(detected)}"
            
            # Step 2: Forward to secondary process (non-blocking to main loop usually)
            asyncio.create_task(forward_frame(image_data, detected, metadata={"faces": face_results}))
        else:
            print("   > Scan clear.")
            response_text = "Scan clear. Monitoring..."

    # Send feedback to client
    try:
        await websocket.send_json({
            "response": response_text,
            "faces": face_results,
            "triggers": detected if 'detected' in locals() else []
        })
    except Exception as e:
        print(f"Client send error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    last_process_time = 0
    latest_frame = None
    latest_message = None
    processing_lock = asyncio.Lock()

    # Background scanner loop
    async def scanner_loop():
        nonlocal last_process_time, latest_frame, latest_message
        while True:
            await asyncio.sleep(0.1)
            if latest_frame and (time.time() - last_process_time >= FRAME_PROCESS_INTERVAL):
                if not processing_lock.locked():
                    async with processing_lock:
                        # Only auto-scan if user isn't currently asking a specific question
                        is_idle = not latest_message.get("prompt") or \
                                  latest_message.get("prompt") == "Analyze this frame according to the system instructions."
                        
                        if is_idle:
                            await process_single_frame(websocket, latest_frame, latest_message or {})
                            last_process_time = time.time()

    scanner_task = asyncio.create_task(scanner_loop())

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("image"):
                latest_frame = msg["image"].split(",")[1] if "," in msg["image"] else msg["image"]
                latest_message = msg

                # If it's an immediate user/face request, bypass scanner timer
                if msg.get("recognize_faces") or (msg.get("prompt") and msg["prompt"] != "Analyze this frame according to the system instructions."):
                     async with processing_lock:
                         await process_single_frame(websocket, latest_frame, msg)
                         last_process_time = time.time()

    except WebSocketDisconnect:
        print("Client disconnected")
        scanner_task.cancel()

@app.on_event("startup")
async def startup():
    load_known_faces()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)