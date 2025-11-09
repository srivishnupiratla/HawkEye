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
MODEL_NAME = "gemma3:4b"
HOST = "0.0.0.0"
PORT = 8000
KNOWN_FACES_DIR = "known_faces"
FRAME_PROCESS_INTERVAL = 2.0

# Trigger Configuration
TARGET_OBJECTS = ["apple", "cell phone", "weapon", "bottle", "falling"]
SECONDARY_PROCESS_URL = "http://localhost:9000/object"

# Speech-to-Text API Configuration
SPEECH_API_URL = "http://localhost:9000/query"  # Update with your API endpoint
# ---------------------

SYSTEM_PROMPT = """
You are a detailed video feed analyst named 'MemoryLink'.
Your job is to carefully observe and describe what you see in the video feed.

ALWAYS provide a detailed description that includes:
- What objects, people, or scenes are visible
- Any actions or movements you can detect
- The setting or environment
- Any notable details or changes

Be conversational and descriptive, as if you're explaining what you see to someone who can't see the screen.
Keep responses concise but informative (2-4 sentences typically).

If the user asks a specific question, answer it while still providing context about what you see.
"""

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
    """Recognize faces in a base64-encoded image."""
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

# --- [NEW] Speech-to-Text Handler ---
async def handle_speech_prompt(prompt_text: str, image_b64: str = None) -> dict:
    """
    Handles speech-to-text prompts with two behaviors:
    1. If prompt contains "who is this", trigger facial recognition
    2. Otherwise, send to speech API endpoint
    
    Returns dict with 'type' and relevant data
    """
    prompt_lower = prompt_text.lower().strip()
    
    # Check if this is a "who is this" query
    if "who is this" in prompt_lower:
        print(f"🎤 Speech prompt: '{prompt_text}' -> FACIAL RECOGNITION")
        
        if not image_b64:
            return {
                "type": "facial_recognition",
                "error": "No image provided for facial recognition"
            }
        
        face_results = recognize_faces_in_frame(image_b64)
        
        # Format response based on results
        if face_results["count"] > 0:
            names = [f["name"] for f in face_results["faces"]]
            recognized = [n for n in names if n != "Unknown"]
            
            if recognized:
                response_text = f"I can see {', '.join(recognized)}."
            else:
                response_text = f"I can see {face_results['count']} person(s), but I don't recognize them."
        else:
            response_text = "I don't see any faces in the current frame."
        
        return {
            "type": "facial_recognition",
            "face_results": face_results,
            "response_text": response_text
        }
    
    else:
        # Send to speech API endpoint
        print(f"🎤 Speech prompt: '{prompt_text}' -> SPEECH API")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    SPEECH_API_URL,
                    params={"text": prompt_text}
                )
                response.raise_for_status()               
                print(response.content)

                return {
                    "type": "speech_api",
                    "api_response": {"response": response.text.strip()},
                    "status_code": response.status_code
                }
            except httpx.RequestError as e:
                print(f"⚠️ Speech API request failed: {e}")
                print(f"    Continuing without speech API response...")
                return {
                    "type": "speech_api",
                    "api_response": {"response": "Speech API is currently unavailable."},
                    "error": str(e),
                    "status_code": 0
                }
            except Exception as e:
                print(f"⚠️ Unexpected error calling speech API: {e}")
                print(f"    Continuing without speech API response...")
                return {
                    "type": "speech_api",
                    "api_response": {"response": "Unable to process speech request."},
                    "error": str(e),
                    "status_code": 0
                }

# --- Forwarder Helper ---
async def forward_frame(image_b64: str, detected_object: str):
    """Sends a SINGLE detected object frame to secondary process."""
    print(f"--> FORWARDING single trigger: '{detected_object}'...")
    
    payload = {
        "object": detected_object,
        "image": image_b64
    }
    
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            response = await client.post(SECONDARY_PROCESS_URL, json=payload)
            print(f"    ✓ Secondary responded for '{detected_object}': {response.status_code}")
        except httpx.TimeoutException:
            print(f"    ⚠️ Timeout forwarding '{detected_object}' (secondary taking too long)")
        except httpx.RequestError as e:
            print(f"    ⚠️ Forwarding failed for '{detected_object}': {e}")
            print(f"    Continuing without secondary process...")
        except Exception as e:
            print(f"    ⚠️ Unexpected error forwarding '{detected_object}': {e}")
            print(f"    Continuing without secondary process...")

async def query_ollama(image_b64: str, prompt: str) -> str:
    """Sends the image and prompt to Ollama and returns the textual response."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME, "prompt": prompt, "system": SYSTEM_PROMPT,
                    "images": [image_b64], "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("response", "No response from model.")
        except httpx.TimeoutException:
            print(f"⚠️ Ollama request timeout")
            return "I'm having trouble processing the image right now (timeout)."
        except httpx.RequestError as e:
            print(f"⚠️ Ollama request error: {e}")
            return "I'm having trouble connecting to the vision model."
        except Exception as e:
            print(f"⚠️ Unexpected Ollama error: {e}")
            return "An unexpected error occurred while analyzing the image."

async def process_single_frame(websocket: WebSocket, image_data: str, message: dict):
    user_prompt = message.get("prompt")
    recognize_faces = message.get("recognize_faces", False)
    is_speech = message.get("is_speech", False)  # NEW: flag to indicate speech input

    # [NEW] Handle speech-to-text prompts differently
    if is_speech and user_prompt:
        speech_result = await handle_speech_prompt(user_prompt, image_data)
        
        if speech_result["type"] == "facial_recognition":
            # Return facial recognition results
            try:
                await websocket.send_json({
                    "response": speech_result.get("response_text", ""),
                    "faces": speech_result.get("face_results"),
                    "speech_handled": True,
                    "should_speak": True,  # Signal to speak this response
                    "triggers": []
                })
            except:
                print("Failed to send speech response to client")
            return
        
        elif speech_result["type"] == "speech_api":
            # Return speech API results
            api_response = speech_result.get("api_response", {})
            response_text = api_response.get("response", "No response") if isinstance(api_response, dict) else str(api_response)
            
            try:
                await websocket.send_json({
                    "response": response_text,
                    "speech_api_result": api_response,
                    "speech_handled": True,
                    "should_speak": True,  # Signal to speak this response
                    "triggers": []
                })
            except:
                print("Failed to send speech API response to client")
            return

    # Original processing logic for non-speech prompts
    # 1. Face Recognition
    face_results = None
    if recognize_faces:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Face ID...")
        face_results = recognize_faces_in_frame(image_data)

    # 2. Determine Prompt
    final_prompt = "Describe what you see in this frame. What's happening?"
    if user_prompt and user_prompt != "Analyze this frame according to the system instructions.":
        print(f"🎤 User prompt: {user_prompt}")
        final_prompt = user_prompt

    # 3. Add Face Context
    if face_results and face_results["count"] > 0:
         detected_names = [f["name"] for f in face_results["faces"] if f["name"] != "Unknown"]
         if detected_names:
             final_prompt += f"\n\nNote: I've detected: {', '.join(detected_names)}."

    # 4. Full VLM Analysis
    print(f"🔍 Running full analysis...")
    analysis = await query_ollama(image_data, final_prompt)
    analysis_clean = analysis.strip()
    print(f"📝 Analysis: {analysis_clean[:100]}...")

    # 5. Word Search Trigger Loop
    analysis_lower = analysis_clean.lower()
    triggers = [obj for obj in TARGET_OBJECTS if obj.lower() in analysis_lower]

    if triggers:
        print(f"🚨 TRIGGER MATCHES: {triggers}")
        for trigger in triggers:
             asyncio.create_task(forward_frame(image_data, trigger))

    # 6. Send Response to Client
    try:
        await websocket.send_json({
            "response": analysis_clean,
            "faces": face_results,
            "triggers": triggers
        })
    except:
        print("Failed to send response to client")

@app.get("/")
async def get_client():
    return FileResponse("client.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    last_process_time = 0
    latest_frame = None
    latest_message = None
    processing_lock = asyncio.Lock()

    async def scanner_loop():
        nonlocal last_process_time, latest_frame, latest_message
        while True:
            await asyncio.sleep(0.1)
            if latest_frame and (time.time() - last_process_time >= FRAME_PROCESS_INTERVAL):
                if not processing_lock.locked():
                    async with processing_lock:
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
