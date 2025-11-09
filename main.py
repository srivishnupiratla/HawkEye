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
MODEL_NAME = "llama3.2"
HOST = "0.0.0.0"
PORT = 8000
KNOWN_FACES_DIR = "known_faces"  # Directory to store known face images
FRAME_PROCESS_INTERVAL = 2.0  # Process frame every 2 seconds
# ---------------------

# --- UPDATED SYSTEM PROMPT ---
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

# Allow CORS to make it easier to test with local HTML files
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
        print(f"Created {KNOWN_FACES_DIR} directory. Add images named as 'PersonName.jpg'")
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
                    # Use filename without extension as the person's name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"Loaded face for: {name}")
                else:
                    print(f"No face found in {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(known_face_names)} known faces")


def recognize_faces_in_frame(image_b64: str) -> dict:
    """
    Recognize faces in a base64-encoded image.
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Find all face locations (DETECTION STEP)
        face_locations = face_recognition.face_locations(image_np)
        
        if not face_locations:
            print("DEBUG: Face Detector FAILED. No face found in image.")

        # Find all face encodings (ENCODING STEP)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        faces_found = []
        image = Image.open(io.BytesIO(image_bytes))
        print("DEBUG:", image.mode, image.size)
        image.save("debug_frame.jpg")
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            confidence_distance = 1.0 

            # Use the known face with the smallest distance to the new face
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                confidence_distance = face_distances[best_match_index]
                
                print(f"DEBUG: Found face. Closest Known Face: {known_face_names[best_match_index]} | Distance Score: {confidence_distance:.2f}")
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            confidence_score = float(1.0 - confidence_distance)
            
            if not known_face_encodings:
                confidence_score = 0.0

            faces_found.append({
                "name": name,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "confidence": confidence_score
            })

        return {
            "count": len(faces_found),
            "faces": faces_found
        }

    except Exception as e:
        print(f"Face recognition error: {e}")
        return {
            "count": 0,
            "faces": [],
            "error": str(e)
        }


@app.get("/")
async def get_client():
    return FileResponse("client.html")


async def query_ollama(image_b64: str, prompt: str) -> str:
    """Sends the image and prompt to Ollama and returns the textual response."""
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "images": [image_b64],
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("response", "No response from model.")
        except httpx.RequestError as e:
            print(f"Ollama connection error: {e}")
            return f"Error communicating with Ollama: {e}"
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error: {e}"


async def process_single_frame(websocket: WebSocket, image_data: str, message: dict):
    """Process a single frame immediately (for button presses or voice commands)"""
    user_prompt = message.get("prompt")
    recognize_faces = message.get("recognize_faces", False)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ IMMEDIATE Processing (Face Recognition: {recognize_faces})...")
    
    # Determine the final prompt
    final_prompt = "Describe what you see in this frame. What's happening?"
    
    # Handle geofence exit detection
    geofence_prompt = "The user has left their home geofence. Check if there are any potential dangers in the current frame."
    
    if user_prompt == geofence_prompt:
        print("🚨 Geofence exit detected. Running security analysis.")
        final_prompt = user_prompt 
    elif user_prompt:
        print(f"🎤 User prompt: {user_prompt}")
        final_prompt = user_prompt
    else:
        print(f"🖼️ Running descriptive analysis")

    # Run face recognition if requested
    face_results = None
    if recognize_faces:
        face_results = recognize_faces_in_frame(image_data)
        
        print(f"--- Face Recognition Result (Count: {face_results['count']}) ---")
        if face_results and face_results["count"] > 0:
            for face in face_results["faces"]:
                print(f"  - Detected: {face['name']} (Conf: {face['confidence']:.2f})")
            
            # Enhance prompt with face recognition context
            detected_names = [f["name"] for f in face_results["faces"]]
            if any(name != "Unknown" for name in detected_names):
                final_prompt += f"\n\nNote: I've detected the following people in frame: {', '.join(detected_names)}. Include this information naturally in your description."

    # ALWAYS run LLM analysis for detailed descriptions
    analysis = await query_ollama(image_data, final_prompt)
    analysis = analysis.strip()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis: {analysis}")

    # Send result back to client
    response_data = {
        "response": analysis,
        "faces": face_results
    }

    try:
        await websocket.send_json(response_data)
    except:
        print("Failed to send response to client")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    DEFAULT_PROMPT = "Describe what you see in this frame. What's happening?"
    
    last_process_time = 0
    latest_frame = None
    latest_message = None
    processing_lock = asyncio.Lock()
    is_processing = False

    async def process_frame_periodically():
        """Background task that processes frames every FRAME_PROCESS_INTERVAL seconds"""
        nonlocal last_process_time, latest_frame, latest_message, is_processing
        
        # Initialize last_process_time to allow immediate first processing
        last_process_time = time.time() - FRAME_PROCESS_INTERVAL
        
        while True:
            try:
                await asyncio.sleep(0.1)  # Check frequently
                
                current_time = time.time()
                if (latest_frame is not None and 
                    current_time - last_process_time >= FRAME_PROCESS_INTERVAL and
                    not is_processing):
                    
                    async with processing_lock:
                        if latest_frame is None:
                            continue
                            
                        # Process the latest frame
                        image_data = latest_frame
                        message = latest_message or {}
                        
                        # Only process background frames if they don't have face recognition enabled
                        # (immediate face recognition requests are handled separately)
                        recognize_faces = message.get("recognize_faces", False)
                        
                        if not recognize_faces:
                            is_processing = True
                            await process_single_frame(websocket, image_data, message)
                            is_processing = False
                        
                        last_process_time = current_time
                        
            except Exception as e:
                print(f"Error in periodic processing: {e}")
                is_processing = False
                break

    # Start background processing task
    process_task = asyncio.create_task(process_frame_periodically())

    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                image_data = message.get("image")
                recognize_faces = message.get("recognize_faces", False)

                if not image_data:
                    await websocket.send_json({"error": "No image data received"})
                    continue

                if "," in image_data:
                    image_data = image_data.split(",")[1]

                # Store the latest frame and message for background processing
                async with processing_lock:
                    latest_frame = image_data
                    latest_message = message

                # If face recognition is requested, process IMMEDIATELY (don't wait for background task)
                if recognize_faces:
                    print("🔥 Face recognition requested - processing immediately!")
                    is_processing = True
                    await process_single_frame(websocket, image_data, message)
                    is_processing = False

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})

    except WebSocketDisconnect:
        print("Client disconnected")
        process_task.cancel()


@app.on_event("startup")
async def startup_event():
    """Load known faces when the server starts."""
    print("Loading known faces...")
    load_known_faces()


if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    print(f"Place known face images in '{KNOWN_FACES_DIR}/' directory")
    print("Name files as 'PersonName.jpg' (e.g., 'JohnDoe.jpg')")
    print(f"Frame processing interval: {FRAME_PROCESS_INTERVAL} seconds")
    uvicorn.run(app, host=HOST, port=PORT)