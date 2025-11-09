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

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"
HOST = "0.0.0.0"
PORT = 8000
KNOWN_FACES_DIR = "known_faces"  # Directory to store known face images
# ---------------------

# --- UPDATED SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a focused video feed analyst named 'MemoryLink'.
Your primary job is to watch this feed specifically for DOORS.
- If you receive a generic or no user prompt, analyze the frame for an open door. If the user is seen leaving a door open, alert them concisely to this fact. If no door is present or the door is closed, respond with only the word "clear".
- If you receive a specific, non-generic prompt (like a question), ignore the door rule and answer the user's question directly based on the image.
- Keep your responses concise and direct.
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


# --- Replacement for recognize_faces_in_frame(image_b64: str) ---

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
        # print(f"DEBUG: Image received and converted to numpy array of shape {image_np.shape}") # Debug line

        # Find all face locations (DETECTION STEP)
        face_locations = face_recognition.face_locations(image_np)
        
        # --- NEW LOGGING ---
        if not face_locations:
            print("DEBUG: Face Detector FAILED. No face found in image.")
        # --- END NEW LOGGING ---

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
                
                # --- NEW LOGGING ---
                # This shows the actual closest known face and the distance score
                print(f"DEBUG: Found face. Closest Known Face: {known_face_names[best_match_index]} | Distance Score: {confidence_distance:.2f}")
                # --- END NEW LOGGING ---
                
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


# --- NEW: Serve the HTML file on the root route ---
@app.get("/")
async def get_client():
    # Assuming 'client.html' exists for testing purposes
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


# --- FIXED WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    DEFAULT_DOOR_PROMPT = "Analyze this frame according to the system instructions."

    try:
        while True:
            # 1. Receive data from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                image_data = message.get("image")
                user_prompt = message.get("prompt") 
                recognize_faces = message.get("recognize_faces", False)

                if not image_data:
                    await websocket.send_json({"error": "No image data received"})
                    continue

                if "," in image_data:
                    image_data = image_data.split(",")[1]

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing frame...")

                # --- FIX: Determine the final prompt and if LLM analysis is required ---
                final_prompt = None 
                ollama_required = True

                # 1. Handle specific requests (Geofence, Spoken Command)
                geofence_prompt = "The user has left their home geofence. Check if there are any potential dangers in the current frame."
                
                if user_prompt == geofence_prompt:
                    print("🚨 Geofence exit detected. Running security analysis.")
                    final_prompt = user_prompt 

                elif user_prompt and user_prompt != DEFAULT_DOOR_PROMPT:
                    print(f"🎤 User prompt received: {user_prompt}")
                    final_prompt = user_prompt 

                # 2. Handle default requests
                elif not recognize_faces:
                    # If it's a default request AND face recognition is NOT forced, run door analysis.
                    print("🖼️ Running default door analysis (Door Watch).")
                    final_prompt = DEFAULT_DOOR_PROMPT
                    
                # 3. Handle pure face recognition request (No LLM needed)
                else:
                    # If recognize_faces is True, but there's no unique prompt, skip Ollama.
                    print("👤 Face recognition only requested (Skipping LLM).")
                    ollama_required = False
                
                # --- END FIX ---

                # Run face recognition if requested (Always runs if recognize_faces is True)
                face_results = None
                if recognize_faces:
                    face_results = recognize_faces_in_frame(image_data)
                    
                    # --- DEBUGGING OUTPUT FOR FACE RECOGNITION ---
                    print(f"--- Face Recognition Result (Count: {face_results['count']}) ---")
                    if face_results and face_results["count"] > 0:
                        for face in face_results["faces"]:
                            print(f"  - Detected: {face['name']} (Conf: {face['confidence']:.2f}, Loc: {face['location']})")

                # Run LLM analysis only if required
                analysis = None
                if ollama_required:
                    tasks = [query_ollama(image_data, final_prompt)]
                    analysis = await tasks[0]
                    analysis = analysis.strip()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis: {analysis}")
                
                # Set a default response if we skipped the LLM
                response_text = analysis if analysis is not None else "clear"

                # 3. Send result back to client
                response_data = {
                    "response": response_text,
                    "faces": face_results
                }

                await websocket.send_json(response_data)

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})

    except WebSocketDisconnect:
        print("Client disconnected")


@app.on_event("startup")
async def startup_event():
    """Load known faces when the server starts."""
    print("Loading known faces...")
    load_known_faces()


if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}")
    print(f"Place known face images in '{KNOWN_FACES_DIR}/' directory")
    print("Name files as 'PersonName.jpg' (e.g., 'JohnDoe.jpg')")
    uvicorn.run(app, host=HOST, port=PORT)