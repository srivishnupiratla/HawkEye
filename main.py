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

SYSTEM_PROMPT = """
You are a focused video feed analyst. 
Your job is to watch this feed specifically for: DOORS.
If you see the user opening a door, take note.
If they do not turn around and close the door, alert them to this fact.
If no doors are present, dont respond to anything else, just say "clear".
Keep your responses concise and direct.
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

    Returns:
        dict with 'faces' (list of recognized names and locations) and 'count' (number of faces)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        faces_found = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            faces_found.append({
                "name": name,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "confidence": float(1 - face_distances[best_match_index]) if known_face_encodings and matches[
                    best_match_index] else 0.0
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # 1. Receive data from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                image_data = message.get("image")
                prompt = message.get("prompt", "Analyze this frame according to the system instructions.")
                recognize_faces = message.get("recognize_faces", False)  # Optional flag

                if not image_data:
                    await websocket.send_json({"error": "No image data received"})
                    continue

                # Handle standard base64 data URI header if present
                if "," in image_data:
                    image_data = image_data.split(",")[1]

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing frame...")

                # Run both analyses concurrently
                tasks = [query_ollama(image_data, prompt)]

                if recognize_faces:
                    # Run face recognition
                    face_results = recognize_faces_in_frame(image_data)
                else:
                    face_results = None

                # Wait for Ollama analysis
                analysis = await tasks[0]
                analysis = analysis.strip()

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis: {analysis}")
                if face_results and face_results["count"] > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Faces detected: {face_results['count']}")
                    for face in face_results["faces"]:
                        print(f"  - {face['name']} (confidence: {face['confidence']:.2f})")

                # 3. Send result back to client
                response_data = {
                    "response": analysis,
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