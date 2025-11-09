import json
import asyncio
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
from datetime import datetime

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"
HOST = "0.0.0.0"
PORT = 8000
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

# --- NEW: Serve the HTML file on the root route ---
@app.get("/")
async def get_client():
    # Assumes client.html is in the same directory
    return FileResponse("client.html")

async def query_ollama(image_b64: str, prompt: str) -> str:
    """Sends the image and prompt to Ollama and returns the textual response."""
    async with httpx.AsyncClient(timeout=None) as client:  # VLMs can be slow, disable timeout
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "images": [image_b64],
                    "stream": False  # For simplicity, we await the full response
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

                if not image_data:
                    await websocket.send_json({"error": "No image data received"})
                    continue

                # Handle standard base64 data URI header if present
                if "," in image_data:
                    image_data = image_data.split(",")[1]

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing frame...")
                analysis = await query_ollama(image_data, prompt)
                
                # Clean up response: Moondream might still be chatty, so we can 
                # do some basic string cleanup here if needed.
                analysis = analysis.strip()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Moondream: {analysis}")

                # 3. Send result back to client
                await websocket.send_json({"response": analysis})

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    print(f"Starting server on http::{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)