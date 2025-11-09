import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
import httpx
from datetime import datetime

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"
TARGET_OBJECT = "apple" 
# ---------------------

app = FastAPI()

# Load client HTML once at startup
with open("client.html", "r") as f:
    CLIENT_HTML = f.read()

@app.get("/")
async def get():
    return HTMLResponse(CLIENT_HTML)

async def query_ollama(image_b64: str, prompt: str) -> str:
    """Queries Ollama with JSON schema enforcement."""
    response_schema = {
        "type": "object",
        "properties": {
            "object_detected": {"type": "boolean", "description": f"True if {TARGET_OBJECT} is visible."},
            "confidence": {"type": "string", "enum": ["high", "low", "none"]}
        },
        "required": ["object_detected", "confidence"]
    }

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME, "prompt": prompt, "images": [image_b64],
                "stream": False, "format": response_schema, "options": {"temperature": 0.0}
            }
        )
        response.raise_for_status()
        return response.json().get("response", "{}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected")
    try:
        await websocket.send_json({"response": f"System: Watching for {TARGET_OBJECT.upper()}"})
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            image_data = msg.get("image", "").split(",")[-1]

            print(f"Scanning for {TARGET_OBJECT}...")
            json_res = await query_ollama(image_data, f"Analyze for presence of {TARGET_OBJECT}.")

            if websocket.client_state == WebSocketState.DISCONNECTED: break

            try:
                vlm_data = json.loads(json_res)
                if vlm_data.get("object_detected"):
                     resp = f"ALERT: {TARGET_OBJECT.upper()} DETECTED ({vlm_data.get('confidence')})"
                     print(f"!!! FOUND {TARGET_OBJECT.upper()} !!!")
                else:
                     resp = f"Scanning for {TARGET_OBJECT}... (clear)"
            except:
                resp = "Error: Invalid model response."
            
            await websocket.send_json({"response": resp})

    except (WebSocketDisconnect, RuntimeError):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)