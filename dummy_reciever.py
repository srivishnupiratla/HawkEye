from fastapi import FastAPI, Request
import uvicorn
import json

app = FastAPI()

@app.post("/deep-inference")
async def receive_frame(request: Request):
    data = await request.json()
    triggers = data.get("triggers", [])
    timestamp = data.get("timestamp")
    
    print("\n" + "="*40)
    print(f"RECEIVED TRIGGER at {timestamp}")
    print(f"Triggers detected: {triggers}")
    # We don't print the full image data because it's huge, just its length
    img_len = len(data.get("image", ""))
    print(f"Image data size: {img_len} chars")
    print("="*40 + "\n")
    
    return {"status": "received", "next_step": "processing"}

if __name__ == "__main__":
    print("Dummy Deep Inference Service running on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)