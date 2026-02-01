from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import csv
import os
from datetime import datetime
from pathlib import Path
import base64
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import torch
from PIL import Image
import io
import cv2
import numpy as np

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
tokenizer = None
image_processor = None
model_name = "nvidia/Alpamayo-R1-10B"

def load_model():
    """Load the model and processor"""
    global model, tokenizer, image_processor
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded successfully!")

def extract_video_frames(video_bytes, num_frames=8):
    """Extract frames from video"""
    # Save video temporarily
    temp_path = "/tmp/temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
    
    # Open video
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    os.remove(temp_path)
    
    return frames

def save_to_csv(data):
    """Save results to CSV"""
    csv_filename = f"result_{model_name.replace('/', '_')}.csv"
    csv_path = csv_filename  # Save in current folder
    
    # No need to create output directory
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    # Write to CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'prompt', 'response', 'temperature', 'response_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)
    
    return csv_filename

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    temperature: float = Form(0.7)
):
    """Process video with prompt"""
    try:
        start_time = time.time()
        
        # Read video file
        video_bytes = await video.read()
        
        # Extract frames from video
        frames = extract_video_frames(video_bytes)
        
        if not frames:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract frames from video"}
            )
        
        # Prepare inputs
        text_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        image_inputs = image_processor(images=frames, return_tensors="pt").to(model.device)
        
        # Combine inputs
        inputs = {**text_inputs, **image_inputs}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=temperature > 0
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare data for CSV
        csv_data = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'temperature': temperature,
            'response_time': f"{response_time:.2f}s"
        }
        
        # Save to CSV
        csv_filename = save_to_csv(csv_data)
        
        return {
            "success": True,
            "response": response,
            "response_time": f"{response_time:.2f}s",
            "temperature": temperature,
            "timestamp": csv_data['timestamp'],
            "csv_file": csv_filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": model_name}

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Default port
    port = 8000
    
    # Check for --port argument
    if "--port" in sys.argv:
        try:
            port_index = sys.argv.index("--port")
            port = int(sys.argv[port_index + 1])
        except (IndexError, ValueError):
            print("Invalid port argument. Using default port 8000")
            port = 8000
    
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)