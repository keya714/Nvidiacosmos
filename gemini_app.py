from flask import Flask, render_template, request, jsonify
import transformers
import torch
import os
import csv
from datetime import datetime
from PIL import Image
import cv2

app = Flask(__name__)

# Store model and processor globally
model = None
processor = None
current_model_name = None

def load_model(model_name):
    global model, processor, current_model_name
    
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it with your HuggingFace token.")
    
    print(f"Loading model: {model_name}...")
    
    # Try to login with the token
    from huggingface_hub import login
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")
    
    # Load model with proper authentication - use Qwen2VL specific class for Qwen models
    if "qwen2-vl" in model_name.lower():
        # Use Qwen2VL specific class
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )
        print("Loaded as Qwen2VL model")
    else:
        # Try general vision model
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )
        print("Loaded as Vision2Seq model")
    
    processor = transformers.AutoProcessor.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    current_model_name = model_name
    print(f"Model loaded successfully!")

def save_to_csv(video_path, prompt, model_name, fps, output):
    """Save prompt and output to CSV file"""
    csv_file = 'gemini_results.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['Timestamp', 'Model Name', 'Video Path', 'FPS', 'Prompt', 'Output'])
        
        # Write data
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            model_name,
            video_path,
            fps,
            prompt,
            output
        ])

def extract_frames_from_video(video_path, fps=1):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        frame_count += 1
    
    cap.release()
    return frames

@app.route('/')
def index():
    return render_template('gemini.html')

@app.route('/process', methods=['POST'])
def process_video():
    try:
        data = request.json
        video_path = data.get('video_path')
        prompt = data.get('prompt')
        model_name = data.get('model_name', 'Qwen/Qwen2-VL-2B-Instruct')
        fps = int(data.get('fps', 1))
        
        # Load model if not loaded or if model name changed
        global current_model_name
        if model is None or current_model_name != model_name:
            load_model(model_name)
        
        # Extract frames from video
        frames = extract_frames_from_video(video_path, fps)
        
        if not frames:
            return jsonify({
                'success': False,
                'error': 'Could not extract frames from video'
            }), 400
        
        # Prepare input with frames and prompt
        # For Gemini/Gemma models with vision support
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in frames],
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = processor(
            text=prompt,
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode output
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        result_output = output_text[0] if output_text else ''
        
        # Save to CSV
        save_to_csv(video_path, prompt, model_name, fps, result_output)
        
        return jsonify({
            'success': True,
            'output': result_output,
            'frames_processed': len(frames)
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details)
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
