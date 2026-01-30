from flask import Flask, render_template, request, jsonify
import transformers
import torch
import os

app = Flask(__name__)

# Store model and processor globally
model = None
processor = None
current_model_name = None

def load_model(model_name):
    global model, processor, current_model_name
    
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        # Try to import Qwen3VLForConditionalGeneration directly
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, 
            dtype=torch.float16, 
            device_map="auto", 
            attn_implementation="sdpa", 
            token=hf_token,
            trust_remote_code=True
        )
    except (AttributeError, ImportError):
        # Fall back to AutoModelForVision2Seq
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            model_name, 
            dtype=torch.float16, 
            device_map="auto", 
            attn_implementation="sdpa", 
            token=hf_token,
            trust_remote_code=True
        )
    
    processor = transformers.AutoProcessor.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    current_model_name = model_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    try:
        data = request.json
        video_path = data.get('video_path')
        prompt = data.get('prompt')
        model_name = data.get('model_name', 'nvidia/Cosmos-Reason2-8B')
        fps = int(data.get('fps', 4))
        
        # Load model if not loaded or if model name changed
        global current_model_name
        if model is None or current_model_name != model_name:
            load_model(model_name)
        
        # Create messages
        video_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video", 
                        "video": video_path,
                        "fps": fps,
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            },
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            video_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        )
        inputs = inputs.to(model.device)
        
        # Run inference
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return jsonify({
            'success': True,
            'output': output_text[0] if output_text else ''
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
