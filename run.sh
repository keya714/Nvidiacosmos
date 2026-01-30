#!/bin/bash

# NVIDIA Cosmos Video Processor - Run Script

echo "Starting NVIDIA Cosmos Video Processor..."
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set."
    echo "Please set it with: export HF_TOKEN=your_token_here"
    echo ""
    read -p "Enter your HuggingFace token (or press Enter to skip): " token
    if [ ! -z "$token" ]; then
        export HF_TOKEN=$token
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the Flask app
echo ""
echo "Starting Flask server..."
echo "Open http://localhost:5000 in your browser"
echo ""
python app.py
