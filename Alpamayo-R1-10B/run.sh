#!/bin/bash

# Alpamayo-R1-10B Video Model Tester - Startup Script
# This script sets up and runs the application on port 5000

echo "=================================="
echo "Alpamayo-R1-10B Video Model Tester"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed"
    exit 1
fi

echo "✓ Python3 found"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt --break-system-packages

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"

# Prompt for Hugging Face token
echo ""
echo "🔑 Hugging Face Authentication"
echo "If the model requires authentication, enter your HF token."
echo "Press Enter to skip if not required."
read -p "Enter your Hugging Face token: " HF_TOKEN

if [ ! -z "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "✓ Token set"
else
    echo "ℹ Proceeding without token"
fi

echo ""
echo "🚀 Starting backend server on port 5000..."
echo ""
echo "Backend API: http://localhost:5000"
echo "Frontend: Open index.html in your browser"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""
echo "=================================="
echo ""

# Start the FastAPI server on port 5000
python3 app.py --port 5000 &
BACKEND_PID=$!

# Wait a moment for the server to start
sleep 2

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "✓ Backend server started successfully (PID: $BACKEND_PID)"
    echo ""
    echo "📝 Instructions:"
    echo "   1. Open index.html in your web browser"
    echo "   2. Upload a video file"
    echo "   3. Enter your prompt"
    echo "   4. Adjust temperature if needed"
    echo "   5. Click 'Process Video'"
    echo ""
    echo "Results will be saved to: result_nvidia_Alpamayo-R1-10B.csv"
    echo ""
    
    # Wait for the backend process
    wait $BACKEND_PID
else
    echo "❌ Error: Failed to start backend server"
    exit 1
fi