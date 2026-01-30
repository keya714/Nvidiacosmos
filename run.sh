#!/bin/bash

# NVIDIA Cosmos Video Processor - Run Script

PORT=5000

# Cleanup function to kill processes on script exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    
    # Kill process running on port 5000
    PID=$(lsof -ti:$PORT)
    if [ ! -z "$PID" ]; then
        echo "Killing process on port $PORT (PID: $PID)"
        kill -9 $PID 2>/dev/null
    fi
    
    echo "Cleanup complete."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

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

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing existing process..."
    kill -9 $(lsof -ti:$PORT) 2>/dev/null
    sleep 1
fi

# Run the Flask app (Backend + Frontend)
echo ""
echo "Starting Backend (Flask) on port $PORT..."
echo "Frontend available at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
python app.py
