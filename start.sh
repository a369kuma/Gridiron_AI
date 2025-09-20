#!/bin/bash

# Gridiron AI Startup Script
# This script starts both the backend API and frontend development servers

echo "🏈 Starting Gridiron AI..."

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/raw data/processed models

# Train the initial model
echo "Training initial model..."
cd backend
python run_training.py
cd ..

# Start the API server in background
echo "Starting API server..."
cd backend
python run_api.py &
API_PID=$!
cd ..

# Wait a moment for API to start
sleep 3

# Start the frontend development server
echo "Starting frontend development server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo "✅ Gridiron AI is starting up!"
echo "📊 API Server: http://localhost:5000"
echo "🎨 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping Gridiron AI..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
