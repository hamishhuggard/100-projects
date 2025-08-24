#!/bin/bash

echo "🚀 Starting Chatbot React Frontend..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
if ! python3 -c "import flask, openai, dotenv" 2>/dev/null; then
    echo "📥 Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

# Check if Node.js dependencies are installed
echo "📦 Checking Node.js dependencies..."
if [ ! -d "node_modules" ]; then
    echo "📥 Installing Node.js dependencies..."
    npm install
fi

echo "✅ Dependencies ready!"

# Start backend in background
echo "🔧 Starting Flask backend..."
python3 app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting React frontend..."
npm start

# Cleanup on exit
trap "echo '🛑 Shutting down...'; kill $BACKEND_PID; exit" INT TERM
wait 