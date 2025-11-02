#!/bin/bash

# CEREBROS Development Environment Startup Script
# Starts both backend API and frontend UI

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_LOG="/tmp/cerebros_backend.log"
FRONTEND_LOG="/tmp/cerebros_frontend.log"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🚀 CEREBROS Development Environment Startup${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Please run:${NC}"
    echo -e "${YELLOW}   python3 -m venv .venv${NC}"
    echo -e "${YELLOW}   source .venv/bin/activate${NC}"
    echo -e "${YELLOW}   pip install -r requirements.txt${NC}"
    exit 1
fi

# Kill any existing processes
echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
pkill -f "server/app.py" 2>/dev/null || true
pkill -f "vite.*web_demo" 2>/dev/null || true
sleep 1

# Start Backend
echo -e "${GREEN}📡 Starting backend server on port 8080...${NC}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
nohup python3 server/app.py > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}   Backend PID: $BACKEND_PID${NC}"
echo -e "${GREEN}   Logs: tail -f $BACKEND_LOG${NC}"

# Wait for backend to start
echo -e "${YELLOW}⏳ Waiting for backend to be ready...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:8080/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend ready!${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${YELLOW}⚠️  Backend may not be ready yet, check logs${NC}"
    fi
    sleep 1
done

# Start Frontend
echo -e "${GREEN}🎨 Starting frontend on port 5173...${NC}"
cd "$PROJECT_ROOT/web_demo"
nohup npm run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}   Frontend PID: $FRONTEND_PID${NC}"
echo -e "${GREEN}   Logs: tail -f $FRONTEND_LOG${NC}"

# Wait for frontend to start
echo -e "${YELLOW}⏳ Waiting for frontend to be ready...${NC}"
sleep 3

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ CEREBROS Development Environment Running${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}📡 Backend:  http://localhost:8080${NC}"
echo -e "${GREEN}   API Docs: http://localhost:8080/docs${NC}"
echo -e "${GREEN}🎨 Frontend: http://localhost:5173${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}📝 Logs:${NC}"
echo -e "${YELLOW}   Backend:  tail -f $BACKEND_LOG${NC}"
echo -e "${YELLOW}   Frontend: tail -f $FRONTEND_LOG${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}🛑 To stop both services:${NC}"
echo -e "${YELLOW}   kill $BACKEND_PID $FRONTEND_PID${NC}"
echo -e "${YELLOW}   OR run: pkill -f 'server/app.py|vite.*web_demo'${NC}"
echo -e "${BLUE}============================================================${NC}"

# Save PIDs to file for easy cleanup
echo "$BACKEND_PID" > /tmp/cerebros_backend.pid
echo "$FRONTEND_PID" > /tmp/cerebros_frontend.pid

echo -e "${GREEN}🎉 Ready! Open http://localhost:5173 in your browser${NC}"
