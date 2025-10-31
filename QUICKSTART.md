# 🚀 CEREBROS NotGPT - Quick Start Guide

**Last Updated:** 2025-10-31  
**Status:** ✅ Fully Operational

---

## 🎯 What's Fixed

### Backend
- ✅ API running on **http://localhost:8080**
- ✅ Upload endpoint `/api/upload` added
- ✅ All 8 REST endpoints operational
- ✅ Demo assistant trained and ready

### Frontend  
- ✅ React app running on **http://localhost:5173**
- ✅ Full routing with React Router
- ✅ Dashboard shows all assistants
- ✅ Chat interface for conversations
- ✅ Training wizard (PromptTraining component)

---

## 🌐 Access the Dashboard

### **Open in Browser:** http://localhost:5173

You'll see:
- **Dashboard** (`/`) - Lists all AI assistants with status
- **Create Assistant** (`/new`) - 5-step training wizard
- **Chat** (`/chat/:id`) - Interactive chat with any assistant

---

## 📋 Available Features

### 1. Dashboard Page
- View all assistants
- See training status (Ready/Training/Unknown)
- Quick stats (Total, Ready, Training counts)
- Click "Chat" to talk to ready assistants
- Click "Details" for status info

### 2. Create Assistant Page
- Upload training data (CSV/JSON files)
- Review and edit prompt examples
- Add reasoning and expected outputs
- Multi-stage training wizard

### 3. Chat Interface
- Real-time conversation with assistants
- Message history
- Typing indicators
- Error handling
- Suggestion prompts

---

## 🔌 Backend Endpoints

All endpoints are on **http://localhost:8080**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/assistants` | List all assistants |
| GET | `/assistants/{id}/status` | Get assistant details |
| POST | `/assistants/{id}/query` | Send query to assistant |
| POST | `/api/upload` | **NEW** Upload training files |
| POST | `/assistants/train` | Start training pipeline |
| DELETE | `/assistants/{id}` | Delete assistant |

---

## 🧪 Test It Out

### Quick Test Flow:
1. Open **http://localhost:5173** 
2. You should see the "demo" assistant (already trained)
3. Click **"Chat"** button
4. Type a message and hit Send
5. Get AI response back!

### Upload Test:
1. Go to **http://localhost:5173/new**
2. Click file upload (step navigation shows you're on step 2)
3. Select a CSV or JSON file
4. Backend will save to `priv/nfs/agents/assistant_*/uploads/`

---

## 🐛 Troubleshooting

### Backend Not Responding
```bash
# Check if running
curl http://localhost:8080/health

# If not, restart:
cd /home/mo/thunderline/cerebros-core-algorithm-alpha
CEREBROS_API_PORT=8080 python3 server/app.py
```

### Frontend Not Loading
```bash
# Check if running
curl http://localhost:5173

# If not, restart:
cd web_demo
npm run dev
```

### Upload Fails
- Make sure backend is running on port 8080
- Check file is CSV or JSON format
- Look at backend logs for errors

---

## 📂 Project Structure

```
cerebros-core-algorithm-alpha/
├── server/
│   └── app.py              # FastAPI backend (8080)
├── web_demo/
│   ├── src/
│   │   ├── App.tsx         # React router setup
│   │   ├── components/
│   │   │   ├── Dashboard.tsx      # Main dashboard
│   │   │   ├── Chat.tsx           # Chat interface
│   │   │   ├── PromptTraining.tsx # Training wizard
│   │   │   └── MultiStageWizard.tsx
│   │   └── index.tsx       # React entry point
│   ├── package.json
│   └── vite.config.ts      # Vite dev server config
├── scripts/
│   └── process_user_samples.py   # Data processing
├── multi_stage_trainer.py  # 5-stage training
└── priv/nfs/               # Data storage
    └── agents/
        └── demo/           # Demo assistant
            ├── checkpoints/
            ├── datasets/
            └── uploads/    # NEW: Upload directory
```

---

## 🎨 Tech Stack

**Frontend:**
- React 18
- TypeScript
- React Router v6
- Tailwind CSS
- Lucide React icons
- Vite dev server

**Backend:**
- Python 3.13
- FastAPI
- Uvicorn
- MLflow
- PyTorch

---

## 🚦 Current Status

| Component | Status | URL |
|-----------|--------|-----|
| Backend API | ✅ Running | http://localhost:8080 |
| React Frontend | ✅ Running | http://localhost:5173 |
| Dashboard | ✅ Working | http://localhost:5173/ |
| Chat | ✅ Working | http://localhost:5173/chat/demo |
| Upload | ✅ Fixed | Backend endpoint added |
| Training | ✅ Complete | Demo assistant ready |

---

## 🎯 Next Steps

### For Demo:
1. ✅ Open http://localhost:5173
2. ✅ Show dashboard with assistants
3. ✅ Click Chat on demo assistant
4. ✅ Send a message, get response
5. ✅ Navigate to /new to show training wizard

### For Development:
- Add file upload UI to PromptTraining component
- Connect training wizard to backend training endpoint
- Add progress tracking for training
- Implement assistant deletion
- Add authentication

---

## 💡 Pro Tips

1. **Use Browser DevTools:** Open Network tab to see API calls
2. **Check Logs:** Backend logs go to `/tmp/cerebros_api.log`
3. **Hot Reload:** Both frontend and backend support live reload
4. **Test API:** Use http://localhost:8080/docs for Swagger UI

---

## 📞 Need Help?

All systems are go! If something breaks:
1. Check both servers are running (ports 8080 and 5173)
2. Look at browser console for frontend errors
3. Check `/tmp/cerebros_api.log` for backend errors
4. Restart both servers if needed

---

**🎉 You're all set! The dashboard is fully operational.**

Open **http://localhost:5173** and start chatting with your AI assistants!
