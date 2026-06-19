# LLM Instruction Guide: High-Fidelity Chatbot with Langfuse Tracking

This guide provides a platform-agnostic, zero-assumption specification that any LLM can use to rebuild this identical chatbot application from scratch.

---

## 1. Application Overview & Architecture
The system consists of:
1. **FastAPI Backend**: Serves static pages, handles CORS/routing, initializes standard OpenAI LLM calls, and transparently logs traces to Langfuse using the `langfuse.openai` wrapper.
2. **Vanilla HTML/CSS/JS Frontend**: A glassmorphic dark-themed chat interface with responsive sidebar, scrolling messages pane, character loading indicators, and context memory tracking.
3. **Docker Stack**: Fully containerized runtime exposing port `8000` with native `.env` loading.

---

## 2. Directory Structure
```text
/
├── app/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
├── .env
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 3. Configuration Setup (`.env`)
Create a `.env` file at the root:
```env
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com" # Or your host region URL
CHAT_GPT_API="your-openai-api-key"
```

---

## 4. Source Files Specifications

### Step 4.1: Dependencies (`requirements.txt`)
Specify the following Python packages:
```text
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
openai>=1.0.0
langfuse>=2.0.0
python-dotenv>=1.0.0
```

---

### Step 4.2: FastAPI Backend (`app/main.py`)
Create a FastAPI app that:
1. Runs `dotenv.load_dotenv()` to pull keys.
2. Automatically maps incoming environment keys to standard keys:
   - Sets `os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]`
   - Sets `os.environ["OPENAI_API_KEY"] = os.environ["CHAT_GPT_API"]`
3. Imports the `OpenAI` client wrapper from `langfuse.openai` to auto-capture traces.
4. Exposes a POST endpoint at `/api/chat` that accepts a payload of message history:
   ```json
   {
     "messages": [
       {"role": "user", "content": "Hello!"}
     ]
   }
   ```
5. Calls `openai_client.chat.completions.create` using `model="gpt-4o-mini"`, with trace parameters (`name="chatbot-message-generation"`, `metadata={"environment": "production"}`). Returns `{"response": bot_response}`.
6. Mounts the directory `app/static` under `/static` using `StaticFiles`.
7. Directs GET `/` to return `FileResponse("app/static/index.html")`.

Here is the code:
```python
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Map custom keys to SDK standard keys
if "LANGFUSE_BASE_URL" in os.environ and "LANGFUSE_HOST" not in os.environ:
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
if "CHAT_GPT_API" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["CHAT_GPT_API"]

from langfuse.openai import OpenAI

app = FastAPI(title="Langfuse Chatbot API")

try:
    openai_client = OpenAI()
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized.")
    try:
        api_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            name="chatbot-message-generation",
            metadata={"environment": "production", "interface": "web-ui"}
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")
```

---

### Step 4.3: Frontend Layout (`app/static/index.html`)
Build a semantic HTML5 single-page application with:
- Google Fonts: **Inter** and **Outfit**.
- **Sidebar (`<aside>`)**: Logo (using inline SVG gradient), tech descriptions, "Clear Conversation" button, and status indicator.
- **Chat Window (`<main>`)**: Header with "Online" indicator, chat messages div (`id="messages-container"`), loading dots, and input form (`id="chat-form"`).
- Imported script: `<script src="/static/script.js"></script>` just before `</body>`.

Here is the structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura Chat - Premium AI Chatbot</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="app-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#grad1)" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <defs>
                            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#06b6d4;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                    </svg>
                    <span>AuraAI</span>
                </div>
            </div>
            <div class="sidebar-content">
                <div class="info-card">
                    <h3>Integration Details</h3>
                    <div class="tech-item">
                        <span class="tech-dot api"></span>
                        <div class="tech-text">
                            <strong>ChatGPT API</strong>
                            <p>Powered by GPT-4o-mini</p>
                        </div>
                    </div>
                    <div class="tech-item">
                        <span class="tech-dot langfuse"></span>
                        <div class="tech-text">
                            <strong>Langfuse Logs</strong>
                            <p>Real-time trace logging</p>
                        </div>
                    </div>
                    <div class="tech-item">
                        <span class="tech-dot docker"></span>
                        <div class="tech-text">
                            <strong>Docker Container</strong>
                            <p>Hosted and isolated</p>
                        </div>
                    </div>
                </div>
                <button type="button" class="new-chat-btn" id="new-chat-btn">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                    <span>Clear Conversation</span>
                </button>
            </div>
            <div class="sidebar-footer">
                <div class="status-indicator">
                    <span class="pulse-dot"></span>
                    <span>System Operational</span>
                </div>
            </div>
        </aside>

        <main class="chat-area">
            <header class="chat-header">
                <div class="bot-info">
                    <div class="bot-avatar"><span>AI</span></div>
                    <div class="bot-status-container">
                        <h2>Aura</h2>
                        <div class="status-badge">
                            <span class="online-indicator"></span>
                            <span>Online</span>
                        </div>
                    </div>
                </div>
            </header>
            <div class="messages-container" id="messages-container">
                <div class="message assistant welcome">
                    <div class="message-bubble">
                        <p>Hello! I am Aura, a simple chatbot powered by GPT-4o-mini and monitored with Langfuse. How can I help you today?</p>
                    </div>
                    <span class="message-time">Just now</span>
                </div>
            </div>
            <div class="typing-indicator-container" id="typing-indicator">
                <div class="bot-avatar mini">AI</div>
                <div class="typing-bubble">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            <footer class="chat-footer">
                <form class="input-container" id="chat-form">
                    <input type="text" id="chat-input" placeholder="Ask Aura something..." autocomplete="off" required>
                    <button type="submit" class="send-btn" id="send-button">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                    </button>
                </form>
            </footer>
        </main>
    </div>
    <script src="/static/script.js"></script>
</body>
</html>
```

---

### Step 4.4: Frontend CSS Styles (`app/static/style.css`)
Generate styles featuring:
1. Custom CSS Variables defining colors: Dark `#090a0f`, Glassy slate panels `rgba(22, 28, 45, 0.45)`, violet `#8b5cf6` and pink `#d946ef` gradients.
2. Backdrop Filter properties for panels.
3. Flex sidebar width `320px`, hiding on devices under `768px`.
4. Pulsing and typing bouncing animation properties.
5. Pill input alignment with a focus gradient border.

Here are the stylesheet properties:
```css
:root {
    --bg-primary: #090a0f;
    --bg-secondary: rgba(17, 19, 31, 0.7);
    --glass-bg: rgba(22, 28, 45, 0.45);
    --glass-border: rgba(255, 255, 255, 0.07);
    --glass-highlight: rgba(255, 255, 255, 0.12);
    
    --accent-purple: #8b5cf6;
    --accent-pink: #d946ef;
    --accent-teal: #0d9488;
    --accent-blue: #0ea5e9;
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    
    --gradient-accent: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-pink) 100%);
    --gradient-bg: radial-gradient(circle at 50% 50%, #1e1b4b 0%, var(--bg-primary) 70%);
    
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --font-sans: 'Inter', sans-serif;
    --font-display: 'Outfit', sans-serif;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: var(--bg-primary);
    background-image: var(--gradient-bg);
    color: var(--text-primary);
    font-family: var(--font-sans);
    height: 100vh;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
}

.app-container {
    width: 95vw;
    height: 90vh;
    max-width: 1200px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    display: flex;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

.sidebar {
    width: 320px;
    border-right: 1px solid var(--glass-border);
    display: flex;
    flex-direction: column;
    background: rgba(10, 11, 18, 0.4);
}

.sidebar-header {
    padding: 24px;
    border-bottom: 1px solid var(--glass-border);
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-content {
    flex: 1;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.info-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 20px;
}

.info-card h3 {
    font-family: var(--font-display);
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 16px;
}

.tech-item {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}

.tech-item:last-child {
    margin-bottom: 0;
}

.tech-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.tech-dot.api { background: var(--accent-blue); }
.tech-dot.langfuse { background: var(--accent-purple); }
.tech-dot.docker { background: var(--accent-teal); }

.tech-text strong {
    display: block;
    font-size: 13px;
    color: var(--text-primary);
}

.tech-text p {
    font-size: 11px;
    color: var(--text-secondary);
}

.new-chat-btn {
    background: transparent;
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    border-radius: 12px;
    padding: 12px 16px;
    font-family: var(--font-sans);
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    cursor: pointer;
    transition: var(--transition-smooth);
}

.new-chat-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: var(--accent-purple);
    transform: translateY(-1px);
}

.sidebar-footer {
    padding: 20px 24px;
    border-top: 1px solid var(--glass-border);
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    color: var(--text-secondary);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    position: relative;
}

.pulse-dot::after {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    background-color: #10b981;
    border-radius: 50%;
    animation: pulse 1.5s infinite ease-in-out;
}

.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: rgba(15, 23, 42, 0.25);
}

.chat-header {
    height: 80px;
    border-bottom: 1px solid var(--glass-border);
    display: flex;
    align-items: center;
    padding: 0 32px;
}

.bot-info {
    display: flex;
    align-items: center;
    gap: 16px;
}

.bot-avatar {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: var(--gradient-accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 16px;
    color: #fff;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
}

.bot-status-container h2 {
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 600;
}

.status-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
}

.online-indicator {
    width: 6px;
    height: 6px;
    background: #10b981;
    border-radius: 50%;
}

.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.messages-container::-webkit-scrollbar {
    width: 6px;
}

.messages-container::-webkit-scrollbar-track {
    background: transparent;
}

.messages-container::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 3px;
}

.messages-container::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.15);
}

.message {
    display: flex;
    flex-direction: column;
    max-width: 75%;
    animation: fadeIn 0.4s ease forwards;
}

.message.user {
    align-self: flex-end;
}

.message.assistant {
    align-self: flex-start;
}

.message-bubble {
    padding: 16px 20px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.5;
    position: relative;
}

.message.user .message-bubble {
    background: var(--gradient-accent);
    color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 10px 20px -5px rgba(139, 92, 246, 0.3);
}

.message.assistant .message-bubble {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
}

.message-time {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
    align-self: flex-start;
}

.message.user .message-time {
    align-self: flex-end;
}

.typing-indicator-container {
    display: none;
    align-items: center;
    gap: 12px;
    padding: 0 32px 20px 32px;
    align-self: flex-start;
}

.bot-avatar.mini {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    font-size: 11px;
}

.typing-bubble {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--glass-border);
    padding: 12px 18px;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    display: flex;
    gap: 5px;
    align-items: center;
}

.typing-dot {
    width: 6px;
    height: 6px;
    background: var(--text-secondary);
    border-radius: 50%;
    animation: typingBounce 1.4s infinite ease-in-out both;
}

.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }

.chat-footer {
    padding: 24px 32px;
    border-top: 1px solid var(--glass-border);
}

.input-container {
    display: flex;
    align-items: center;
    gap: 16px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 8px 8px 8px 20px;
    transition: var(--transition-smooth);
}

.input-container:focus-within {
    border-color: var(--accent-purple);
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.15);
    background: rgba(255, 255, 255, 0.04);
}

.input-container input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 15px;
}

.input-container input::placeholder {
    color: var(--text-muted);
}

.send-btn {
    background: var(--gradient-accent);
    color: #fff;
    border: none;
    border-radius: 12px;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: var(--transition-smooth);
    box-shadow: 0 4px 10px rgba(139, 92, 246, 0.3);
}

.send-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 15px rgba(139, 92, 246, 0.4);
}

.send-btn:active {
    transform: scale(0.95);
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes typingBounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
}

@media(max-width: 768px) {
    .sidebar {
        display: none;
    }
    .app-container {
        width: 100vw;
        height: 100vh;
        border-radius: 0;
        border: none;
    }
    .chat-header {
        padding: 0 20px;
    }
    .messages-container {
        padding: 20px;
    }
    .chat-footer {
        padding: 16px;
    }
}
```

---

### Step 4.5: Frontend Interactivity (`app/static/script.js`)
Create JavaScript that handles event submissions directly (omitting the `DOMContentLoaded` event listener since the script sits at the end of the body) to prevent race conditions:
1. Grabs `chat-form`, `chat-input`, `messages-container`, `typing-indicator`, `new-chat-btn`, and `send-button`.
2. Stores a local `chatHistory` array.
3. Implements `appendMessage(role, content)` that splits input text on `\n` to prevent loss of spacing, sanitizes content via `textContent` insertion, stamps current times, and scrolls to bottom.
4. Triggers `e.preventDefault()` on submission, disables inputs, displays the typing indicator, posts history list to `/api/chat`, resolves response data, updates UI, and enables controls back.
5. Enables clear-state trigger via `new-chat-btn`.

Here is the Javascript logic:
```javascript
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const messagesContainer = document.getElementById('messages-container');
const typingIndicator = document.getElementById('typing-indicator');
const newChatBtn = document.getElementById('new-chat-btn');
const sendButton = document.getElementById('send-button');

let chatHistory = [];

function appendMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role);

    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble');

    const paragraphs = content.split('\n');
    paragraphs.forEach(text => {
        const p = document.createElement('p');
        p.textContent = text;
        if (text.trim() === '') {
            p.style.minHeight = '1em';
        }
        bubble.appendChild(p);
    });

    const timeSpan = document.createElement('span');
    timeSpan.classList.add('message-time');
    const now = new Date();
    timeSpan.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(bubble);
    messageDiv.appendChild(timeSpan);
    messagesContainer.appendChild(messageDiv);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const prompt = chatInput.value.trim();
    if (!prompt) return;

    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    appendMessage('user', prompt);
    chatHistory.push({ role: 'user', content: prompt });

    typingIndicator.style.display = 'flex';
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ messages: chatHistory })
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || `Server returned status ${response.status}`);
        }

        const data = await response.json();
        const reply = data.response;

        typingIndicator.style.display = 'none';

        appendMessage('assistant', reply);
        chatHistory.push({ role: 'assistant', content: reply });

    } catch (err) {
        console.error('Chat error:', err);
        typingIndicator.style.display = 'none';
        appendMessage('assistant', `Unable to get a response. Error: ${err.message}`);
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
});

newChatBtn.addEventListener('click', () => {
    const welcomeMessage = messagesContainer.querySelector('.welcome');
    messagesContainer.innerHTML = '';
    if (welcomeMessage) {
        messagesContainer.appendChild(welcomeMessage);
    }
    chatHistory = [];
    chatInput.focus();
});
```

---

## 5. Dockerization Setup

### Step 5.1: `Dockerfile`
```dockerfile
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 5.2: `docker-compose.yml`
```yaml
services:
  chatbot:
    build: .
    container_name: aura-chatbot
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

---

## 6. Execution Commands
To build and start the application stack on any environment with Docker installed, run:
```bash
# 1. Start the container in detached mode
docker compose up --build -d

# 2. Monitor execution output
docker logs -f aura-chatbot

# 3. Access in browser
http://localhost:8000
```
