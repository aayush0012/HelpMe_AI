import { useState, useEffect, useRef } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  
  // Typewriter states
  const [typingText, setTypingText] = useState("");
  const [displayedTyping, setDisplayedTyping] = useState("");
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom of the chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, displayedTyping, thinking]);

  // Typewriter effect logic
  useEffect(() => {
    if (!typingText) {
      setDisplayedTyping("");
      return;
    }
    
    let i = 0;
    const intervalId = setInterval(() => {
      setDisplayedTyping(typingText.slice(0, i + 1));
      i++;
      if (i >= typingText.length) {
        clearInterval(intervalId);
        // Once typing is done, push it to the main messages array and clear typing state
        setMessages(prev => [...prev, { id: Date.now(), sender: "bot", text: typingText }]);
        setTypingText("");
        setDisplayedTyping("");
      }
    }, 15); // Adjust typing speed here

    return () => clearInterval(intervalId);
  }, [typingText]);

  const uploadFile = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:8000/upload", formData);
      setMessage(response.data.message);
    } catch (error) {
      console.log(error);
      setMessage("Upload failed. Please try again.");
    }
    setUploading(false);
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userQuestion = input;
    // Add user message immediately
    setMessages(prev => [...prev, { id: Date.now(), sender: "user", text: userQuestion }]);
    setInput(""); 
    setThinking(true);

    try {
      const response = await axios.post(
        `http://127.0.0.1:8000/chat?question=${userQuestion}`
      );
      setThinking(false);
      // Trigger the typewriter effect for the AI's answer
      setTypingText(response.data.answer);
    } catch (error) {
      console.log(error);
      setThinking(false);
      setTypingText("Failed to reach the server. Please check your connection.");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!thinking && !typingText) handleSend();
    }
  };

  const cssStyles = `
    .app-container {
      min-height: 100vh;
      background-color: #0b0f19;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 30px 20px;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      color: #f8fafc;
    }
    .main-layout {
      display: flex;
      width: 100%;
      max-width: 1200px;
      height: 85vh;
      background: #111827;
      border-radius: 20px;
      border: 1px solid #1e293b;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      overflow: hidden;
    }
    .sidebar {
      width: 320px;
      background: #0f172a;
      border-right: 1px solid #1e293b;
      padding: 24px;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }
    .chat-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #111827;
    }
    
    /* Buttons & Inputs */
    .btn {
      border: none;
      padding: 12px 20px;
      border-radius: 12px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .btn:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }
    .btn-upload { background: #4f46e5; color: white; width: 100%; }
    .btn-upload:hover:not(:disabled) { background: #6366f1; }
    .btn-send { background: #0ea5e9; color: white; padding: 0 24px; }
    .btn-send:hover:not(:disabled) { background: #38bdf8; }
    
    .file-input-wrapper input[type="file"] { display: none; }
    .file-label {
      padding: 14px;
      background: #1e293b;
      border: 1px dashed #475569;
      border-radius: 12px;
      cursor: pointer;
      font-size: 14px;
      color: #cbd5e1;
      display: block;
      text-align: center;
      margin-bottom: 12px;
      transition: all 0.2s ease;
    }
    .file-label:hover { background: #334155; border-color: #64748b; }
    
    .chat-input {
      flex: 1;
      padding: 16px 20px;
      border-radius: 12px;
      border: 1px solid #334155;
      background: #0f172a;
      color: white;
      font-size: 15px;
      outline: none;
      transition: border-color 0.2s;
    }
    .chat-input:focus { border-color: #0ea5e9; }
    
    /* Animations */
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    .cursor {
      display: inline-block;
      width: 8px;
      height: 16px;
      background-color: #0ea5e9;
      margin-left: 4px;
      vertical-align: middle;
      animation: blink 1s step-end infinite;
    }
    @keyframes typingDot {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
      30% { transform: translateY(-4px); opacity: 1; }
    }
    .typing-indicator span {
      display: inline-block;
      width: 6px;
      height: 6px;
      background-color: #94a3b8;
      border-radius: 50%;
      margin: 0 2px;
      animation: typingDot 1.4s infinite ease-in-out both;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
  `;

  return (
    <>
      <style>{cssStyles}</style>
      <div className="app-container">
        
        {/* Main 2-Column Layout */}
        <div className="main-layout">
          
          {/* Left Sidebar */}
          <div className="sidebar">
            <h1 style={{ fontSize: "24px", fontWeight: "700", margin: "0 0 8px 0", display: "flex", alignItems: "center", gap: "8px" }}>
              HelpMe 🤖
            </h1>
            <p style={{ color: "#94a3b8", fontSize: "14px", margin: "0 0 32px 0", lineHeight: "1.5" }}>
              Upload your documents and ask questions to extract insights instantly.
            </p>

            {/* Document Upload Section */}
            <div style={{ marginBottom: "32px" }}>
              <h3 style={{ fontSize: "14px", fontWeight: "600", color: "#e2e8f0", margin: "0 0 12px 0", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                1. Data Source
              </h3>
              <div className="file-input-wrapper">
                <label className="file-label">
                  <input type="file" onChange={(e) => setFile(e.target.files[0])} />
                  {file ? `📄 ${file.name}` : "📁 Select PDF File"}
                </label>
              </div>
              <button 
                className="btn btn-upload" 
                onClick={uploadFile} 
                disabled={!file || uploading}
              >
                {uploading ? "Extracting..." : "Upload Document"}
              </button>
              {message && (
                <p style={{ color: message.includes("failed") ? "#ef4444" : "#10b981", margin: "12px 0 0 0", fontSize: "13px", textAlign: "center" }}>
                  {message}
                </p>
              )}
            </div>

            {/* Info Section */}
            <div>
              <h3 style={{ fontSize: "14px", fontWeight: "600", color: "#e2e8f0", margin: "0 0 12px 0", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                System Info
              </h3>
              <div style={{ background: "#1e293b", padding: "16px", borderRadius: "12px", fontSize: "13px", color: "#cbd5e1", lineHeight: "1.6" }}>
                <p style={{ margin: "0 0 8px 0" }}>• Powered by FastAPI & React</p>
                <p style={{ margin: "0 0 8px 0" }}>• Uses semantic search</p>
                <p style={{ margin: 0 }}>• Dark mode optimized</p>
              </div>
            </div>
          </div>

          {/* Right Chat Area */}
          <div className="chat-area">
            
            {/* Chat Header */}
            <div style={{ padding: "20px 24px", borderBottom: "1px solid #1e293b", background: "rgba(15, 23, 42, 0.5)" }}>
              <h2 style={{ margin: 0, fontSize: "18px", fontWeight: "600", color: "#f8fafc" }}>Conversation</h2>
            </div>

            {/* Chat Messages Scrollable Area */}
            <div style={{ flex: 1, padding: "24px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "24px" }}>
              
              {messages.length === 0 && !thinking && !typingText && (
                <div style={{ margin: "auto", textAlign: "center", color: "#475569" }}>
                  <span style={{ fontSize: "40px", display: "block", marginBottom: "16px" }}>💬</span>
                  <p>Upload a document and start asking questions.</p>
                </div>
              )}

              {/* Render History */}
              {messages.map((msg) => (
                <div key={msg.id} style={{ display: "flex", gap: "16px", flexDirection: msg.sender === "user" ? "row-reverse" : "row" }}>
                  <div style={{ width: "36px", height: "36px", borderRadius: "50%", background: msg.sender === "user" ? "#0ea5e9" : "#334155", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px", flexShrink: 0 }}>
                    {msg.sender === "user" ? "👤" : "🤖"}
                  </div>
                  <div style={{ background: msg.sender === "user" ? "#0ea5e9" : "#1e293b", color: "#f8fafc", padding: "14px 18px", borderRadius: "16px", borderTopRightRadius: msg.sender === "user" ? "4px" : "16px", borderTopLeftRadius: msg.sender === "bot" ? "4px" : "16px", maxWidth: "80%", fontSize: "15px", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                    {msg.text}
                  </div>
                </div>
              ))}

              {/* Render Thinking Indicator */}
              {thinking && (
                <div style={{ display: "flex", gap: "16px" }}>
                  <div style={{ width: "36px", height: "36px", borderRadius: "50%", background: "#334155", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px" }}>🤖</div>
                  <div style={{ background: "#1e293b", padding: "16px 20px", borderRadius: "16px", borderTopLeftRadius: "4px" }}>
                    <div className="typing-indicator"><span></span><span></span><span></span></div>
                  </div>
                </div>
              )}

              {/* Render Typewriter Effect for Newest Bot Message */}
              {typingText && (
                <div style={{ display: "flex", gap: "16px" }}>
                  <div style={{ width: "36px", height: "36px", borderRadius: "50%", background: "#334155", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px", flexShrink: 0 }}>🤖</div>
                  <div style={{ background: "#1e293b", color: "#f8fafc", padding: "14px 18px", borderRadius: "16px", borderTopLeftRadius: "4px", maxWidth: "80%", fontSize: "15px", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                    {displayedTyping}
                    <span className="cursor"></span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input Area */}
            <div style={{ padding: "20px 24px", borderTop: "1px solid #1e293b", background: "#0f172a" }}>
              <div style={{ display: "flex", gap: "12px" }}>
                <input
                  type="text"
                  className="chat-input"
                  placeholder="Ask something about your document..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={thinking || typingText !== ""}
                />
                <button 
                  className="btn btn-send" 
                  onClick={handleSend} 
                  disabled={!input.trim() || thinking || typingText !== ""}
                >
                  Send
                </button>
              </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}

export default App;