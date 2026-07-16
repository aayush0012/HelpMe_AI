import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "https://help-me-zdr2.onrender.com");

const getSessionId = () => {
  let sessionId = sessionStorage.getItem("helpme_session_id");
  if (!sessionId) {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      sessionId = crypto.randomUUID();
    } else {
      sessionId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    }
    sessionStorage.setItem("helpme_session_id", sessionId);
  }
  return sessionId;
};

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  };

  useEffect(() => {
    if (!showLanding) {
      scrollToBottom();
    }
  }, [messages, thinking, showLanding]);

  const uploadFile = async () => {
    if (!file) return;
    setUploading(true);
    setMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const sessionId = getSessionId();
      const response = await axios.post(`${API_BASE}/upload?session_id=${sessionId}`, formData);
      setMessage(response.data.message);
    } catch (err) {
      console.log(err);
      const detail = err.response?.data?.detail;
      setMessage(detail ? `Upload failed: ${detail}` : "Upload Failed");
    }
    setUploading(false);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const question = input;
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        sender: "user",
        text: question,
      },
    ]);

    setInput("");
    setThinking(true);

    try {
      const sessionId = getSessionId();
      const response = await axios.post(
        `${API_BASE}/chat?question=${encodeURIComponent(question)}&session_id=${sessionId}`
      );
      setThinking(false);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: "bot",
          text: response.data.answer,
          sources: response.data.sources || [],
        },
      ]);
    } catch (err) {
      console.log(err);
      setThinking(false);
      const detail = err.response?.data?.detail;
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: "bot",
          text: detail || "Unable to connect to server.",
          sources: [],
        },
      ]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!thinking) {
        handleSend();
      }
    }
  };

  if (showLanding) {
    return (
      <div className="landing-container">
        <div className="landing-hero">
          <h1 className="landing-logo">HelpMe AI</h1>
          <p className="landing-subtitle">Your AI Assistant</p>
          <p className="landing-tagline">Ask questions about your PDFs</p>
          <p className="landing-description">
            Upload any PDF document and start chatting. The assistant will search the document to find answers and show you the exact page numbers they came from.
          </p>
          <ul className="landing-bullet-list">
            <li>Upload and search through PDF files</li>
            <li>Read scanned text and tables using cloud OCR</li>
            <li>Find the exact page numbers for every answer</li>
          </ul>
        </div>

        <div className="landing-action-panel">
          <div className="action-card">
            <h3>Start Now</h3>
            <p>Go to the workspace to upload your PDF and start asking questions.</p>
            <button className="landing-btn" onClick={() => setShowLanding(false)}>
              Start Now
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container animate-fade-in">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-row">
            <h1 className="brand-logo">HelpMe AI</h1>
            <button className="back-btn" onClick={() => setShowLanding(true)} title="Back to welcome page">
              Home
            </button>
          </div>
          <p className="brand-subtitle">Document Search & RAG System</p>
        </div>

        <div className="sidebar-section">
          <h2 className="section-title">Ingest Document</h2>
          <p className="section-description">
            Upload a PDF file to process and build a semantic search index.
          </p>

          <label className="compact-upload-card">
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => {
                setFile(e.target.files[0]);
                setMessage("");
              }}
            />
            <span className="upload-icon">📄</span>
            <div className="upload-details">
              <span className="file-status">
                {file ? file.name : "Select PDF Document"}
              </span>
              <span className="upload-action-text">Click to browse</span>
            </div>
          </label>

          <button
            className="action-btn"
            onClick={uploadFile}
            disabled={!file || uploading}
          >
            {uploading ? "Ingesting..." : "Process File"}
          </button>
        </div>

        {message && (
          <div className="notification-banner">
            {message}
          </div>
        )}
      </aside>

      <main className="main-content">
        <div className="chat-window">
          {messages.length === 0 && !thinking && (
            <div className="workspace-welcome">
              <h2>Start Exploring</h2>
              <p>
                Upload a document on the left sidebar. Ask questions about the
                content, and retrieve answers with exact page numbers.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-row ${
                msg.sender === "user" ? "user-row" : "system-row"
              }`}
            >
              <div className="chat-bubble-group">
                <div className="chat-bubble">
                  <div className="chat-text">{msg.text}</div>
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-wrapper">
                    <div className="sources-list">
                      {msg.sources.map((src, idx) => {
                        if (src.pages === "web") {
                          return (
                            <a
                              key={idx}
                              href={src.source}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="source-pill web-source-pill"
                              title={src.source}
                              style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}
                            >
                              🌐 {src.title || "Web Link"}
                            </a>
                          );
                        }
                        return (
                          <span
                            key={idx}
                            className="source-pill"
                            title={`Document: ${src.source}`}
                            style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}
                          >
                            📄 Page {src.pages}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {thinking && (
            <div className="chat-row system-row">
              <div className="chat-bubble loading-bubble">
                <div className="typing-loader">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}



          <div ref={messagesEndRef}></div>
        </div>

        <div className="bottom-dock">
          <div className="input-box-wrapper">
            <input
              className="chat-field"
              type="text"
              placeholder="Search document content..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={thinking}
            />
            <button
              className="send-trigger"
              onClick={handleSend}
              disabled={!input.trim() || thinking}
            >
              Ask
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;