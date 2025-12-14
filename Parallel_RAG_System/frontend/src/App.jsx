import { useState, useEffect, useRef } from "react";
import { MessageSquare, AlertCircle } from "lucide-react";
import ModelSelector from "./components/ModelSelector/ModelSelector";
import Message from "./components/Message/Message";
import ChatInput from "./components/ChatInput/ChatInput";
import Sidebar from "./components/Sidebar/Sidebar";
import { chatAPI } from "./services/api.service";
import "./App.css";

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [messages, setMessages] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [retrieving, setRetrieving] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const messagesEndRef = useRef(null);

  // Load available models and sessions on mount
  useEffect(() => {
    loadModels();
    loadSessions();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadModels = async () => {
    try {
      setModelsLoading(true);
      const response = await chatAPI.getModels();
      if (response.success) {
        setModels(response.models);
        // Auto-select first available model
        const firstAvailable = response.models.find((m) => m.available);
        if (firstAvailable) {
          setSelectedModel(firstAvailable.id);
        }
      }
    } catch (err) {
      console.error("Failed to load models:", err);
      setError("Failed to load models. Please check your backend connection.");
    } finally {
      setModelsLoading(false);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await chatAPI.getSessions();
      if (response.success) {
        setSessions(response.history);
      }
    } catch (err) {
      console.error("Failed to load sessions:", err);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      setIsLoading(true);
      const response = await chatAPI.getSession(sessionId);
      if (response.success) {
        setMessages(response.session.messages);
        setCurrentSessionId(sessionId);
      }
    } catch (err) {
      console.error("Failed to load session:", err);
      setError("Failed to load chat session.");
    } finally {
      setIsLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async (messageText, rerankFlag = false) => {
    if (!selectedModel || !messageText.trim()) return;

    const userMessage = {
      role: "user",
      content: messageText,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setRetrieving(false);
    setGenerating(false);
    setError(null);

    try {
      // Get conversation history (last 10 messages for context)
      const conversationHistory = messages.slice(-10);

      // If rerank requested, call RAG retrieve first and display intermediate results
      if (rerankFlag) {
        setRetrieving(true);
        try {
          const retrieveResp = await chatAPI.retrieveDocuments(messageText, true);
          if (retrieveResp && retrieveResp.results) {
            const retrievalMessage = {
              role: "assistant",
              type: "retrieval",
              results: retrieveResp.results,
              model: "RAG",
              timestamp: new Date().toISOString(),
            };
            setMessages((prev) => [...prev, retrievalMessage]);
          }
        } catch (err) {
          console.warn("RAG retrieve failed:", err);
          setError("RAG retrieval failed. See console for details.");
        } finally {
          setRetrieving(false);
        }
      }

      // Proceed to get LLM response
      setGenerating(true);
      const response = await chatAPI.sendMessage(
        selectedModel,
        messageText,
        conversationHistory,
        currentSessionId
      );

      if (response.success) {
        const assistantMessage = {
          role: "assistant",
          content: response.response,
          model: response.model,
          timestamp: new Date().toISOString(),
          metrics: response.metrics,
        };
        setMessages((prev) => [...prev, assistantMessage]);
        
        // If this was a new session (implicitly created by backend or first message), update session list
        if (!currentSessionId || response.conversationId !== currentSessionId) {
            setCurrentSessionId(response.conversationId);
            loadSessions(); // Reload list to show new title/session
        } else {
            // Update local session list title if needed (optional optimization)
            loadSessions();
        }
      } else {
        throw new Error(response.error || "Failed to get response");
      }
    } catch (err) {
      console.error("Error sending message:", err);
      setError(
        err.error || err.message || "Failed to send message. Please try again."
      );
      // Remove the user message if there was an error
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setGenerating(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentSessionId(null);
    setError(null);
  };

  const handleClearHistory = async () => {
    if (window.confirm("Are you sure you want to clear all chat history? This cannot be undone.")) {
      try {
        await chatAPI.clearHistory();
        setMessages([]);
        setSessions([]);
        setCurrentSessionId(null);
        setError(null);
      } catch (err) {
        console.error("Failed to clear history:", err);
        setError("Failed to clear history.");
      }
    }
  };

  const handleSelectSession = (sessionId) => {
    loadSession(sessionId);
  };

  const handleSelectModel = (model) => {
    setSelectedModel(model.id);
    setError(null);
  };

  const selectedModelName = models.find((m) => m.id === selectedModel)?.name;

  return (
    <div className="app">
      <Sidebar 
        sessions={sessions} 
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onClearHistory={handleClearHistory}
      />
      
      <div className="app-content">
        <header className="app-header">
          <div className="header-content">
            <div className="header-title">
              <MessageSquare className="logo-icon" />
              <div>
                <h1>Parallel RAG</h1>
                <p className="subtitle">
                  Compare the best AI models in real-time
                </p>
              </div>
            </div>
            <div className="header-actions">
              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                onSelectModel={handleSelectModel}
                loading={modelsLoading}
              />
            </div>
          </div>
        </header>

        <main className="app-main">
          {error && (
            <div className="error-banner">
              <AlertCircle className="icon" />
              <span>{error}</span>
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}

          <div className="messages-container">
            {messages.length === 0 && !modelsLoading ? (
              <div className="welcome-screen">
                <div className="welcome-icon">
                  <MessageSquare />
                </div>
                <h2>Welcome to Parallel RAG</h2>
                <p>
                  Select a model and start chatting with the world's most advanced
                  AI systems
                </p>
                <div className="features">
                  <div className="feature">
                    <span className="feature-icon">⚡</span>
                    <span>Real-time Responses</span>
                  </div>
                  <div className="feature">
                    <span className="feature-icon">📊</span>
                    <span>Detailed Metrics</span>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <Message key={index} message={message} isLoading={false} />
                ))}
                {retrieving && (
                  <Message
                    message={{ role: "assistant", content: "Retrieving..." }}
                    isLoading={true}
                  />
                )}
                {generating && (
                  <Message
                    message={{ role: "assistant", content: "" }}
                    isLoading={true}
                  />
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </main>

        <ChatInput
          onSendMessage={handleSendMessage}
          disabled={retrieving || generating || modelsLoading}
          selectedModel={selectedModelName}
        />
      </div>
    </div>
  );
}

export default App;
