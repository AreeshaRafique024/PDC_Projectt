/**
 * ChatInput Component
 * Input field for sending messages
 */

import React, { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import "./ChatInput.css";

const ChatInput = ({ onSendMessage, disabled = false, selectedModel }) => {
  const [message, setMessage] = useState("");
  const [rerank, setRerank] = useState(true);
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message, rerank);
      setMessage("");
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        textareaRef.current.scrollHeight + "px";
    }
  }, [message]);

  return (
    <div className="chat-input-container">
      <form onSubmit={handleSubmit} className="chat-input-form">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            selectedModel
              ? "Type your message... (Shift+Enter for new line)"
              : "Please select a model first..."
          }
          disabled={disabled || !selectedModel}
          className="chat-input-textarea"
          rows={1}
        />
        <button
          type="submit"
          disabled={disabled || !message.trim() || !selectedModel}
          className="chat-input-button"
          title="Send message"
        >
          <Send className="icon" />
        </button>
      </form>
      <div className="chat-input-controls">
        <label className="rerank-checkbox">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
          />
          <span>Rerank</span>
        </label>
      </div>
      {selectedModel && (
        <div className="chat-input-info">
          <span className="model-indicator">Using: {selectedModel}</span>
        </div>
      )}
    </div>
  );
};

export default ChatInput;
