import React from "react";
import { MessageSquare, Trash2, Plus } from "lucide-react";
import "./Sidebar.css";

const Sidebar = ({ sessions, currentSessionId, onSelectSession, onNewChat, onClearHistory }) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-button" onClick={onNewChat}>
          <Plus size={16} />
          <span>New Chat</span>
        </button>
      </div>

      <div className="sidebar-content">
        <div className="history-label">Chat History</div>
        <div className="history-list">
          {sessions.length === 0 ? (
            <div className="empty-history">No history yet</div>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                className={`history-item ${currentSessionId === session.id ? "active" : ""}`}
                onClick={() => onSelectSession(session.id)}
              >
                <MessageSquare size={14} className="history-icon" />
                <span className="history-text">{session.title || "New Chat"}</span>
              </button>
            ))
          )}
        </div>
      </div>

      <div className="sidebar-footer">
        <button className="clear-history-button" onClick={onClearHistory}>
          <Trash2 size={16} />
          <span>Clear All History</span>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
