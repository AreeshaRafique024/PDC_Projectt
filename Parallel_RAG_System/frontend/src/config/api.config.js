// API Configuration
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/chat/health`,
  MODELS: `${API_BASE_URL}/api/chat/models`,
  CHAT: `${API_BASE_URL}/api/chat`,
  CHAT_HISTORY: `${API_BASE_URL}/api/chat/history`,
  RAG_RETRIEVE: `http://localhost:8000/retrieve`,
};

export default API_BASE_URL;
