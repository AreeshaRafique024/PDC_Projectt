/**
 * API Service
 * Handles all HTTP requests to the backend
 */

import axios from "axios";
import { API_ENDPOINTS } from "../config/api.config.js";

// Create axios instance with default config
const apiClient = axios.create({
  timeout: 60000, // 60 seconds
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Chat API Service
 */
export const chatAPI = {
  /**
   * Send a message to the selected model
   */
  sendMessage: async (modelId, message, conversationHistory, conversationId) => {
    try {
      const response = await apiClient.post(API_ENDPOINTS.CHAT, {
        modelId,
        message,
        conversationHistory,
        conversationId,
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  getSessions: async () => {
    try {
      const response = await apiClient.get(API_ENDPOINTS.CHAT_HISTORY);
      return response.data;
    } catch (error) {
      console.error("Error fetching sessions:", error);
      return { success: false, history: [] };
    }
  },

  getSession: async (sessionId) => {
    try {
      const response = await apiClient.get(`${API_ENDPOINTS.CHAT_HISTORY}/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  createSession: async () => {
    try {
      const response = await apiClient.post(API_ENDPOINTS.CHAT_HISTORY);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  clearHistory: async () => {
    try {
      const response = await apiClient.delete(API_ENDPOINTS.CHAT_HISTORY);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },
  
  deleteSession: async (sessionId) => {
    try {
      const response = await apiClient.delete(`${API_ENDPOINTS.CHAT_HISTORY}/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  /**
   * Get list of available models
   */
  getModels: async () => {
    try {
      const response = await apiClient.get(API_ENDPOINTS.MODELS);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  /**
   * Retrieve documents from the RAG pipeline (supports reranking)
   * payload: { query: string, rerank: boolean, rerank_k?: number }
   */
  retrieveDocuments: async (query, rerank = false, rerank_k = null) => {
    try {
      const payload = { query, rerank };
      if (rerank_k) payload.rerank_k = rerank_k;
      const response = await apiClient.post(API_ENDPOINTS.RAG_RETRIEVE, payload);
      return response.data;
    } catch (error) {
      console.error("RAG retrieve error:", error.response?.data || error.message);
      throw error.response?.data || error.message;
    }
  },

  /**
   * Check API health
   */
  checkHealth: async () => {
    try {
      const response = await apiClient.get(API_ENDPOINTS.HEALTH);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },
};

export default apiClient;
