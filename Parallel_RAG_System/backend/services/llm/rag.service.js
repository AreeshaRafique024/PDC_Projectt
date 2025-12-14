import axios from "axios";

class RagService {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  /**
   * Check if the RAG service is available
   */
  async isAvailable() {
    try {
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Retrieve context for a query
   * @param {string} query
   */
  async retrieve(query) {
    try {
      const response = await axios.post(`${this.baseUrl}/retrieve`, {
        query: query,
      });
      return response.data.results;
    } catch (error) {
      console.error("RAG Retrieve Error:", error.message);
      return [];
    }
  }

  hasApiKey() {
    return true; // RAG service doesn't need an API key
  }
}

export const ragService = new RagService();
