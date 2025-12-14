import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, "../../data");
const HISTORY_FILE = path.join(DATA_DIR, "chats.json");

class ChatHistoryService {
  constructor() {
    this.ensureDataDir();
  }

  async ensureDataDir() {
    try {
      await fs.mkdir(DATA_DIR, { recursive: true });
      try {
        await fs.access(HISTORY_FILE);
      } catch {
        await fs.writeFile(HISTORY_FILE, JSON.stringify([]));
      }
    } catch (error) {
      console.error("Error initializing chat history:", error);
    }
  }

  async getSessions() {
    try {
      const data = await fs.readFile(HISTORY_FILE, "utf-8");
      return JSON.parse(data);
    } catch (error) {
      console.error("Error reading chat history:", error);
      return [];
    }
  }

  async getSession(sessionId) {
    const sessions = await this.getSessions();
    return sessions.find((s) => s.id === sessionId) || null;
  }

  async createSession(title = "New Chat") {
    const sessions = await this.getSessions();
    const newSession = {
      id: crypto.randomUUID(),
      title,
      created_at: new Date().toISOString(),
      messages: [],
    };
    sessions.unshift(newSession);
    await this.saveSessions(sessions);
    return newSession;
  }

  async saveMessage(sessionId, userMessage, assistantMessage) {
    try {
      const sessions = await this.getSessions();
      const sessionIndex = sessions.findIndex((s) => s.id === sessionId);

      if (sessionIndex === -1) {
        // If session doesn't exist, create it (fallback)
        const newSession = await this.createSession(userMessage.content.substring(0, 30) + "...");
        newSession.messages.push(userMessage);
        newSession.messages.push(assistantMessage);
        // Re-read sessions to ensure we have the latest state if needed, but here we just modify the list
        // Actually, createSession saves to file, so we should re-read or just push to our local list if we were careful.
        // Let's simplify: just call createSession which saves, then getSessions again.
        // Optimization: just modify the sessions array we have.
        // But createSession modifies the file.
        // Let's just do it cleanly.
        const updatedSessions = await this.getSessions();
        const newSessionIndex = updatedSessions.findIndex(s => s.id === newSession.id);
        updatedSessions[newSessionIndex].messages.push(userMessage);
        updatedSessions[newSessionIndex].messages.push(assistantMessage);
        await this.saveSessions(updatedSessions);
        return updatedSessions[newSessionIndex];
      }

      sessions[sessionIndex].messages.push(userMessage);
      sessions[sessionIndex].messages.push(assistantMessage);
      
      // Update title if it's the first message and title is generic
      if (sessions[sessionIndex].messages.length === 2 && sessions[sessionIndex].title === "New Chat") {
        sessions[sessionIndex].title = userMessage.content.substring(0, 30) + (userMessage.content.length > 30 ? "..." : "");
      }

      await this.saveSessions(sessions);
      return sessions[sessionIndex];
    } catch (error) {
      console.error("Error saving chat history:", error);
      throw error;
    }
  }

  async saveSessions(sessions) {
    await fs.writeFile(HISTORY_FILE, JSON.stringify(sessions, null, 2));
  }

  async clearHistory() {
    try {
      await fs.writeFile(HISTORY_FILE, JSON.stringify([]));
      return [];
    } catch (error) {
      console.error("Error clearing chat history:", error);
      throw error;
    }
  }
  
  async deleteSession(sessionId) {
    const sessions = await this.getSessions();
    const filtered = sessions.filter(s => s.id !== sessionId);
    await this.saveSessions(filtered);
    return filtered;
  }
}

export const chatHistoryService = new ChatHistoryService();
