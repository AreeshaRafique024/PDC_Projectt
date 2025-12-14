/**
 * Chat Routes
 * Handles all chat-related endpoints
 */

import express from "express";
import { llmServiceFactory } from "../services/llm/index.js";
import { getModelsList } from "../config/models.config.js";

import { chatHistoryService } from "../services/chatHistory.service.js";
import { metricsService } from "../services/metrics.service.js";

const router = express.Router();

/**
 * GET /api/chat/history
 * Get list of chat sessions
 */
router.get("/history", async (req, res) => {
  try {
    const sessions = await chatHistoryService.getSessions();
    res.json({ success: true, history: sessions });
  } catch (error) {
    console.error("Error fetching history:", error);
    res.status(500).json({ success: false, error: "Failed to fetch history" });
  }
});

/**
 * GET /api/chat/history/:id
 * Get specific session details
 */
router.get("/history/:id", async (req, res) => {
  try {
    const session = await chatHistoryService.getSession(req.params.id);
    if (!session) {
      return res.status(404).json({ success: false, error: "Session not found" });
    }
    res.json({ success: true, session });
  } catch (error) {
    console.error("Error fetching session:", error);
    res.status(500).json({ success: false, error: "Failed to fetch session" });
  }
});

/**
 * POST /api/chat/history
 * Create a new session
 */
router.post("/history", async (req, res) => {
  try {
    const session = await chatHistoryService.createSession();
    res.json({ success: true, session });
  } catch (error) {
    console.error("Error creating session:", error);
    res.status(500).json({ success: false, error: "Failed to create session" });
  }
});

/**
 * DELETE /api/chat/history
 * Clear all chat history
 */
router.delete("/history", async (req, res) => {
  try {
    await chatHistoryService.clearHistory();
    res.json({ success: true, message: "History cleared" });
  } catch (error) {
    console.error("Error clearing history:", error);
    res.status(500).json({ success: false, error: "Failed to clear history" });
  }
});

/**
 * DELETE /api/chat/history/:id
 * Delete specific session
 */
router.delete("/history/:id", async (req, res) => {
  try {
    await chatHistoryService.deleteSession(req.params.id);
    res.json({ success: true, message: "Session deleted" });
  } catch (error) {
    console.error("Error deleting session:", error);
    res.status(500).json({ success: false, error: "Failed to delete session" });
  }
});

/**
 * POST /api/chat
 * Send a message to selected LLM model
 */
router.post("/", async (req, res) => {
  const startTime = Date.now();
  try {
    const { modelId, message, conversationHistory = [], conversationId } = req.body;

    // Validate request
    if (!modelId) {
      return res.status(400).json({
        success: false,
        error: "Model ID is required",
      });
    }

    if (!message || message.trim() === "") {
      return res.status(400).json({
        success: false,
        error: "Message is required",
      });
    }

    // Create user message object
    const userMessage = {
      role: "user",
      content: message,
      timestamp: new Date().toISOString(),
    };

    // Generate response
    const result = await llmServiceFactory.generateResponse(
      modelId,
      message,
      conversationHistory
    );

    const latencyMs = Date.now() - startTime;

    if (result.success) {
      // Create assistant message object
      const assistantMessage = {
        role: "assistant",
        content: result.response,
        model: result.model,
        modelId: result.modelId,
        timestamp: new Date().toISOString(),
        metrics: {
          latencyMs,
          inputTokens: result.usage?.prompt_tokens,
          outputTokens: result.usage?.completion_tokens,
        }
      };

      // Save to history (session)
      // If conversationId is provided, use it. If not, create a new session implicitly?
      // Frontend should handle session creation, but for robustness:
      let sessionId = conversationId;
      if (!sessionId) {
        const newSession = await chatHistoryService.createSession();
        sessionId = newSession.id;
      }

      await chatHistoryService.saveMessage(sessionId, userMessage, assistantMessage);

      // Log metrics
      await metricsService.logMetrics({
        conversationId: sessionId,
        modelId: result.modelId,
        prompt: message,
        response: result.response,
        latencyMs,
        usage: result.usage,
      });

      res.json({
        ...result,
        conversationId: sessionId,
        metrics: assistantMessage.metrics
      });
    } else {
      // Log error metrics
      await metricsService.logMetrics({
        conversationId: conversationId || "unknown",
        modelId,
        prompt: message,
        error: result.error,
        latencyMs,
      });
      res.status(500).json(result);
    }
  } catch (error) {
    console.error("Chat route error:", error);
    const latencyMs = Date.now() - startTime;
    await metricsService.logMetrics({
      conversationId: req.body.conversationId || "unknown",
      modelId: req.body.modelId,
      prompt: req.body.message,
      error: error.message,
      latencyMs,
    });
    res.status(500).json({
      success: false,
      error: error.message || "Internal server error",
    });
  }
});

/**
 * GET /api/chat/models
 * Get list of all available models
 */
router.get("/models", (req, res) => {
  try {
    const models = getModelsList();
    const availableServices = llmServiceFactory.getAvailableServices();

    // Add availability status to each model
    const modelsWithStatus = models.all.map((model) => ({
      ...model,
      available: availableServices[model.provider] || false,
    }));

    res.json({
      success: true,
      models: modelsWithStatus,
      grouped: {
        deepseek: models.deepseek.map((m) => ({
          ...m,
          available: availableServices[m.provider] || false,
        })),
        openai: models.openai.map((m) => ({
          ...m,
          available: availableServices[m.provider] || false,
        })),
        llama: models.llama.map((m) => ({
          ...m,
          available: availableServices[m.provider] || false,
        })),
        qwen: models.qwen.map((m) => ({
          ...m,
          available: availableServices[m.provider] || false,
        })),
      },
      services: availableServices,
    });
  } catch (error) {
    console.error("Models route error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "Internal server error",
    });
  }
});

/**
 * GET /api/chat/health
 * Check API health and service availability
 */
router.get("/health", (req, res) => {
  const availableServices = llmServiceFactory.getAvailableServices();
  const totalServices = Object.keys(availableServices).length;
  const activeServices =
    Object.values(availableServices).filter(Boolean).length;

  res.json({
    success: true,
    status: "online",
    services: availableServices,
    summary: `${activeServices}/${totalServices} services available`,
  });
});

export default router;
