/**
 * LLM Service Factory
 * Manages all LLM service instances and routes requests to appropriate providers
 */

import { HuggingFaceService } from "./huggingface.service.js";
import { getModelConfig } from "../../config/models.config.js";
import { ragService } from "./rag.service.js";

class LLMServiceFactory {
  constructor() {
    this.services = {};
    this.initializeServices();
  }

  /**
   * Initialize all LLM service instances
   */
  initializeServices() {
    // Debug: Log API key status
    console.log("🔑 Initializing LLM Services:");
    console.log(
      "  HuggingFace API Key:",
      process.env.HUGGINGFACE_API_KEY
        ? `✅ Loaded (${process.env.HUGGINGFACE_API_KEY.substring(0, 7)}...)`
        : "❌ Not found"
    );

    this.services = {
      huggingface: new HuggingFaceService(process.env.HUGGINGFACE_API_KEY),
    };
  }

  /**
   * Get appropriate service for a model
   */
  getService(modelId) {
    const modelConfig = getModelConfig(modelId);

    if (!modelConfig) {
      throw new Error(`Model ${modelId} not found in configuration`);
    }

    const service = this.services[modelConfig.provider];

    if (!service) {
      throw new Error(`Provider ${modelConfig.provider} not supported`);
    }

    if (!service.hasApiKey()) {
      throw new Error(`API key for ${modelConfig.provider} is not configured`);
    }

    return { service, modelConfig };
  }

  /**
   * Generate response from specified model
   */
  async generateResponse(modelId, message, conversationHistory = []) {
    try {
      const { service, modelConfig } = this.getService(modelId);

      console.log(
        `Generating response with ${modelConfig.name} (${modelConfig.provider})`
      );

      // RAG Integration: Retrieve context first
      console.log(`Retrieving context for: "${message}"`);
      const retrievedDocs = await ragService.retrieve(message);

      let finalMessage = message;
      let context = "";

      if (retrievedDocs && retrievedDocs.length > 0) {
        context = retrievedDocs.map((doc) => doc.content).join("\n\n");
        console.log(`Found ${retrievedDocs.length} documents for context.`);

        // Guardrail: Strict instruction
        finalMessage = `You are a helpful assistant. You must answer the user's question based ONLY on the context provided below. 
If the answer is not in the context, say "I cannot answer this based on the available information." 
Do not use your own knowledge.

Context:
${context}

Question: ${message}

Answer:`;
      } else {
        console.log("No context found via RAG.");
        // Guardrail for no context
        // Depending on requirements, we might want to return early or let the LLM answer generically but with a warning.
        // The user said "system now should respond from the content I have there... and not from it's original server."
        // So if no context, we should probably say we can't answer.
        return {
          success: true,
          response:
            "I cannot answer this question because there is no relevant information in the provided documents.",
          usage: null,
          model: modelConfig.name,
          modelId: modelId,
        };
      }

      const response = await service.generateResponse(
        modelConfig.model,
        finalMessage,
        conversationHistory
      );

      return {
        success: true,
        response: typeof response === "string" ? response : response.text,
        usage: typeof response === "object" ? response.usage : null,
        model: modelConfig.name,
        modelId: modelId,
        contextUsed: true, // Flag to indicate RAG was used
      };
    } catch (error) {
      console.error(
        `Error generating response with ${modelId}:`,
        error.message
      );
      return {
        success: false,
        error: error.message,
        modelId: modelId,
      };
    }
  }

  /**
   * Check which services are available (have valid API keys)
   */
  getAvailableServices() {
    const available = {};
    for (const [provider, service] of Object.entries(this.services)) {
      available[provider] = service.hasApiKey();
    }
    return available;
  }
}

// Export singleton instance
export const llmServiceFactory = new LLMServiceFactory();
