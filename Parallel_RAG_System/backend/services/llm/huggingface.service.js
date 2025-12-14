/**
 * HuggingFace Inference API Service
 * Handles all HuggingFace model interactions via OpenAI Compatibility Layer
 */

import { OpenAI } from "openai";
import { BaseLLMService } from "./base.service.js";

export class HuggingFaceService extends BaseLLMService {
  constructor(apiKey) {
    super(apiKey);
    this.client = apiKey
      ? new OpenAI({
          baseURL: "https://router.huggingface.co/v1",
          apiKey: apiKey,
        })
      : null;
  }

  /**
   * Generate response from HuggingFace model
   */
  async generateResponse(model, message, conversationHistory = []) {
    if (!this.hasApiKey()) {
      throw new Error("HuggingFace API key not configured");
    }

    try {
      // Prepare messages
      const messages = this.buildMessages(conversationHistory, message);

      // Check for image content in the message
      // If the message is an object with type 'image_url', we need to format it correctly for the API
      // However, the current frontend likely sends text strings.
      // If we want to support images, we need to check if 'message' is an array or object.
      // Based on user snippets, for VL models, content can be an array.

      const response = await this.client.chat.completions.create({
        model: model,
        messages: messages,
        max_tokens: 1024, // Default max tokens
        temperature: 0.7,
        stream: false,
      });

      return response.choices[0].message.content;
    } catch (error) {
      console.error(
        "HuggingFace API Error:",
        error.response?.data || error.message
      );
      throw new Error(`HuggingFace API Error: ${error.message}`);
    }
  }

  /**
   * Build messages array for OpenAI format
   */
  buildMessages(history, currentMessage) {
    const messages = [
      {
        role: "system",
        content:
          "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.",
      },
    ];

    // Add conversation history
    for (const msg of history) {
      messages.push({
        role: msg.role === "assistant" ? "assistant" : "user",
        content: msg.content,
      });
    }

    // Add current message
    // Check if currentMessage is a string or structured object (for images)
    // The frontend currently sends strings, but if we want to support images as per user request snippets:
    // We'll assume if it's a string, it's just text.
    // If the user provided code implies we might receive structured data, we should handle it.
    // For now, we'll treat it as content.

    messages.push({
      role: "user",
      content: currentMessage,
    });

    return messages;
  }
}
