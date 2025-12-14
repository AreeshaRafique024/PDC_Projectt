/**
 * Base LLM Service Class
 * Abstract class that all LLM providers must extend
 */

export class BaseLLMService {
  constructor(apiKey) {
    if (this.constructor === BaseLLMService) {
      throw new Error(
        "BaseLLMService is an abstract class and cannot be instantiated directly"
      );
    }
    this.apiKey = apiKey;
  }

  /**
   * Generate chat completion
   * Must be implemented by child classes
   * @param {string} model - Model identifier
   * @param {string} message - User message
   * @param {Array} conversationHistory - Previous messages
   * @returns {Promise<string>} - Model response
   */
  async generateResponse(model, message, conversationHistory = []) {
    throw new Error("generateResponse() must be implemented by child class");
  }

  /**
   * Validate API key
   * @returns {boolean}
   */
  hasApiKey() {
    return !!this.apiKey && this.apiKey !== "your_api_key_here";
  }

  /**
   * Format error message
   * @param {Error} error
   * @returns {string}
   */
  formatError(error) {
    if (error.response) {
      return `API Error: ${error.response.status} - ${JSON.stringify(
        error.response.data
      )}`;
    }

    if (error?.request?.url) {
      return `API Error: ${error.message} (${error.request.url})`;
    }

    if (error?.cause?.message) {
      return error.cause.message;
    }

    return error.message || "Unknown error occurred";
  }
}
