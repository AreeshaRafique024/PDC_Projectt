/**
 * Configuration for all supported LLM models
 * Each model has a unique ID, display name, provider, and API endpoint configuration
 */

export const MODELS_CONFIG = {
  // DeepSeek Models
  "deepseek-v3.2-exp": {
    id: "deepseek-v3.2-exp",
    name: "DeepSeek V3.2 Exp",
    provider: "huggingface",
    model: "deepseek-ai/DeepSeek-V3.2-Exp:novita",
    category: "DeepSeek",
  },
  "deepseek-r1": {
    id: "deepseek-r1",
    name: "DeepSeek R1",
    provider: "huggingface",
    model: "deepseek-ai/DeepSeek-R1:novita",
    category: "DeepSeek",
  },
  // DeepSeek Distill Models
  "deepseek-r1-distill-qwen-32b": {
    id: "deepseek-r1-distill-qwen-32b",
    name: "DeepSeek R1 Distill Qwen 32B",
    provider: "huggingface",
    model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    category: "DeepSeek",
  },
  "deepseek-r1-distill-llama-70b": {
    id: "deepseek-r1-distill-llama-70b",
    name: "DeepSeek R1 Distill Llama 70B",
    provider: "huggingface",
    model: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    category: "DeepSeek",
  },

  // GPT-OSS Models
  "gpt-oss-120b": {
    id: "gpt-oss-120b",
    name: "GPT-OSS 120B",
    provider: "huggingface",
    model: "openai/gpt-oss-120b:groq",
    category: "OpenAI OSS",
  },
  "gpt-oss-20b": {
    id: "gpt-oss-20b",
    name: "GPT-OSS 20B",
    provider: "huggingface",
    model: "openai/gpt-oss-20b",
    category: "OpenAI OSS",
  },

  // Llama Models
  "llama-4-scout": {
    id: "llama-4-scout",
    name: "Llama 4 Scout 17B",
    provider: "huggingface",
    model: "meta-llama/Llama-4-Scout-17B-16E-Instruct:groq",
    category: "Llama",
  },
  "llama-3.3-70b": {
    id: "llama-3.3-70b",
    name: "Llama 3.3 70B Instruct",
    provider: "huggingface",
    model: "meta-llama/Llama-3.3-70B-Instruct:groq",
    category: "Llama",
  },
  "llama-3.2-1b": {
    id: "llama-3.2-1b",
    name: "Llama 3.2 1B Instruct",
    provider: "huggingface",
    model: "meta-llama/Llama-3.2-1B-Instruct:novita",
    category: "Llama",
  },
  "llama-3.2-3b": {
    id: "llama-3.2-3b",
    name: "Llama 3.2 3B Instruct",
    provider: "huggingface",
    model: "meta-llama/Llama-3.2-3B-Instruct:novita",
    category: "Llama",
  },
  "llama-3.1-8b": {
    id: "llama-3.1-8b",
    name: "Llama 3.1 8B Instruct",
    provider: "huggingface",
    model: "meta-llama/Llama-3.1-8B-Instruct:novita",
    category: "Llama",
  },
  "llama-4-maverick-fp8": {
    id: "llama-4-maverick-fp8",
    name: "Llama 4 Maverick 17B FP8",
    provider: "huggingface",
    model: "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:novita",
    category: "Llama",
  },
  "llama-4-maverick-groq": {
    id: "llama-4-maverick-groq",
    name: "Llama 4 Maverick 17B (Groq)",
    provider: "huggingface",
    model: "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq",
    category: "Llama",
  },

  // Mistral Models
  "mistral-7b-instruct": {
    id: "mistral-7b-instruct",
    name: "Mistral 7B Instruct v0.2",
    provider: "huggingface",
    model: "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    category: "Mistral",
  },

  // Qwen Models
  "qwen-3-vl-thinking": {
    id: "qwen-3-vl-thinking",
    name: "Qwen 3 VL 235B Thinking",
    provider: "huggingface",
    model: "Qwen/Qwen3-VL-235B-A22B-Thinking:novita",
    category: "Qwen",
  },
  "qwen-3-vl-instruct": {
    id: "qwen-3-vl-instruct",
    name: "Qwen 3 VL 235B Instruct",
    provider: "huggingface",
    model: "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    category: "Qwen",
  },
  "qwen-2.5-7b": {
    id: "qwen-2.5-7b",
    name: "Qwen 2.5 7B Instruct",
    provider: "huggingface",
    model: "Qwen/Qwen2.5-7B-Instruct:together",
    category: "Qwen",
  },
  "qwen-2.5-vl-7b": {
    id: "qwen-2.5-vl-7b",
    name: "Qwen 2.5 VL 7B Instruct",
    provider: "huggingface",
    model: "Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
    category: "Qwen",
  },

  // Anthropic Models - REMOVED
  // Google Models - REMOVED
};

/**
 * Get list of all available models grouped by category
 */
export const getModelsList = () => {
  const modelsList = Object.values(MODELS_CONFIG);
  return {
    all: modelsList,
    deepseek: modelsList.filter((m) => m.category === "DeepSeek"),
    openai: modelsList.filter((m) => m.category === "OpenAI OSS"),
    llama: modelsList.filter((m) => m.category === "Llama"),
    qwen: modelsList.filter((m) => m.category === "Qwen"),
    mistral: modelsList.filter((m) => m.category === "Mistral"),
  };
};

/**
 * Get model configuration by ID
 */
export const getModelConfig = (modelId) => {
  return MODELS_CONFIG[modelId] || null;
};
