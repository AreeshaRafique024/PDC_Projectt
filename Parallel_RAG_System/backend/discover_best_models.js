
import dotenv from 'dotenv';
import { HfInference } from "@huggingface/inference";

dotenv.config();

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

const candidates = {
  "Llama 3.1": [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct"
  ],
  "Qwen 2.5": [
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"
  ],
  "Gemma 2": [
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it"
  ],
  "Mistral": [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-7B-Instruct-v0.3"
  ],
  "Phi 3.5": [
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-medium-128k-instruct"
  ],
  "DeepSeek": [
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct"
  ]
};

async function discover() {
  console.log('Discovering best available models...');
  const workingModels = {};

  for (const [family, models] of Object.entries(candidates)) {
    console.log(`\nTesting family: ${family}`);
    let found = false;
    
    for (const model of models) {
      process.stdout.write(`  Checking ${model}... `);
      try {
        const response = await hf.chatCompletion({
          model: model,
          messages: [{ role: "user", content: "Hi" }],
          max_tokens: 5
        });
        console.log(`✅ Works!`);
        workingModels[family] = model;
        found = true;
        break; // Stop after finding the best (first) working one
      } catch (error) {
        console.log(`❌ Failed`);
      }
    }
    
    if (!found) {
      console.log(`  ⚠️ No working models found for ${family}`);
    }
  }

  console.log('\nSummary of Best Working Models:');
  console.log(JSON.stringify(workingModels, null, 2));
}

discover();
