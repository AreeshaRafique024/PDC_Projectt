import { fileURLToPath } from "url";
import { dirname, join } from "path";
import dotenv from "dotenv";

import { HuggingFaceService } from "../services/llm/huggingface.service.js";
import { MODELS_CONFIG } from "../config/models.config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config({ path: join(__dirname, "..", ".env") });

const hfModels = Object.values(MODELS_CONFIG).filter(
  (model) => model.provider === "huggingface"
);

if (!process.env.HUGGINGFACE_API_KEY) {
  console.error("HUGGINGFACE_API_KEY is not configured in backend/.env");
  process.exit(1);
}

const service = new HuggingFaceService(process.env.HUGGINGFACE_API_KEY);

async function runTests() {
  console.log(`\n🔍 Testing ${hfModels.length} HuggingFace models...`);
  const results = [];

  for (const model of hfModels) {
    const prompt =
      "Respond with: HF test OK - " +
      model.name +
      " and mention the provider name only.";

    process.stdout.write(`\n▶️  ${model.name} (${model.model}) ... `);
    const start = Date.now();

    try {
      const response = await service.generateResponse(model.model, prompt, []);
      const duration = ((Date.now() - start) / 1000).toFixed(1);
      console.log(`✅ ${duration}s`);
      results.push({
        model: model.id,
        success: true,
        response: response.slice(0, 120),
      });
    } catch (error) {
      const duration = ((Date.now() - start) / 1000).toFixed(1);
      console.log(`❌ ${duration}s - ${error.message}`);
      results.push({ model: model.id, success: false, error: error.message });
    }
  }

  const successCount = results.filter((r) => r.success).length;
  const failureCount = results.length - successCount;

  console.log(`\nSummary: ${successCount} succeeded, ${failureCount} failed.`);
  if (failureCount > 0) {
    console.log("\n❌ Failures:");
    for (const failure of results.filter((r) => !r.success)) {
      console.log(` - ${failure.model}: ${failure.error}`);
    }
  }
}

runTests().catch((err) => {
  console.error("Unexpected error while testing models:", err);
  process.exit(1);
});
