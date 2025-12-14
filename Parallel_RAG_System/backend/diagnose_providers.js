
import dotenv from 'dotenv';
import axios from 'axios';
import { HfInference } from "@huggingface/inference";
import { MODELS_CONFIG } from './config/models.config.js';

dotenv.config();

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

async function diagnoseModels() {
  console.log('Starting deep diagnosis of top models...');
  
  const hfModels = Object.values(MODELS_CONFIG).filter(m => m.provider === 'huggingface');
  
  for (const model of hfModels) {
    console.log(`\n🔍 Investigating ${model.name} (${model.model})...`);
    
    try {
      // 1. Fetch available providers from HF API
      const url = `https://huggingface.co/api/models/${model.model}?expand[]=inferenceProviderMapping`;
      const response = await axios.get(url, {
        headers: { Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}` }
      });
      
      const mapping = response.data?.inferenceProviderMapping || {};
      const providers = Object.entries(mapping)
        .filter(([, info]) => info?.status === 'live')
        .map(([name, info]) => ({ name, ...info }));
        
      console.log(`   Found ${providers.length} live providers: ${providers.map(p => p.name).join(', ')}`);
      
      if (providers.length === 0) {
        console.log(`   ⚠️ No live providers found. Checking if base HF inference works...`);
        await testProvider(model.model, undefined, 'hf-inference');
        continue;
      }

      // 2. Test each provider specifically
      for (const provider of providers) {
        await testProvider(model.model, provider.name, provider.name);
      }
      
    } catch (error) {
      console.error(`   ❌ Error fetching model info: ${error.message}`);
    }
  }
}

async function testProvider(modelId, providerName, label) {
  process.stdout.write(`   Testing provider: ${label}... `);
  try {
    const response = await hf.chatCompletion({
      model: modelId,
      provider: providerName, // Explicitly request this provider
      messages: [{ role: "user", content: "Hello, are you working?" }],
      max_tokens: 10
    });
    console.log(`✅ SUCCESS!`);
  } catch (error) {
    console.log(`❌ Failed (${error.message.substring(0, 100)}...)`);
  }
}

diagnoseModels();
