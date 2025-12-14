import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import ExcelJS from "exceljs";
import { encode } from "gpt-3-encoder";
import si from "systeminformation";
import crypto from "crypto";
import natural from "natural";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, "../data");
const METRICS_FILE = path.join(DATA_DIR, "metrics.xlsx");

class MetricsService {
  constructor() {
    this.ensureDataDir();
    this.tokenizer = new natural.WordTokenizer();
  }

  ensureDataDir() {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }
  }

  estimateTokens(text) {
    if (!text) return 0;
    try {
      // Use natural for tokenization as requested for dynamic calculation
      return this.tokenizer.tokenize(text).length;
    } catch (e) {
      return 0;
    }
  }

  async getSystemMetrics() {
    try {
      const load = await si.currentLoad();
      const mem = await si.mem();
      return {
        cpu_percent: load.currentLoad.toFixed(2),
        ram_mb: (mem.active / 1024 / 1024).toFixed(2)
      };
    } catch (e) {
      console.error("Error getting system metrics:", e);
      return { cpu_percent: null, ram_mb: null };
    }
  }

  
  detectHallucination(prompt, response) {
    if (!response || !prompt) return { flag: false, rate: 0, reason: "insufficient_data" };
    
    const promptLower = prompt.toLowerCase().trim();
    const responseLower = response.toLowerCase().trim();

    // 1. Empty response check
    if (responseLower.length === 0) {
      return { flag: true, rate: 1.0, reason: "empty_response" };
    }

    // 2. Repetition check using Jaro-Winkler distance from natural library
    // If distance is very high (close to 1), it means high similarity
    const similarity = natural.JaroWinklerDistance(promptLower, responseLower);
    if (similarity > 0.95 && responseLower.length < promptLower.length * 1.5) {
       return { flag: true, rate: similarity, reason: "repetition_high_similarity" };
    }

    // 3. Gibberish check (Entropy based - simplified)
    // A very simple heuristic: if the ratio of unique words to total words is extremely low
    const tokens = this.tokenizer.tokenize(responseLower);
    if (tokens.length > 10) {
        const uniqueTokens = new Set(tokens);
        const ttr = uniqueTokens.size / tokens.length; // Type-Token Ratio
        if (ttr < 0.1) {
            return { flag: true, rate: 1.0 - ttr, reason: "gibberish_low_ttr" };
        }
    }

    return { flag: false, rate: 0, reason: "pass" };
  }

  async logMetrics(data) {
    try {
      const sysMetrics = await this.getSystemMetrics();
      // Use natural tokenizer for dynamic calculation if usage stats are missing
      const inputTokens = data.usage?.prompt_tokens || this.estimateTokens(data.prompt);
      const outputTokens = data.usage?.completion_tokens || this.estimateTokens(data.response);
      
      // Ensure latency is present, default to 0 if missing (though it should be passed)
      const latencyMs = data.latencyMs || 0;
      const latencySec = latencyMs / 1000;
      const throughput = latencySec > 0 ? (outputTokens / latencySec).toFixed(2) : 0;

      const hallucination = this.detectHallucination(data.prompt, data.response);

      const workbook = new ExcelJS.Workbook();
      let worksheet;

      const columns = [
          { header: "ID", key: "id", width: 36 },
          { header: "Timestamp (UTC)", key: "timestamp_utc", width: 25 },
          { header: "User ID", key: "user_id", width: 15 },
          { header: "Conversation ID", key: "conversation_id", width: 36 },
          { header: "Prompt", key: "prompt", width: 40 },
          { header: "Input Tokens", key: "input_tokens", width: 15 },
          { header: "Model Name", key: "model_name", width: 25 },
          { header: "Status Code", key: "status_code", width: 12 },
          { header: "Latency (ms)", key: "latency_ms", width: 15 },
          { header: "Response Text", key: "response_text", width: 40 },
          { header: "Output Tokens", key: "output_tokens", width: 15 },
          { header: "Error", key: "error", width: 20 },
          { header: "Throughput (t/s)", key: "throughput_tokens_per_sec", width: 18 },
          { header: "CPU %", key: "cpu_percent", width: 12 },
          { header: "RAM (MB)", key: "ram_mb", width: 15 },
          { header: "Hallucination Flag", key: "hallucination_flag", width: 18 },
          { header: "Hallucination Rate", key: "hallucination_rate", width: 18 },
          { header: "Req Size (Bytes)", key: "request_size_bytes", width: 15 },
          { header: "Res Size (Bytes)", key: "response_size_bytes", width: 15 },
          { header: "Notes", key: "notes", width: 20 },
      ];

      // Critical Fix: Properly load existing file to append
      if (fs.existsSync(METRICS_FILE)) {
        try {
            await workbook.xlsx.readFile(METRICS_FILE);
            worksheet = workbook.getWorksheet("Metrics");
            if (worksheet) {
                // ExcelJS doesn't save column definitions, so we must re-attach them
                // to allow adding rows by key-value pairs
                worksheet.columns = columns;
            }
        } catch (e) {
            console.error("Error reading existing metrics file, creating new one:", e);
        }
      }

      if (!worksheet) {
        worksheet = workbook.addWorksheet("Metrics");
        worksheet.columns = columns;

        // Style header
        worksheet.getRow(1).font = { bold: true, color: { argb: "FFFFFFFF" } };
        worksheet.getRow(1).fill = {
          type: "pattern",
          pattern: "solid",
          fgColor: { argb: "FF2563EB" }, // Blue background
        };
        worksheet.getRow(1).alignment = { horizontal: "center" };
      }

      const row = worksheet.insertRow(worksheet.rowCount + 1, {
        id: crypto.randomUUID(),
        timestamp_utc: new Date().toISOString(),
        user_id: data.userId || "anonymous",
        conversation_id: data.conversationId || "unknown",
        prompt: data.prompt || "",
        input_tokens: inputTokens,
        model_name: data.modelId,
        status_code: data.error ? 500 : 200,
        latency_ms: latencyMs,
        response_text: typeof data.response === 'string' ? data.response : JSON.stringify(data.response),
        output_tokens: outputTokens,
        error: data.error || "",
        throughput_tokens_per_sec: throughput,
        cpu_percent: sysMetrics.cpu_percent,
        ram_mb: sysMetrics.ram_mb,
        hallucination_flag: hallucination.flag,
        hallucination_rate: hallucination.rate,
        request_size_bytes: Buffer.byteLength(data.prompt || "", 'utf8'),
        response_size_bytes: Buffer.byteLength(data.response || "", 'utf8'),
        notes: hallucination.reason !== "pass" ? `Hallucination check: ${hallucination.reason}` : "",
      });

      // Highlight row borders
      row.eachCell((cell) => {
        cell.border = {
          top: { style: "thin" },
          left: { style: "thin" },
          bottom: { style: "thin" },
          right: { style: "thin" },
        };
      });

      // Atomic write
      const tempFile = METRICS_FILE + ".tmp";
      await workbook.xlsx.writeFile(tempFile);
      fs.renameSync(tempFile, METRICS_FILE);
      
    } catch (error) {
      console.error("Error logging metrics:", error);
    }
  }
}

export const metricsService = new MetricsService();

