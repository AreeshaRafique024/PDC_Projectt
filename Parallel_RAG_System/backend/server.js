/**
 * Main Server Entry Point
 * Parallel RAG System Backend
 */

// Load environment variables FIRST - before any other imports
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env file from current directory
dotenv.config({ path: join(__dirname, ".env") });

import express from "express";
import { corsMiddleware } from "./middleware/cors.js";
import { errorHandler, notFoundHandler } from "./middleware/errorHandler.js";
import chatRoutes from "./routes/chat.routes.js";

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(corsMiddleware);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Routes
app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Parallel RAG System API",
    version: "1.0.0",
    endpoints: {
      health: "/api/chat/health",
      models: "/api/chat/models",
      chat: "/api/chat",
    },
  });
});

app.use("/api/chat", chatRoutes);

// Error handlers (must be last)
app.use(notFoundHandler);
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🤖 Parallel RAG System - Backend Server                  ║
║                                                            ║
║   Server running on: http://localhost:${PORT}                 ║
║   Environment: ${
    process.env.NODE_ENV || "development"
  }                                 ║
║                                                            ║
║   API Endpoints:                                           ║
║   • Health Check:    GET  /api/chat/health                 ║
║   • List Models:     GET  /api/chat/models                 ║
║   • Send Message:    POST /api/chat                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
  `);
});

// Handle uncaught exceptions
process.on("uncaughtException", (error) => {
  console.error("Uncaught Exception:", error);
  process.exit(1);
});

process.on("unhandledRejection", (error) => {
  console.error("Unhandled Rejection:", error);
  process.exit(1);
});
