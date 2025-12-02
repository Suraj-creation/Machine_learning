import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";
import { 
  geminiSystemPrompt, 
  generateTermPrompt, 
  generateCodePrompt,
  generateQuestionPrompt 
} from "@/lib/project-context";

// Initialize Gemini with error handling
const apiKey = process.env.GEMINI_API_KEY;
const genAI = apiKey ? new GoogleGenerativeAI(apiKey) : null;

// In-memory cache for server-side responses
const serverCache = new Map<string, { explanation: string; timestamp: number }>();
const CACHE_DURATION = 1000 * 60 * 60; // 1 hour

// Request queue and rate limiting
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 1200; // 1.2 seconds between requests
const requestQueue: Array<() => Promise<void>> = [];
let isProcessingQueue = false;

// Generate cache key
function getCacheKey(params: {
  term?: string;
  code?: string;
  question?: string;
  mode: string;
  sectionContext?: string;
}): string {
  const { term, code, question, mode, sectionContext } = params;
  const content = term || code || question || "";
  return `${mode}:${sectionContext || ""}:${content.toLowerCase().trim().substring(0, 100)}`;
}

// Get from cache
function getFromCache(key: string): string | null {
  const entry = serverCache.get(key);
  if (!entry) return null;
  
  const now = Date.now();
  if (now - entry.timestamp > CACHE_DURATION) {
    serverCache.delete(key);
    return null;
  }
  
  return entry.explanation;
}

// Set in cache
function setInCache(key: string, explanation: string): void {
  serverCache.set(key, {
    explanation,
    timestamp: Date.now(),
  });
  
  // Cleanup old entries (max 500)
  if (serverCache.size > 500) {
    const keysToDelete = Array.from(serverCache.keys()).slice(0, 100);
    keysToDelete.forEach(k => serverCache.delete(k));
  }
}

// Process request queue with rate limiting
async function processQueue(): Promise<void> {
  if (isProcessingQueue || requestQueue.length === 0) return;
  
  isProcessingQueue = true;
  
  while (requestQueue.length > 0) {
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime;
    
    if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
      await new Promise(resolve => 
        setTimeout(resolve, MIN_REQUEST_INTERVAL - timeSinceLastRequest)
      );
    }
    
    const request = requestQueue.shift();
    if (request) {
      lastRequestTime = Date.now();
      await request();
    }
    
    // Small additional delay
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  
  isProcessingQueue = false;
}

// Retry helper with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 5,
  baseDelay: number = 2000
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const isRateLimit = error instanceof Error && 
        (error.message.includes("429") || 
         error.message.includes("quota") || 
         error.message.includes("limit") ||
         error.message.includes("RESOURCE_EXHAUSTED"));
      
      if (isRateLimit && attempt < maxRetries - 1) {
        const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000; // Add jitter
        console.log(`Rate limited, retrying in ${Math.floor(delay)}ms (attempt ${attempt + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }
      throw error;
    }
  }
  throw new Error("Max retries exceeded");
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { term, surroundingText, sectionContext, mode, code, language, description, question } = body;

    // Validate API key
    if (!apiKey || !genAI) {
      console.error("Gemini API key not configured");
      return NextResponse.json(
        { error: "AI service not configured. Please add GEMINI_API_KEY to .env.local" },
        { status: 500 }
      );
    }

    // Check cache first
    const cacheKey = getCacheKey({ term, code, question, mode: mode || "term", sectionContext });
    const cached = getFromCache(cacheKey);
    
    if (cached) {
      console.log(`Cache hit for: ${cacheKey.substring(0, 50)}`);
      return NextResponse.json({ 
        explanation: cached,
        mode: mode || "term",
        term: term || null,
        cached: true,
      });
    }

    // Use Gemini Flash model (fast, good rate limits)
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash",
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 800, // Reduced for faster responses
      },
      systemInstruction: geminiSystemPrompt,
    });

    let userPrompt: string;

    // Generate appropriate prompt based on mode
    switch (mode) {
      case "code":
        // Code explanation mode
        if (!code) {
          return NextResponse.json(
            { error: "Code content is required for code explanation mode" },
            { status: 400 }
          );
        }
        userPrompt = generateCodePrompt(code, language || "python", description || "");
        break;
      
      case "question":
        // General question mode
        if (!question) {
          return NextResponse.json(
            { error: "Question is required for question mode" },
            { status: 400 }
          );
        }
        userPrompt = generateQuestionPrompt(question, sectionContext || "General");
        break;
      
      case "term":
      default:
        // Term explanation mode (default)
        if (!term) {
          return NextResponse.json(
            { error: "Term is required for term explanation mode" },
            { status: 400 }
          );
        }
        userPrompt = generateTermPrompt(
          term, 
          surroundingText || "", 
          sectionContext || "General"
        );
        break;
    }

    // Generate response with retry logic and rate limiting
    const generateResponse = async () => {
      const result = await model.generateContent(userPrompt);
      const response = await result.response;
      return response.text();
    };

    const text = await retryWithBackoff(generateResponse, 5, 2000);

    // Validate response
    if (!text || text.trim().length === 0) {
      throw new Error("Empty response from AI model");
    }

    // Store in cache
    setInCache(cacheKey, text);
    console.log(`Cached response for: ${cacheKey.substring(0, 50)}`);

    return NextResponse.json({ 
      explanation: text,
      mode: mode || "term",
      term: term || null,
      cached: false,
    });

  } catch (error) {
    console.error("Gemini API error:", error);
    
    // Provide more specific error messages
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    
    if (errorMessage.includes("API_KEY") || errorMessage.includes("invalid")) {
      return NextResponse.json(
        { error: "Invalid API key. Please check your GEMINI_API_KEY." },
        { status: 401 }
      );
    }
    
    if (errorMessage.includes("quota") || errorMessage.includes("limit") || errorMessage.includes("429") || errorMessage.includes("RESOURCE_EXHAUSTED")) {
      return NextResponse.json(
        { error: "API rate limit reached. Please wait a few seconds and try again." },
        { status: 429 }
      );
    }
    
    if (errorMessage.includes("blocked") || errorMessage.includes("safety")) {
      return NextResponse.json(
        { error: "Content was blocked by safety filters. Please try a different term." },
        { status: 400 }
      );
    }

    if (errorMessage.includes("not found") || errorMessage.includes("404")) {
      return NextResponse.json(
        { error: "AI model not available. Please try again later." },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { error: "Failed to generate explanation. Please try again." },
      { status: 500 }
    );
  }
}
