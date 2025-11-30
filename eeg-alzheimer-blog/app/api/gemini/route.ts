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

// Retry helper with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
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
        const delay = baseDelay * Math.pow(2, attempt);
        console.log(`Rate limited, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`);
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

    // Use Gemini 2.0 Flash model (fast, reliable, good rate limits)
    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.0-flash",
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 1024,
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

    // Generate response with retry logic for rate limits
    const generateResponse = async () => {
      const result = await model.generateContent(userPrompt);
      const response = await result.response;
      return response.text();
    };

    const text = await retryWithBackoff(generateResponse, 3, 1000);

    // Validate response
    if (!text || text.trim().length === 0) {
      throw new Error("Empty response from AI model");
    }

    return NextResponse.json({ 
      explanation: text,
      mode: mode || "term",
      term: term || null,
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
