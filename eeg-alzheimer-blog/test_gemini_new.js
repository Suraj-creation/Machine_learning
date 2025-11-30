#!/usr/bin/env node

/**
 * Gemini API Test Script
 * Tests the configured API key and available models
 * 
 * Usage: node test_gemini_new.js
 */

const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');

// Load API key from .env.local
function loadApiKey() {
  const envPath = path.join(__dirname, '.env.local');
  
  if (!fs.existsSync(envPath)) {
    console.error('❌ ERROR: .env.local file not found!');
    console.log('\n📝 Create .env.local with:');
    console.log('GEMINI_API_KEY=your_api_key_here\n');
    process.exit(1);
  }
  
  const envContent = fs.readFileSync(envPath, 'utf8');
  const match = envContent.match(/GEMINI_API_KEY=(.+)/);
  
  if (!match || !match[1] || match[1].includes('YOUR_NEW_API_KEY_HERE')) {
    console.error('❌ ERROR: GEMINI_API_KEY not configured in .env.local');
    console.log('\n📝 Get your API key from: https://aistudio.google.com/app/apikey');
    console.log('Then add it to .env.local:\n');
    console.log('GEMINI_API_KEY=AIzaSy...\n');
    process.exit(1);
  }
  
  return match[1].trim();
}

// Test a specific model
async function testModel(genAI, modelName) {
  try {
    console.log(`\n🧪 Testing ${modelName}...`);
    const model = genAI.getGenerativeModel({ model: modelName });
    
    const startTime = Date.now();
    const result = await model.generateContent('Say "Hello from Gemini!" in exactly 5 words.');
    const response = await result.response;
    const elapsed = Date.now() - startTime;
    
    const text = response.text().trim();
    
    console.log(`✅ ${modelName} WORKS!`);
    console.log(`   Response: "${text}"`);
    console.log(`   Time: ${elapsed}ms`);
    
    return { success: true, model: modelName, response: text, time: elapsed };
  } catch (error) {
    const status = error.status || 'unknown';
    const message = error.message || 'Unknown error';
    
    console.log(`❌ ${modelName} FAILED`);
    console.log(`   Status: ${status}`);
    console.log(`   Error: ${message.split('\n')[0]}`);
    
    // Provide specific guidance
    if (status === 403) {
      console.log(`   🔑 API key is blocked or leaked - get new key from:`);
      console.log(`      https://aistudio.google.com/app/apikey`);
    } else if (status === 401) {
      console.log(`   🔑 Invalid API key - check .env.local`);
    } else if (status === 429) {
      console.log(`   ⏰ Rate limit exceeded - wait 60 seconds`);
    } else if (status === 404 || message.includes('not found')) {
      console.log(`   📦 Model not available or deprecated`);
    }
    
    return { success: false, model: modelName, error: message, status };
  }
}

// Main test function
async function runTests() {
  console.log('🚀 Gemini API Test Script');
  console.log('═'.repeat(50));
  
  // Load API key
  let apiKey;
  try {
    apiKey = loadApiKey();
    console.log(`\n✅ API Key loaded: ${apiKey.substring(0, 20)}...${apiKey.substring(apiKey.length - 4)}`);
  } catch (error) {
    console.error(`❌ Failed to load API key: ${error.message}`);
    process.exit(1);
  }
  
  const genAI = new GoogleGenerativeAI(apiKey);
  
  // Models to test (ordered by preference)
  const modelsToTest = [
    'gemini-2.0-flash-exp',      // Latest experimental
    'gemini-2.0-flash',          // Current default
    'gemini-1.5-flash-latest',   // Stable fallback
    'gemini-1.5-flash',          // Stable
    'gemini-1.5-pro-latest',     // More capable
    'gemini-pro',                // Legacy
  ];
  
  console.log(`\n📋 Testing ${modelsToTest.length} models...`);
  
  const results = [];
  
  for (const modelName of modelsToTest) {
    const result = await testModel(genAI, modelName);
    results.push(result);
    
    // If we found a working model, we can stop early (optional)
    // Uncomment the next line to stop after first success:
    // if (result.success) break;
  }
  
  // Summary
  console.log('\n' + '═'.repeat(50));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(50));
  
  const working = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`\n✅ Working models: ${working.length}/${results.length}`);
  
  if (working.length > 0) {
    console.log('\n🎉 Recommended models:');
    working.forEach((r, i) => {
      console.log(`   ${i + 1}. ${r.model} (${r.time}ms)`);
    });
    
    console.log(`\n💡 Current configuration: gemini-2.0-flash`);
    
    const recommended = working[0];
    if (recommended.model !== 'gemini-2.0-flash') {
      console.log(`\n⚠️  Consider switching to: ${recommended.model}`);
      console.log(`   Edit app/api/gemini/route.ts line 58`);
    }
  }
  
  if (failed.length > 0) {
    console.log(`\n❌ Failed models: ${failed.length}/${results.length}`);
    failed.forEach(r => {
      console.log(`   - ${r.model}: ${r.status}`);
    });
  }
  
  // Final verdict
  console.log('\n' + '═'.repeat(50));
  
  if (working.length === 0) {
    console.log('❌ NO WORKING MODELS FOUND!');
    console.log('\n🔧 Action Required:');
    console.log('   1. Get new API key: https://aistudio.google.com/app/apikey');
    console.log('   2. Update .env.local with new key');
    console.log('   3. Run this script again\n');
    process.exit(1);
  } else {
    console.log('✅ GEMINI API IS WORKING!');
    console.log('\n🎯 Next steps:');
    console.log('   1. Restart dev server: npm run dev');
    console.log('   2. Open: http://localhost:3000');
    console.log('   3. Double-click any term for AI explanation\n');
    process.exit(0);
  }
}

// Run tests
runTests().catch(error => {
  console.error('\n💥 Unexpected error:', error.message);
  process.exit(1);
});
