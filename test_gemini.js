const { GoogleGenerativeAI } = require('./eeg-alzheimer-blog/node_modules/@google/generative-ai');

async function testModels() {
  // Load API key from .env.local instead of hardcoding
  const fs = require('fs');
  const path = require('path');
  const envPath = path.join(__dirname, 'eeg-alzheimer-blog', '.env.local');
  const envContent = fs.readFileSync(envPath, 'utf8');
  const match = envContent.match(/GEMINI_API_KEY=(.+)/);
  const apiKey = match ? match[1].trim() : '';
  
  if (!apiKey) {
    console.error('❌ No API key found in .env.local');
    process.exit(1);
  }
  const genAI = new GoogleGenerativeAI(apiKey);
  
  // List of models to try
  const modelsToTry = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-exp', 
    'gemini-1.5-flash-latest',
    'gemini-1.5-pro-latest',
    'gemini-pro',
  ];
  
  console.log('Testing available Gemini models...\n');
  
  for (const modelName of modelsToTry) {
    try {
      console.log(`Testing ${modelName}...`);
      const model = genAI.getGenerativeModel({ model: modelName });
      const result = await model.generateContent('Say hello in 5 words or less');
      const response = await result.response;
      console.log(`✅ ${modelName} WORKS!`);
      console.log(`   Response: ${response.text().trim()}\n`);
      return modelName; // Return first working model
    } catch (error) {
      console.log(`❌ ${modelName} failed: ${error.status || error.message}\n`);
    }
  }
}

testModels().then(working => {
  if (working) console.log(`\n🎉 Use model: ${working}`);
});
