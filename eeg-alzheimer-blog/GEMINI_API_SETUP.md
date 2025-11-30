# 🔑 Gemini API Setup Guide

## ⚠️ CRITICAL ERROR FOUND

**Error**: `[403 Forbidden] Your API key was reported as leaked. Please use another API key.`

**Cause**: The API key was exposed publicly (likely in GitHub or other public repository) and was automatically revoked by Google for security.

---

## ✅ How to Fix (5 minutes)

### Step 1: Get a New API Key

1. **Visit**: https://aistudio.google.com/app/apikey
2. **Sign in** with your Google account
3. Click **"Create API Key"**
4. Choose **"Create API key in new project"** (recommended) or select an existing project
5. **Copy** the generated API key (starts with `AIzaSy...`)

### Step 2: Update `.env.local`

1. Open: `c:\Users\Govin\Desktop\ML_dash\eeg-alzheimer-blog\.env.local`
2. Replace `YOUR_NEW_API_KEY_HERE` with your new API key:

```env
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Step 3: Restart the Development Server

```powershell
# Stop the current server (Ctrl+C in terminal)
# Then restart:
cd "c:\Users\Govin\Desktop\ML_dash\eeg-alzheimer-blog"
npm run dev
```

### Step 4: Test the API

```powershell
# Run the test script
node test_gemini_new.js
```

---

## 🔒 Security Best Practices

### ❌ DO NOT:
- Commit `.env.local` to Git
- Share API keys in public repositories
- Hardcode API keys in source code
- Post API keys in forums/Discord/Slack

### ✅ DO:
- Keep `.env.local` in `.gitignore` (already configured)
- Use environment variables for sensitive data
- Rotate API keys if exposed
- Use API key restrictions in Google Cloud Console

---

## 📋 Verify `.gitignore` Protection

The `.env.local` file should already be ignored. Verify:

```powershell
# Check if .env.local is in .gitignore
cat .gitignore | Select-String "env.local"
```

Expected output: `.env*.local`

---

## 🧪 Test Gemini Integration

After setting up the new API key, test with:

```powershell
# Quick API test
node -e "require('dotenv').config({ path: '.env.local' }); const { GoogleGenerativeAI } = require('@google/generative-ai'); const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY); genAI.getGenerativeModel({ model: 'gemini-2.0-flash' }).generateContent('Hello').then(r => console.log('✅ SUCCESS:', r.response.text())).catch(e => console.error('❌ ERROR:', e.message));"
```

---

## 🎯 Models Available

Current configuration uses: **`gemini-2.0-flash`**

**Alternative models** (if 2.0-flash has issues):
- `gemini-1.5-flash-latest` - Stable, fast
- `gemini-1.5-pro-latest` - More powerful
- `gemini-pro` - Legacy stable

To change model, edit `app/api/gemini/route.ts` line 58.

---

## 🐛 Common Errors & Solutions

### Error: `403 Forbidden - API key leaked`
**Solution**: Get new API key (see Step 1 above)

### Error: `401 Unauthorized - Invalid API key`
**Solution**: Check API key is correct in `.env.local`

### Error: `429 Too Many Requests - Rate limit`
**Solution**: Wait 60 seconds, or upgrade to paid plan

### Error: `404 Not Found - Model not available`
**Solution**: Change to `gemini-1.5-flash-latest` in route.ts

### Error: `AI service not configured`
**Solution**: Ensure `.env.local` exists with `GEMINI_API_KEY`

---

## 💰 Pricing (Free Tier)

**Gemini 2.0 Flash Free Tier:**
- ✅ **15 requests per minute (RPM)**
- ✅ **1 million tokens per day**
- ✅ **1,500 requests per day**

**More than enough** for development and testing!

Upgrade to paid: https://ai.google.dev/pricing

---

## 🔗 Useful Links

- **Get API Key**: https://aistudio.google.com/app/apikey
- **API Documentation**: https://ai.google.dev/docs
- **Pricing**: https://ai.google.dev/pricing
- **Model Garden**: https://ai.google.dev/models/gemini

---

## ✅ Quick Checklist

- [ ] Get new API key from https://aistudio.google.com/app/apikey
- [ ] Update `.env.local` with new key
- [ ] Verify `.env.local` is in `.gitignore`
- [ ] Restart dev server (`npm run dev`)
- [ ] Test by double-clicking any term in the blog
- [ ] Confirm API working (no 403 errors in console)

---

**After completing these steps, the Gemini AI features will work perfectly!** 🎉
