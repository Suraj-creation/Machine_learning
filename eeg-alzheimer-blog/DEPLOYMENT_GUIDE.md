# 🚀 Vercel Deployment Guide

## ✅ YES - Auto-Rebuild on GitHub Push

**When you push to GitHub, Vercel will automatically:**
1. ✅ Detect the push
2. ✅ Pull latest code
3. ✅ Run `npm install`
4. ✅ Run `npm run build`
5. ✅ Deploy to the **same URL**

---

## 🔒 CRITICAL: Set Environment Variable on Vercel

### ⚠️ BEFORE PUSHING - Add API Key to Vercel

Your `.env.local` is NOT pushed to GitHub (it's gitignored). You **MUST** add the API key to Vercel's dashboard:

1. **Go to**: https://vercel.com/dashboard
2. Click your project (eeg-alzheimer-blog or similar)
3. Click **"Settings"** tab
4. Click **"Environment Variables"** in sidebar
5. Add new variable:
   - **Name**: `GEMINI_API_KEY`
   - **Value**: `AIzaSyDGINJH34Jh6TvBm3n2QfKU00E1H91u3GQ`
   - **Environment**: Check all (Production, Preview, Development)
6. Click **"Save"**

### 📸 Screenshot Path:
```
Dashboard → Your Project → Settings → Environment Variables → Add
```

---

## 📋 Step-by-Step Deployment Process

### Step 1: Verify Security ✅

```powershell
# 1. Check .env.local is gitignored
cd "c:\Users\Govin\Desktop\ML_dash\eeg-alzheimer-blog"
git check-ignore -v .env.local
# Should show: .gitignore:17:.env.local

# 2. Verify no API keys in tracked files
git grep -i "AIzaSy" -- '*.js' '*.ts' '*.tsx'
# Should return: NOTHING

# 3. Check what will be committed
git status
# Should NOT show .env.local
```

### Step 2: Commit Changes

```powershell
cd "c:\Users\Govin\Desktop\ML_dash\eeg-alzheimer-blog"

# Stage your changes (excluding .env.local)
git add .

# Check staged files (verify no .env.local)
git status

# Commit
git commit -m "feat: update Gemini AI integration with secure API key handling"

# Push to GitHub
git push origin main
```

### Step 3: Vercel Auto-Deploy

**Vercel will automatically:**
- Trigger build on push detection (~10 seconds)
- Install dependencies (~30 seconds)
- Build production bundle (~2 minutes)
- Deploy to your existing URL (~15 seconds)

**Total deployment time: ~3-4 minutes**

### Step 4: Verify Deployment

1. Check Vercel dashboard: https://vercel.com/dashboard
2. Look for build status (should show "Building..." then "Ready")
3. Once deployed, visit your site
4. Test AI feature: Double-click any term

---

## 🔍 Monitoring Deployment

### Via Vercel Dashboard
1. Go to: https://vercel.com/dashboard
2. Click your project
3. Click **"Deployments"** tab
4. See real-time build logs

### Via CLI (Optional)
```powershell
# Install Vercel CLI (one-time)
npm i -g vercel

# Login
vercel login

# Check deployments
vercel ls

# View logs
vercel logs <deployment-url>
```

---

## 🎯 Expected Deployment Output

```
Vercel CLI:
✓ Queued
✓ Building
✓ Uploading
✓ Deploying
✓ Ready! Deployed to production
```

**Your URLs:**
- **Production**: https://your-app.vercel.app
- **Preview**: https://your-app-git-branch.vercel.app (for other branches)

---

## 🔧 Troubleshooting

### Issue: Build fails with "GEMINI_API_KEY not configured"

**Solution:**
1. Add API key to Vercel environment variables (see above)
2. Redeploy: Click "Redeploy" in Vercel dashboard

### Issue: API returns 403 Forbidden

**Solution:**
1. API key is leaked/blocked
2. Get new key: https://aistudio.google.com/app/apikey
3. Update Vercel environment variables
4. Redeploy

### Issue: Changes not showing on production

**Solution:**
1. Clear browser cache (Ctrl+Shift+R)
2. Check Vercel deployment status
3. Verify correct branch is deployed (should be `main`)

### Issue: Build succeeds but AI not working

**Solution:**
1. Check browser console for errors (F12)
2. Verify API key is set in Vercel
3. Check API key has correct permissions
4. Test locally first: `npm run build && npm start`

---

## 📊 Deployment Checklist

**Before pushing:**
- [ ] API key added to Vercel dashboard
- [ ] `.env.local` is gitignored (verify with `git status`)
- [ ] No hardcoded API keys in code
- [ ] Build succeeds locally (`npm run build`)
- [ ] All changes committed (`git status` clean)

**Push command:**
```powershell
git add .
git commit -m "your message"
git push origin main
```

**After deployment:**
- [ ] Check Vercel build logs (green checkmark)
- [ ] Visit production URL
- [ ] Test AI feature (double-click any term)
- [ ] Check browser console for errors (F12)
- [ ] Test code explanation feature

---

## 🔄 Development Workflow

### For Each Update:

```powershell
# 1. Make changes locally
code .

# 2. Test locally
npm run dev
# Open http://localhost:3000

# 3. Build and verify
npm run build
npm start

# 4. Commit and push
git add .
git commit -m "feat: your feature description"
git push origin main

# 5. Wait for Vercel auto-deploy (3-4 min)
# 6. Test on production URL
```

---

## 🌿 Branch-Based Deployments

Vercel creates preview deployments for each branch:

```powershell
# Create feature branch
git checkout -b feature/new-feature

# Make changes and push
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Vercel creates preview URL:
# https://your-app-git-feature-new-feature.vercel.app
```

Merge to `main` when ready:
```powershell
git checkout main
git merge feature/new-feature
git push origin main
# Deploys to production URL
```

---

## 🔐 Security Best Practices

### ✅ DO:
- Use Vercel environment variables for secrets
- Keep `.env.local` in `.gitignore`
- Use different API keys for dev/prod
- Monitor API usage on Google Console

### ❌ DON'T:
- Commit `.env.local` to Git
- Hardcode API keys in source code
- Share API keys publicly
- Use same key for multiple projects

---

## 🆘 Emergency Rollback

If deployment breaks production:

### Via Vercel Dashboard:
1. Go to **Deployments** tab
2. Find last working deployment
3. Click **"..."** → **"Promote to Production"**

### Via CLI:
```powershell
vercel rollback <deployment-url>
```

---

## 📱 Vercel Dashboard Quick Links

- **Project Settings**: https://vercel.com/dashboard → Your Project → Settings
- **Environment Variables**: Settings → Environment Variables
- **Deployments**: Your Project → Deployments
- **Build Logs**: Deployments → Click deployment → "Building" tab
- **Analytics**: Your Project → Analytics (if enabled)

---

## ✅ Current Configuration

**Repository**: https://github.com/Suraj-creation/Machine_learning  
**Branch**: main  
**Framework**: Next.js 14.2.33  
**Build Command**: `npm run build`  
**Environment Variables Required**:
- `GEMINI_API_KEY` ← **Must be added to Vercel dashboard**

---

## 🎉 Ready to Deploy!

**Summary:**
1. ✅ Add `GEMINI_API_KEY` to Vercel dashboard
2. ✅ Verify `.env.local` is gitignored
3. ✅ Push to GitHub: `git push origin main`
4. ✅ Vercel auto-deploys in ~3-4 minutes
5. ✅ Test at your production URL

**That's it!** Every future push to `main` will automatically rebuild and deploy to the same URL. 🚀
