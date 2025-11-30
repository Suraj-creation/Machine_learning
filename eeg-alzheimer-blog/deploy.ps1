#!/usr/bin/env pwsh
# Quick Deployment Script for EEG Alzheimer Blog
# Usage: .\deploy.ps1 "your commit message"

param(
    [string]$CommitMessage = "Update blog content"
)

Write-Host "🚀 EEG Alzheimer Blog - Deployment Script" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Set location
Set-Location $PSScriptRoot

# Step 1: Security Check
Write-Host "`n🔒 Step 1: Security Check..." -ForegroundColor Yellow
$gitStatus = git status --porcelain
if ($gitStatus -match "\.env\.local") {
    Write-Host "❌ ERROR: .env.local is staged for commit!" -ForegroundColor Red
    Write-Host "This file contains your API key and should NOT be committed." -ForegroundColor Red
    exit 1
}

$envCheck = git check-ignore .env.local 2>&1
if (-not $envCheck) {
    Write-Host "⚠️  WARNING: .env.local might not be gitignored!" -ForegroundColor Yellow
}

# Search for hardcoded API keys
$apiKeySearch = git grep -i "AIzaSy" -- "*.js" "*.ts" "*.tsx" 2>&1
if ($apiKeySearch -and $apiKeySearch -notmatch "fatal") {
    Write-Host "❌ ERROR: Found hardcoded API keys in tracked files!" -ForegroundColor Red
    Write-Host $apiKeySearch -ForegroundColor Red
    exit 1
}

Write-Host "✅ Security check passed - no API keys in tracked files" -ForegroundColor Green

# Step 2: Build Test
Write-Host "`n🔨 Step 2: Testing build..." -ForegroundColor Yellow
npm run build 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed! Fix errors before deploying." -ForegroundColor Red
    npm run build
    exit 1
}
Write-Host "✅ Build successful" -ForegroundColor Green

# Step 3: Git Status
Write-Host "`n📝 Step 3: Checking changes..." -ForegroundColor Yellow
$status = git status --short
if (-not $status) {
    Write-Host "⚠️  No changes to commit" -ForegroundColor Yellow
    exit 0
}

Write-Host "Changes to be committed:" -ForegroundColor Cyan
git status --short

# Step 4: Confirm Deployment
Write-Host "`n⚠️  Are you sure you want to deploy to production?" -ForegroundColor Yellow
Write-Host "Commit message: '$CommitMessage'" -ForegroundColor Cyan
$confirm = Read-Host "Continue? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "❌ Deployment cancelled" -ForegroundColor Red
    exit 0
}

# Step 5: Commit and Push
Write-Host "`n📤 Step 4: Committing and pushing..." -ForegroundColor Yellow
git add .
git commit -m $CommitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Commit failed" -ForegroundColor Red
    exit 1
}

git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green

# Step 6: Monitor Deployment
Write-Host "`n🔍 Step 5: Monitoring deployment..." -ForegroundColor Yellow
Write-Host "Vercel is now building your app..." -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Check deployment status:" -ForegroundColor Cyan
Write-Host "   https://vercel.com/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "⏱️  Estimated deployment time: 3-4 minutes" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔔 You'll receive an email when deployment completes" -ForegroundColor Cyan

# Step 7: Final Checklist
Write-Host "`n📋 Post-Deployment Checklist:" -ForegroundColor Yellow
Write-Host "   [ ] Check Vercel dashboard for green checkmark" -ForegroundColor White
Write-Host "   [ ] Visit your production URL" -ForegroundColor White
Write-Host "   [ ] Test AI features (double-click any term)" -ForegroundColor White
Write-Host "   [ ] Check browser console for errors (F12)" -ForegroundColor White
Write-Host ""

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🎉 Deployment initiated successfully!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
