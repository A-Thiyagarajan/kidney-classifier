#!/bin/bash
# Quick Deployment Script
# Run this after verifying local setup

echo "==========================================="
echo "Kidney Classification - GitHub Push"
echo "==========================================="
echo ""

# 1. Check Git is initialized
if [ ! -d ".git" ]; then
    echo "[1] Initializing Git repository..."
    git init
    git add .
    git commit -m "Kidney classification model - production ready"
else
    echo "[1] Git repository already initialized"
fi

# 2. Verify files are staged
echo ""
echo "[2] Project status:"
git status

# 3. Push instructions
echo ""
echo "==========================================="
echo "NEXT STEPS"
echo "==========================================="
echo ""
echo "[3] Add GitHub remote:"
echo "    git remote add origin https://github.com/YOUR_USERNAME/Kidney-Classify.git"
echo ""
echo "[4] Push to GitHub:"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
echo "[5] Deploy on Render:"
echo "    1. Go to https://dashboard.render.com"
echo "    2. Click 'New +' > 'Web Service'"
echo "    3. Select 'Connect Repository'"
echo "    4. Choose your repository"
echo "    5. Configure:"
echo "       - Name: kidney-classify"
echo "       - Runtime: Python 3.10.13"
echo "       - Build: pip install -r requirements.txt"
echo "       - Start: gunicorn app:app"
echo "    6. Click 'Create Web Service'"
echo ""
echo "[6] Test after deployment:"
echo "    curl -X POST -F 'file=@c.jpg' https://kidney-classify.onrender.com/predict"
echo ""
echo "==========================================="
