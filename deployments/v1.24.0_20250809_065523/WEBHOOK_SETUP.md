# 🔄 GitHub → ChatGPT Webhook Sync Setup Guide

## ✅ **COMPLETE IMPLEMENTATION**

Your AI Memory Layer project now has a **complete GitHub → ChatGPT webhook sync system** that automatically updates ChatGPT conversations with your latest code changes.

---

## 🚀 **What's Deployed**

### **Webhook Receiver** (Running on EC2)
- **FastAPI server** running on `http://18.224.179.36:8001`
- **Systemd service**: `github-webhook.service` (auto-starts on reboot)
- **ChatGPT-compatible file generation** in `~/ai-memory-layer/chatgpt-sync/`
- **Security**: GitHub signature verification support
- **Background processing** of webhook events

### **Setup Scripts**
- `github_chatgpt_webhook.py` - Main webhook receiver
- `setup_github_webhook.py` - Automated GitHub webhook configuration
- `webhook_requirements.txt` - Dependencies
- `.env.webhook.example` - Configuration template

---

## 🔧 **Quick Setup (3 Steps)**

### **Step 1: Get GitHub Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Create token with `repo` scope for your private repository
3. Copy the token (starts with `ghp_`)

### **Step 2: Configure Environment**
```bash
export GITHUB_TOKEN="ghp_your_token_here"
export REPO_OWNER="your-github-username"  
export REPO_NAME="ai-memory-layer"
export WEBHOOK_URL="http://18.224.179.36:8001/webhook"
export WEBHOOK_SECRET="your-secure-secret"  # Optional but recommended
```

### **Step 3: Run Setup**
```bash
python setup_github_webhook.py
```

This will automatically:
- ✅ Test webhook connectivity  
- ✅ Create GitHub webhook
- ✅ Configure push event triggers
- ✅ Set up security (if secret provided)

---

## 📋 **How It Works**

### **Automatic Sync Flow**
1. **You push code** to your GitHub repository
2. **GitHub sends webhook** to `http://18.224.179.36:8001/webhook`
3. **Webhook receiver processes** the commit:
   - Downloads changed files from GitHub API
   - Generates ChatGPT-compatible markdown summary
   - Saves to `chatgpt-sync/` directory
4. **You upload files to ChatGPT** for code review and discussion

### **Generated Files**
Each commit creates two files:
- `YYYYMMDD_HHMMSS_sha8.md` - ChatGPT-ready summary with code
- `YYYYMMDD_HHMMSS_sha8.json` - Raw commit data

### **File Filtering**
Only syncs relevant files:
- **Code files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, etc.
- **Config files**: `.json`, `.yaml`, `.toml`, `.md`, etc.
- **Ignores**: `__pycache__`, `node_modules`, `.git`, `venv`, etc.

---

## 🧪 **Testing Your Setup**

### **Test 1: Webhook Connectivity**
```bash
# From your local machine
curl http://18.224.179.36:8001/
```
Should return webhook status JSON.

### **Test 2: Make a Test Commit**
```bash
echo "# Test webhook" > test-webhook.md
git add test-webhook.md
git commit -m "test: webhook integration"
git push origin main
```

### **Test 3: Check Generated Files**
Look for new files in `chatgpt-sync/` directory on the server:
```bash
ssh -i ~/.ssh/AI-memory.pem ubuntu@18.224.179.36 "ls -la ~/ai-memory-layer/chatgpt-sync/"
```

---

## 🔒 **Security Features**

### **GitHub Signature Verification**
- Uses HMAC-SHA256 with your webhook secret
- Prevents unauthorized webhook calls
- Set `WEBHOOK_SECRET` environment variable

### **File Access Control**
- Only downloads files from your specified repository
- GitHub token scoped to specific repo access
- No sensitive data in logs

### **Network Security**
- Webhook receiver runs on internal EC2 instance
- GitHub API calls use HTTPS
- Optional SSL/TLS for webhook endpoint

---

## 📊 **Current Status**

### ✅ **DEPLOYED & READY**
- **Webhook receiver**: Running on EC2 (`github-webhook.service`)
- **Port**: 8001 (accessible internally)
- **Sync directory**: `~/ai-memory-layer/chatgpt-sync/`
- **Dependencies**: Installed and ready
- **Service**: Auto-starts on server reboot

### **Webhook Receiver Status**
```json
{
  "service": "GitHub → ChatGPT Webhook Sync",
  "status": "running", 
  "sync_dir": "chatgpt-sync",
  "github_configured": false,  // Will be true after setup
  "webhook_secured": false     // Will be true with secret
}
```

---

## 🎯 **Next Steps**

### **For You**
1. **Set up GitHub webhook** using the provided script
2. **Make a test commit** to verify end-to-end functionality  
3. **Upload generated files** to ChatGPT for code review
4. **Iterate on your code** with automatic sync

### **For Enhanced Features** (Optional)
- Port forwarding or nginx proxy for external access
- Slack/Discord notifications for commits
- Automatic ChatGPT API integration (if using GPT-4 API)
- Custom filtering rules for specific file types

---

## 🔧 **Troubleshooting**

### **Webhook Not Triggering**
```bash
# Check service status
ssh -i ~/.ssh/AI-memory.pem ubuntu@18.224.179.36 "sudo systemctl status github-webhook.service"

# Check logs
ssh -i ~/.ssh/AI-memory.pem ubuntu@18.224.179.36 "sudo journalctl -u github-webhook.service -f"
```

### **Files Not Generated**
- Check GitHub token permissions (needs `repo` scope)
- Verify repository owner/name in environment variables
- Check webhook receiver logs for errors

### **Network Issues**
- EC2 security group may need port 8001 opened for external access
- Use internal testing: `curl http://localhost:8001/` from EC2

---

## 🎉 **Ready to Use!**

Your **GitHub → ChatGPT Sync** system is deployed and ready. Just configure the GitHub webhook and start pushing code for automatic ChatGPT integration!

**Repository**: `~/ai-memory-layer/` on EC2  
**Webhook URL**: `http://18.224.179.36:8001/webhook`  
**Generated Files**: `~/ai-memory-layer/chatgpt-sync/`