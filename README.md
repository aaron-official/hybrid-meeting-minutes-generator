---
title: Hybrid Meeting Minutes Generator
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: false
license: mit
tags:
- gradio
- meeting-minutes
- speech-recognition
- text-generation
- whisper
- llama
- openai
- gemini
- audio-processing
- nlp
models:
- openai/whisper-tiny
- openai/whisper-base
- openai/whisper-small
- Qwen/Qwen2.5-0.5B-Instruct
- HuggingFaceTB/SmolLM-1.7B-Instruct
- microsoft/Phi-3-mini-4k-instruct
short_description: AI tool that turns meeting audio into minutes
---

# ⚡🌐 Hybrid Meeting Minutes Generator

Transform your meeting recordings into professional, executive-level meeting minutes using cutting-edge AI. Choose between lightning-fast local processing for privacy or premium API models for maximum quality.

## 🚀 Key Features

### **🔥 Ultra-Fast Local Processing**
- **10x faster** than traditional models
- **100% private** - no data leaves your machine
- **Zero API costs** - completely free to run
- **Optimized models**: Whisper Tiny/Base/Small + Qwen2.5/SmolLM/Phi-3

### **🌐 Premium API Integration**
- **OpenAI**: Whisper-1 + GPT-4o-Mini
- **Google**: Gemini 2.5 Flash Preview
- **Enterprise quality** output
- **3-15 second** processing times

### **🌍 Multi-Language Support**
- English, French, German, Spanish
- Kiswahili, Italian, Russian
- Professional business terminology for each language

### **⚡ Performance Modes**
| Mode | Local Speed | Quality | Privacy | Cost |
|------|-------------|---------|---------|------|
| **Ultra Fast** | 15-30s | Good | 🔒 Private | Free |
| **Balanced** | 20-45s | Very Good | 🔒 Private | Free |
| **High Quality** | 30-90s | High | 🔒 Private | Free |
| **OpenAI API** | 5-15s | Excellent | Shared | ~$0.01-0.10 |
| **Google API** | 3-10s | Excellent | Shared | ~$0.01-0.05 |

## 📋 Generated Output Format

The system produces executive-level meeting minutes with:

- **Executive Summary**: Key outcomes and decisions
- **Attendee List**: Participants and their roles
- **Discussion Points**: Organized topic-by-topic
- **Key Decisions**: Formal resolutions and votes
- **Action Items**: Clear tasks with owners and deadlines
- **Next Steps**: Follow-up meetings and deliverables

## 🛠️ Quick Start

### **1. Local Processing (Recommended for First Use)**

1. Select "🚀 Local Ultra-Fast" for both audio and text
2. Choose your speed/quality preference
3. Upload your meeting audio file
4. Click "🚀 Generate Meeting Minutes"

**No API keys required!** Processing happens entirely on your machine.

### **2. API Processing (For Premium Quality)**

⚠️ **Security Warning**: Only use test/trial API keys. Never use production keys.

1. Get test API keys:
   - **OpenAI**: [Get test key](https://platform.openai.com/api-keys)
   - **Google**: [Get test key](https://ai.google.dev/)

2. Select API models in the interface
3. Enter your **TEST** API keys when prompted
4. **Immediately revoke keys** after testing

## 🔒 Security & Privacy

### **Local Processing Benefits**
- ✅ **100% Private**: Audio never leaves your machine
- ✅ **GDPR Compliant**: No data transmission
- ✅ **Enterprise Safe**: Perfect for confidential meetings
- ✅ **No Internet Required**: Works offline after setup

### **API Processing Considerations**
- ⚠️ **Data Sharing**: Audio/transcript sent to API providers
- ⚠️ **API Key Security**: Use test keys only in interface
- ⚠️ **Rate Limits**: API providers may limit usage
- ⚠️ **Costs**: Small charges per API call

### **Best Practices**
1. **Start with local models** for testing and sensitive content
2. **Use APIs** only for production-quality requirements
3. **Set environment variables** for production API keys
4. **Never commit** API keys to version control
5. **Regularly rotate** API keys

## 🎯 Use Cases

### **Corporate Environments**
- Board meetings and executive sessions
- Team standups and sprint planning
- Client calls and sales meetings
- Training sessions and workshops

### **Government & Municipal**
- City council meetings
- Public hearings and forums
- Committee sessions
- Administrative meetings

### **Academic & Research**
- Conference presentations
- Research group meetings
- Thesis defenses
- Academic conferences

### **International Business**
- Multi-language business meetings
- Global team collaborations
- International client calls
- Cross-cultural negotiations

## 📊 Performance Benchmarks

Tested on various hardware configurations:

### **Local Models Performance**
```
RTX 4090 (24GB VRAM):
├── Ultra Fast: 15-20 seconds (5min audio)
├── Balanced: 25-35 seconds (5min audio)  
└── Quality: 45-60 seconds (5min audio)

RTX 3060 (12GB VRAM):
├── Ultra Fast: 20-30 seconds (5min audio)
├── Balanced: 35-50 seconds (5min audio)
└── Quality: 60-90 seconds (5min audio)

CPU Only (16GB RAM):
├── Ultra Fast: 45-90 seconds (5min audio)
├── Balanced: 90-150 seconds (5min audio)
└── Quality: 150-300 seconds (5min audio)
```

### **API Models Performance**
```
OpenAI Whisper-1 + GPT-4o-Mini:
└── 5-15 seconds total (any audio length)

Google Gemini 2.5 Flash:
└── 3-10 seconds total (any audio length)
```

## 🛠️ Local Installation

If you want to run this locally or modify the code:

### **Prerequisites**
- Python 3.8+ (3.10+ recommended)
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM (16GB+ recommended for local models)

### **Setup**

```bash
# Clone the repository
git clone https://github.com/aaron-official/hybrid-meeting-minutes-generator.git
cd hybrid-meeting-minutes-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### **GPU Optimization (Recommended)**

For NVIDIA GPUs with CUDA support:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚨 Troubleshooting

### **Common Issues**

**GPU Out of Memory:**
```bash
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU mode or reduce model size in the interface settings.

**Model Download Fails:**
```bash
OSError: Unable to load weights
```
**Solution**: Check internet connection. Models are downloaded automatically on first use.

**Slow Processing on CPU:**
- Use "Ultra Fast" mode
- Reduce audio file length
- Consider using API models for better performance

**Poor Quality Output:**
- Use "Quality" mode for local models
- Try API models for best results
- Ensure clear audio input

## 📈 Roadmap

### **Coming Soon**
- [ ] **Speaker Diarization**: Identify individual speakers
- [ ] **Real-time Processing**: Live meeting transcription
- [ ] **Custom Templates**: User-defined minute formats
- [ ] **Export Options**: PDF, Word, email integration
- [ ] **Batch Processing**: Multiple files simultaneously

### **Future Enhancements**
- [ ] **Meeting Analytics**: Sentiment analysis, talk time
- [ ] **Integration APIs**: REST endpoints for enterprise
- [ ] **Mobile App**: iOS/Android applications
- [ ] **Cloud Deployment**: One-click cloud hosting

## 🤝 Contributing

We welcome contributions! To contribute:

### **Development Setup**

```bash
# Clone and setup development environment
git clone https://github.com/aaron-official/hybrid-meeting-minutes-generator.git
cd hybrid-meeting-minutes-generator

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Format code
black app.py
```

## 🙏 Acknowledgments

- **Hugging Face**: For providing the transformer models and hosting platform
- **OpenAI**: For Whisper and GPT models
- **Google**: For Gemini AI capabilities  
- **Gradio**: For the excellent web interface framework

---

**⭐ If this project helps you, please give it a star on GitHub!**

Made with ❤️ for better meetings everywhere.
