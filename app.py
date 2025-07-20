import os
import requests
import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    BitsAndBytesConfig
)
from openai import OpenAI
import google.generativeai as genai
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SPEED-OPTIMIZED LOCAL MODEL CONSTANTS
SPEED_CONFIGS = {
    "ultra_fast": {
        "audio_model": "openai/whisper-tiny",           # 39M params, ~5x faster
        "text_model": "Qwen/Qwen2.5-0.5B-Instruct",   # 0.5B params, very fast
        "description": "Ultra Fast - Good for quick processing"
    },
    "balanced": {
        "audio_model": "openai/whisper-base",           # 74M params, ~3x faster  
        "text_model": "HuggingFaceTB/SmolLM-1.7B-Instruct", # 1.7B params, optimized
        "description": "Balanced - Good speed and quality"
    },
    "quality": {
        "audio_model": "openai/whisper-small",          # 244M params, ~2x faster
        "text_model": "microsoft/Phi-3-mini-4k-instruct", # 3.8B params, efficient
        "description": "High Quality - Best output, moderate speed"
    }
}

# API MODEL CONSTANTS
OPENAI_AUDIO_MODEL = "whisper-1"
OPENAI_TEXT_MODEL = "gpt-4o-mini"
GOOGLE_TEXT_MODEL = "gemini-2.5-flash-preview-04-17"

# Language options
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de", 
    "Spanish": "es",
    "Kiswahili": "sw",
    "Italian": "it",
    "Russian": "ru"
}

# Global variables for models
speech_model = None
processor = None
pipe = None
tokenizer = None
model = None
openai_client = None
google_client = None

def get_api_key_from_env_or_input(env_var, user_input):
    """Get API key from environment variable or user input."""
    env_key = os.getenv(env_var)
    return env_key if env_key else user_input

def get_optimal_device_config():
    """Detect hardware and return optimal configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        try:
            # Get VRAM info safely
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Available VRAM: {vram_gb:.1f}GB")
            
            # Conservative quantization for Spaces
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            
            return {
                "device": device,
                "torch_dtype": torch.float16,
                "quantization_config": quant_config,
                "device_map": "auto",
                "vram_gb": vram_gb
            }
        except Exception as e:
            logger.warning(f"Error getting CUDA info: {e}, falling back to CPU")
            device = "cpu"
    
    # CPU fallback
    return {
        "device": "cpu",
        "torch_dtype": torch.float32,
        "quantization_config": None,
        "device_map": None,
        "vram_gb": 0
    }

def init_ultra_fast_whisper(speed_config="balanced"):
    """Initialize ultra-fast Whisper model."""
    global speech_model, processor, pipe
    
    if pipe is not None:
        return True
        
    try:
        config = get_optimal_device_config()
        model_name = SPEED_CONFIGS[speed_config]["audio_model"]
        
        logger.info(f"Loading {model_name} on {config['device']}...")
        
        speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=config["torch_dtype"],
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        speech_model.to(config["device"])
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Optimized pipeline for speed
        pipe = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=config["torch_dtype"],
            device=config["device"],
            chunk_length_s=15,
            stride_length_s=2,
            return_timestamps=False,
            batch_size=1,
        )
        
        logger.info(f"✅ Whisper {model_name} loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading Whisper: {e}")
        return False

def init_ultra_fast_llm(speed_config="balanced"):
    """Initialize ultra-fast LLM."""
    global tokenizer, model
    
    if tokenizer is not None and model is not None:
        return True
        
    try:
        config = get_optimal_device_config()
        model_name = SPEED_CONFIGS[speed_config]["text_model"]
        
        logger.info(f"Loading {model_name} on {config['device']}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=config["device_map"],
            quantization_config=config["quantization_config"],
            torch_dtype=config["torch_dtype"],
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        if config["device"] == "cpu":
            model = model.to("cpu")
            
        logger.info(f"✅ LLM {model_name} loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading LLM: {e}")
        return False

def init_openai_client(api_key):
    """Initialize OpenAI client."""
    global openai_client
    try:
        openai_client = OpenAI(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        return False

def init_google_client(api_key):
    """Initialize Google Gemini client."""
    global google_client
    try:
        if not api_key:
            return False

        genai.configure(api_key=api_key)
        google_client = genai.GenerativeModel(GOOGLE_TEXT_MODEL)

        # Test the connection
        test_response = google_client.generate_content(
            "Hello",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.1
            )
        )
        return True
    except Exception as e:
        logger.error(f"Error initializing Google client: {e}")
        return False

def transcribe_ultra_fast(audio_file_path, speed_config="balanced"):
    """Ultra-fast transcription with optimizations."""
    global pipe
    
    if pipe is None:
        if not init_ultra_fast_whisper(speed_config):
            return "Error: Failed to initialize Whisper model"
    
    try:
        start_time = time.time()
        
        result = pipe(
            audio_file_path,
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
                "do_sample": False,
                "num_beams": 1,
                "length_penalty": 1.0,
                "repetition_penalty": 1.0,
            },
            return_timestamps=False
        )
        
        processing_time = time.time() - start_time
        
        if isinstance(result, dict) and "text" in result:
            text = result["text"].strip()
        else:
            text = str(result).strip()
            
        logger.info(f"⚡ Transcription completed in {processing_time:.2f}s")
        return text
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Error with transcription: {str(e)}"

def transcribe_with_openai(audio_file_path, api_key):
    """Transcribe audio using OpenAI Whisper API."""
    try:
        api_key = get_api_key_from_env_or_input("OPENAI_API_KEY", api_key)
        if not api_key:
            return "Error: No OpenAI API key provided"

        if not init_openai_client(api_key):
            return "Error: Failed to initialize OpenAI client"

        with open(audio_file_path, 'rb') as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model=OPENAI_AUDIO_MODEL,
                file=audio_file,
                response_format="text"
            )
        return transcription
    except Exception as e:
        logger.error(f"OpenAI transcription error: {e}")
        return f"Error with OpenAI transcription: {str(e)}"

def generate_ultra_fast(prompt, system_message, speed_config="balanced"):
    """Ultra-fast text generation with aggressive optimizations."""
    try:
        if not init_ultra_fast_llm(speed_config):
            return "Error: Failed to initialize LLM"
            
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        input_length = inputs.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1500,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_beams=1,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        response = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ Text generation completed in {processing_time:.2f}s")
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        return f"Error with text generation: {str(e)}"

def generate_with_openai(prompt, system_message, api_key):
    """Generate text using OpenAI GPT."""
    try:
        api_key = get_api_key_from_env_or_input("OPENAI_API_KEY", api_key)
        if not api_key:
            return "Error: No OpenAI API key provided"

        if not init_openai_client(api_key):
            return "Error: Failed to initialize OpenAI client"

        response = openai_client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI generation error: {e}")
        return f"Error with OpenAI generation: {str(e)}"

def generate_with_google(prompt, system_message, api_key):
    """Generate text using Google Gemini."""
    try:
        api_key = get_api_key_from_env_or_input("GOOGLE_API_KEY", api_key)
        if not api_key:
            return "Error: No Google API key provided"

        if not init_google_client(api_key):
            return "Error: Failed to initialize Google client"

        full_prompt = f"{system_message}\n\n{prompt}"

        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2000,
        )

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = google_client.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return response.text
    except Exception as e:
        logger.error(f"Google generation error: {e}")
        return f"Error with Google generation: {str(e)}"

def get_concise_system_message(language):
    """Streamlined system message for faster processing."""
    base_message = """You are an AI meeting minutes specialist. Transform the transcript into professional meeting minutes with this structure:

# MEETING MINUTES

## Summary
- Meeting purpose, date, key outcomes

## Attendees  
- List participants

## Key Points
- Main discussion topics in bullet points

## Decisions
- Formal decisions made

## Action Items
- Task | Owner | Due Date

Keep it concise and professional."""

    language_instructions = {
        "fr": " Respond in professional French.",
        "de": " Respond in professional German.", 
        "es": " Respond in professional Spanish.",
        "sw": " Respond in professional Kiswahili.",
        "it": " Respond in professional Italian.",
        "ru": " Respond in professional Russian.",
        "en": ""
    }
    
    return base_message + language_instructions.get(language, "")

def get_comprehensive_system_message(language):
    """Get comprehensive system message for API models (more detailed than local)."""
    base_message = """You are a professional multilingual meeting minutes specialist with expertise in transforming raw transcripts into polished, executive-level documentation. Your role is to create comprehensive meeting minutes that read as if written by an experienced administrative professional who was present at the meeting.

Your output should demonstrate:
- Professional business writing standards with clear, concise language
- Logical organizational structure that's easy to follow and reference
- Accurate capture of discussions, decisions, and commitments
- Proper formatting for professional distribution and archival
- Appropriate tone for the meeting context (corporate, municipal, nonprofit, etc.)

Transform casual speech into professional documentation while preserving all substantive content and maintaining the integrity of what was discussed."""

    language_instructions = {
        "fr": " Please respond in French (Français). Use professional French business terminology and format the meeting minutes according to French business standards.",
        "de": " Please respond in German (Deutsch). Use professional German business terminology and format the meeting minutes according to German business standards.",
        "es": " Please respond in Spanish (Español). Use professional Spanish business terminology and format the meeting minutes according to Spanish business standards.",
        "sw": " Please respond in Kiswahili. Use professional Kiswahili business terminology and format the meeting minutes according to appropriate business standards.",
        "it": " Please respond in Italian (Italiano). Use professional Italian business terminology and format the meeting minutes according to Italian business standards.",
        "ru": " Please respond in Russian (Русский). Use professional Russian business terminology and format the meeting minutes according to Russian business standards.",
        "en": ""
    }

    return base_message + language_instructions.get(language, "")

def create_concise_prompt(transcript, language):
    """Streamlined prompt for faster processing."""
    return f"""Transform this transcript into professional meeting minutes:

{transcript}

Keep it structured, professional, and concise."""

def create_comprehensive_prompt(transcript, language):
    """Create comprehensive prompt for API models."""
    base_prompt = f"""Transform the following transcript into comprehensive meeting minutes using professional business language. Structure your response as follows:

**MEETING MINUTES**

## Executive Summary
- Meeting type, date, time, and location
- Primary purpose and key outcomes
- Major decisions made

## Attendees
- List all participants with their titles/roles when available
- Note any absent members if mentioned

## Discussion Points
- Organize by agenda items or major topics
- Convert casual language to professional tone
- Eliminate redundancies while preserving key information
- Include relevant context and background

## Key Decisions & Resolutions
- Document all formal decisions made
- Include vote tallies if applicable
- Note any dissenting opinions or concerns raised

## Action Items
- **Task**: [Clear description]
- **Owner**: [Responsible party]
- **Due Date**: [Deadline if specified]
- **Status**: [New/In Progress/Pending]

## Next Steps
- Upcoming meetings or deadlines
- Follow-up items requiring attention

---

**Raw Transcript:**
{transcript}

**Instructions**: Maintain accuracy while enhancing readability. Use professional terminology appropriate for the meeting context. Ensure all action items have clear ownership and timelines where possible."""

    return base_prompt

def process_meeting_hybrid(audio_file_path, audio_model, text_model, speed_config, language_choice, openai_key, google_key, progress=gr.Progress()):
    """Hybrid processing with both local and API options."""
    try:
        if audio_file_path is None:
            return "Please upload an audio file first.", "", ""
            
        language_code = LANGUAGES.get(language_choice, "en")
        
        # Step 1: Audio Transcription
        progress(0.1, desc="Starting audio transcription...")
        
        if audio_model == "🚀 Local Ultra-Fast":
            config_info = SPEED_CONFIGS[speed_config]["description"]
            yield f"🚀 Transcribing locally ({config_info})...", "", ""
            transcript = transcribe_ultra_fast(audio_file_path, speed_config)
            
        elif audio_model == "🌐 OpenAI Whisper-1 (API)":
            yield "🌐 Transcribing with OpenAI API...", "", ""
            transcript = transcribe_with_openai(audio_file_path, openai_key)
            
        else:
            yield "Error: Invalid audio model selection", "", ""
            return

        if transcript.startswith("Error"):
            yield transcript, "", ""
            return

        progress(0.6, desc="Generating meeting minutes...")
        yield "✅ Transcription complete! Generating meeting minutes...", f"**Transcript:**\n\n{transcript}", ""

        # Step 2: Text Generation
        if text_model == "🚀 Local Ultra-Fast":
            system_message = get_concise_system_message(language_code)
            user_prompt = create_concise_prompt(transcript, language_code)
            minutes = generate_ultra_fast(user_prompt, system_message, speed_config)
            
        elif text_model == "🌐 OpenAI GPT-4o-Mini (API)":
            system_message = get_comprehensive_system_message(language_code)
            user_prompt = create_comprehensive_prompt(transcript, language_code)
            minutes = generate_with_openai(user_prompt, system_message, openai_key)
            
        elif text_model == "🌐 Google Gemini 2.5 Flash (API)":
            system_message = get_comprehensive_system_message(language_code)
            user_prompt = create_comprehensive_prompt(transcript, language_code)
            minutes = generate_with_google(user_prompt, system_message, google_key)
            
        else:
            yield "Error: Invalid text model selection", f"**Transcript:**\n\n{transcript}", ""
            return

        if minutes.startswith("Error"):
            yield minutes, f"**Transcript:**\n\n{transcript}", ""
            return

        progress(1.0, desc="Complete!")
        processing_type = "🚀 Local" if "Local" in audio_model and "Local" in text_model else "🌐 Hybrid" if "Local" in audio_model or "Local" in text_model else "🌐 API"
        yield minutes, f"**Transcript:**\n\n{transcript}", f"🎉 {processing_type} processing complete!"

    except Exception as e:
        logger.error(f"Processing error: {e}")
        yield f"Error: {str(e)}", "", ""

# Gradio Interface
def create_hybrid_interface():
    """Create hybrid interface with both local and API options."""
    
    with gr.Blocks(title="Hybrid Meeting Minutes Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚡🌐 Hybrid Meeting Minutes Generator")
        gr.Markdown("**Choose your perfect balance: Lightning-fast local processing OR premium API quality!**")
        
        # Security Warning
        with gr.Row():
            gr.Markdown("""
            ### ⚠️ **CRITICAL SECURITY WARNING**
            
            **🔒 API Key Safety Guidelines:**
            1. **NEVER use production API keys in this interface**
            2. **Use test/trial API keys only** 
            3. **Immediately revoke any API keys after testing**
            4. **For production: Set OPENAI_API_KEY and GOOGLE_API_KEY as environment variables**
            5. **Consider local models for sensitive/confidential meetings**
            
            **💡 Recommendation:** Use local models for privacy, APIs for maximum quality.
            """)

        with gr.Row():
            with gr.Column():
                # Model Selection
                gr.Markdown("### 🎛️ Model Configuration")
                
                audio_model = gr.Dropdown(
                    choices=[
                        "🚀 Local Ultra-Fast",
                        "🌐 OpenAI Whisper-1 (API)"
                    ],
                    value="🚀 Local Ultra-Fast",
                    label="Audio Transcription Model",
                    info="Local = Private & Free, API = Fastest & Premium Quality"
                )

                text_model = gr.Dropdown(
                    choices=[
                        "🚀 Local Ultra-Fast", 
                        "🌐 OpenAI GPT-4o-Mini (API)", 
                        "🌐 Google Gemini 2.5 Flash (API)"
                    ],
                    value="🚀 Local Ultra-Fast",
                    label="Text Generation Model",
                    info="Local = Private & Free, APIs = Best Quality"
                )

                # Speed configuration (only for local models)
                speed_config = gr.Dropdown(
                    choices=[
                        ("⚡ Ultra Fast - Quick processing", "ultra_fast"),
                        ("⚖️ Balanced - Good speed + quality", "balanced"), 
                        ("🎯 High Quality - Best local output", "quality")
                    ],
                    value="balanced",
                    label="Local Model Speed/Quality",
                    info="Only applies to local models",
                    visible=True
                )
                
                # Language selection
                language_choice = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Output Language for Meeting Minutes"
                )

                # API Key inputs (initially hidden)
                gr.Markdown("### 🔑 API Configuration (Test Keys Only!)")
                
                openai_key = gr.Textbox(
                    type="password",
                    placeholder="Enter TEST OpenAI API Key (will be revoked after testing)",
                    label="⚠️ TEST OpenAI API Key",
                    visible=False,
                    info="🚨 Use test keys only! Revoke immediately after testing!"
                )

                google_key = gr.Textbox(
                    type="password",
                    placeholder="Enter TEST Google API Key (will be revoked after testing)",
                    label="⚠️ TEST Google API Key", 
                    visible=False,
                    info="🚨 Use test keys only! Revoke immediately after testing!"
                )

                # Audio input
                gr.Markdown("### 🎙️ Audio Input")
                audio_file = gr.Audio(
                    sources=['upload', 'microphone'],
                    type='filepath',
                    label="Upload Meeting Audio"
                )

                generate_btn = gr.Button("🚀 Generate Meeting Minutes", variant="primary", size="lg")

            with gr.Column():
                # Status
                status = gr.Textbox(
                    label="Processing Status", 
                    value="Ready! Choose local for privacy or API for premium quality...", 
                    interactive=False
                )

                # Outputs
                gr.Markdown("### 📋 Generated Meeting Minutes")
                output_minutes = gr.Markdown(
                    value="Professional meeting minutes will appear here...",
                    height=400
                )

                with gr.Accordion("📝 Raw Transcript", open=False):
                    output_transcript = gr.Markdown(
                        value="Audio transcript will appear here...",
                        height=200
                    )

        # Performance comparison info
        with gr.Row():
            gr.Markdown("""
            ### ⚡ Performance & Quality Comparison:
            
            | Option | Speed | Quality | Privacy | Cost |
            |--------|-------|---------|---------|------|
            | 🚀 **Local Ultra-Fast** | 15-30s | Good | 100% Private | Free |
            | 🚀 **Local Balanced** | 20-45s | Very Good | 100% Private | Free |
            | 🚀 **Local Quality** | 30-90s | High | 100% Private | Free |
            | 🌐 **OpenAI API** | 5-15s | Excellent | Shared with OpenAI | ~$0.01-0.10 |
            | 🌐 **Google API** | 3-10s | Excellent | Shared with Google | ~$0.01-0.05 |
            
            **💡 Pro Tip:** Start with local models for testing, use APIs for production-quality outputs.
            """)

        def update_ui_visibility(audio_choice, text_choice):
            """Update UI visibility based on model selections."""
            needs_openai = "OpenAI" in audio_choice or "OpenAI" in text_choice
            needs_google = "Google" in text_choice
            has_local = "Local" in audio_choice or "Local" in text_choice
            
            return (
                gr.update(visible=needs_openai),  # openai_key
                gr.update(visible=needs_google),  # google_key
                gr.update(visible=has_local)      # speed_config
            )

        # Update visibility when models change
        for dropdown in [audio_model, text_model]:
            dropdown.change(
                fn=update_ui_visibility,
                inputs=[audio_model, text_model],
                outputs=[openai_key, google_key, speed_config]
            )

        # Connect the generate button
        generate_btn.click(
            fn=process_meeting_hybrid,
            inputs=[audio_file, audio_model, text_model, speed_config, language_choice, openai_key, google_key],
            outputs=[output_minutes, output_transcript, status]
        )

    return demo

# Launch the hybrid app
if __name__ == "__main__":
    logger.info("Starting Hybrid Meeting Minutes Generator...")
    try:
        demo = create_hybrid_interface()
        # Fix for Hugging Face Spaces compatibility
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            # show_error=True,  # Show errors in the interface
            quiet=False       # Enable detailed logging
        )
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        raise