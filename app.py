from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI(
    title="Optimized Janus AI Server (CPU-Friendly)",
    description="24/7 Optimized Backend API running on Lightning.ai",
    version="2.1"
)

# CORS Middleware ताकि बाहरी ऐप्स या वेबसाइट्स कनेक्ट हो सकें
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "deepseek-ai/Janus-1B"

print("🔄 Initializing and loading CPU-Optimized Janus Model...")
model = None
tokenizer = None

try:
    # टोकनाइज़र लोड करना
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # CPU के लिए लो मेमोरी यूसेज के साथ मॉडल लोड करना
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # CPU स्पीड बढ़ाने के लिए PyTorch Dynamic Quantization (Q8/Q4 जैसा इफ़ेक्ट)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("✅ Model successfully quantized and loaded into RAM!")

except Exception as e:
    print(f"❌ Failed to load model: {e}")

# रिक्वेस्ट का डेटा स्ट्रक्चर
class TextQuery(BaseModel):
    prompt: str
    max_tokens: int = 100

# होम / हेल्थ चेक एंडपॉइंट
@app.get("/")
def home():
    return {
        "status": "Online",
        "model": MODEL_NAME,
        "mode": "CPU-Optimized Quantized Server",
        "message": "Janus AI Backend is live!"
    }

# टेक्स्ट जनरेशन एंडपॉइंट
@app.post("/generate/text")
def generate_text(query: TextQuery):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="AI Model is not initialized.")
    
    try:
        inputs = tokenizer(query.prompt, return_tensors="pt")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=query.max_tokens,
            do_sample=True,
            top_p=0.9
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "success": True,
            "prompt": query.prompt,
            "response": response_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
