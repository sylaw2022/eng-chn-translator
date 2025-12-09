import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from custom_transformer import Transformer, create_causal_mask, create_padding_mask
import os

# Define request model
class TranslationRequest(BaseModel):
    text: str
    max_len: int = 128

class TranslationResponse(BaseModel):
    translation: str

app = FastAPI(title="Eng-Chn Translation API")

# Enable CORS for the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model, tokenizer
    
    MODEL_PATH = "model/final_model.pt"
    TOKENIZER_PATH = "model/final_model_tokenizer"
    
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise e

    # Initialize Model Architecture
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(device)
    
    # Load Weights
    try:
        # Load with map_location to handle CPU/GPU mismatch
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    text = request.text
    
    try:
        # Inference Logic
        # 1. Prepare Source
        src_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        src_mask = create_padding_mask(src_tokens, tokenizer.pad_token_id)
        
        # 2. Prepare Decoder Input (Start with PAD)
        decoder_input = torch.full((1, 1), tokenizer.pad_token_id, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Encode
            src_emb = model.src_emb(src_tokens) * torch.sqrt(torch.tensor(model.d_model, device=device))
            memory = model.encoder(src_emb, mask=src_mask)
            
            # Decode
            for _ in range(request.max_len):
                tgt_len = decoder_input.shape[1]
                causal_mask = create_causal_mask(tgt_len, device)
                tgt_pad_mask = create_padding_mask(decoder_input, tokenizer.pad_token_id)
                tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) & tgt_pad_mask
                
                tgt_emb = model.tgt_emb(decoder_input) * torch.sqrt(torch.tensor(model.d_model, device=device))
                output = model.decoder(tgt_emb, memory, src_mask=src_mask, tgt_mask=tgt_mask)
                logits = model.out_proj(output)
                
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                decoder_input = torch.cat((decoder_input, next_token), dim=1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # 3. Decode to Text
        translation_tokens = decoder_input[0, 1:]
        translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
        
        return {"translation": translation}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use port 8080 for Cloud Run default
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

