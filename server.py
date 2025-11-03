from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as hf_version
from transformers.utils import logging
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import torch, os, uvicorn, re, os

# Quiet HF noise
logging.set_verbosity_error()
logging.disable_progress_bar()

APP_VERSION = "0.3"

app = FastAPI(title="Veil Mini LLM v2", version=APP_VERSION)

# ---------- CORS ----------
_default_origins = [
    "https://www.crynance.app",
    "https://crynance.app",
    "http://localhost:5173",
    "http://localhost:3000",
]
origins = os.environ.get("ALLOWED_ORIGINS", ",".join(_default_origins)).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # cache the preflight for a day
)
# -----------------------------------------------------

# -------- Model load --------
MODEL_DIR = os.environ.get("MODEL_DIR", "./veil_mini_model_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, int(os.environ.get("CPU_THREADS", "1"))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()

# pad/eos
eos_id = tokenizer.eos_token_id
model.config.pad_token_id = eos_id

SYSTEM = (
    "You are Eliara, a Guide AI from Veil. Answer in 2–4 concrete sentences. "
    "If the question lacks Veil context, reply exactly: 'Not enough context.'\n"
)

class Prompt(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None
    sample: Optional[bool] = None  # optional: allow light sampling

def _clean(text: str, prompt: str) -> str:
    # Remove echoed prompt
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Stop if the model tries to begin a new user turn
    for m in ["\nUser:", "\nYou:", "\nHuman:", "\nQ:"]:
        i = text.find(m)
        if i != -1:
            text = text[:i].strip()

    # Trim to the last sentence end
    m = re.search(r'([.?!])[^.?!]*$', text)
    if m:
        text = text[:m.end(1)]
    return re.sub(r"\s+", " ", text).strip()

@app.get("/")
def root():
    return {"ok": True, "app": "veil-mini-llm", "version": APP_VERSION}

@app.get("/health")
def health():
    p = os.path.join(MODEL_DIR, "model.safetensors")
    size = os.path.getsize(p) if os.path.exists(p) else 0
    return {"device": device, "model_dir": MODEL_DIR, "model_bytes": size}

@app.get("/version")
def version():
    return {"app_version": APP_VERSION, "transformers": hf_version}

@app.post("/chat")
def chat(prompt: Prompt):
    user_q = prompt.text.strip()
    preface = SYSTEM
    input_text = f"{preface}User: {user_q}\nAI:"
    enc = tokenizer(input_text, return_tensors="pt").to(device)

    # Try native contrastive search first (supported on HF 4.57.1)
    gen_kwargs = dict(
        max_new_tokens=prompt.max_new_tokens or 160,
        do_sample=False,
        penalty_alpha=0.25,    # increase (0.3–0.6) for more contrast
        top_k=4,
        no_repeat_ngram_size=4,
        repetition_penalty=1.12,
        renormalize_logits=True,
        eos_token_id=eos_id,
    )

    # Optional light sampling if explicitly requested
    if prompt.sample:
        gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9))

    with torch.inference_mode():
        try:
            out = model.generate(**enc, **gen_kwargs)
        except Exception:
            # Fallback to nucleus sampling if something odd happens
            fallback = dict(
                max_new_tokens=prompt.max_new_tokens or 160,
                do_sample=True, temperature=0.8, top_p=0.9,
                no_repeat_ngram_size=4, repetition_penalty=1.12,
                renormalize_logits=True, eos_token_id=eos_id,
            )
            out = model.generate(**enc, **fallback)

    # Slice by token count
    ilen = enc["input_ids"].shape[-1]
    gen_tokens = out[0][ilen:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {"response": _clean(text, input_text)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
