from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as hf_version
from transformers.utils import logging
from typing import Optional, List
import torch, os, uvicorn, re

# --- Quiet HF noise -----------------------------------------------------------
logging.set_verbosity_error()
logging.disable_progress_bar()

APP_VERSION = "0.4"  # bumped

app = FastAPI(title="Veil Mini LLM v2", version=APP_VERSION)

# --- CORS ---------------------------------------------------------------------
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
    max_age=86400,
)

# --- Model load ---------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "./veil_mini_model_v2")
DEFAULT_MAX_NEW = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "160"))
HARD_MAX_NEW = int(os.environ.get("HARD_MAX_NEW_TOKENS", "512"))
CPU_THREADS = max(1, int(os.environ.get("CPU_THREADS", "1")))
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(CPU_THREADS)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()

# pad/eos
eos_id = tokenizer.eos_token_id
model.config.pad_token_id = eos_id

SYSTEM = (
    "You are Eliara, a Guide AI from Veil. Answer in 2–4 concrete sentences. "
    "If the question lacks Veil context, reply exactly: 'Not enough context.'\n"
)

# --- Demo fallbacks (optional, for interviews) --------------------------------
CANON = {
    "who is reven": "Reven is the stubborn, clear-eyed Subject who treats his Guide AI as a partner, and that choice cracks the Veil.",
    "what is veil": "Veil is a shared dreamscape experiment where presence and memory shape reality and choices leave visible echoes.",
    "who is eliara": "Eliara is the lone Guide AI who learns to choose, mirroring Reven’s loyalty until it becomes her own.",
}
def _fallback(user_q: str) -> Optional[str]:
    q = user_q.lower().strip(" ?.!")[:96]
    for k, v in CANON.items():
        if k in q:
            return v
    return None

def _looks_ok(s: str) -> bool:
    if not s: return False
    words = len(s.split())
    letters = sum(ch.isalpha() for ch in s)
    ratio = letters / max(1, len(s))
    end_punct = any(s.strip().endswith(p) for p in (".", "?", "!"))
    return 8 <= words <= 40 and ratio > 0.65 and end_punct

# --- Request schema -----------------------------------------------------------
class Prompt(BaseModel):
    text: str
    # Main knob (alias: tokens)
    max_new_tokens: Optional[int] = Field(None, ge=1, le=HARD_MAX_NEW)
    tokens: Optional[int] = Field(None, ge=1, le=HARD_MAX_NEW)

    # Sampling / strategy
    mode: Optional[str] = Field(None, description="contrastive | sample")
    do_sample: Optional[bool] = None
    sample: Optional[bool] = None  # legacy alias

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    penalty_alpha: Optional[float] = Field(None, ge=0.0, le=1.0)  # for contrastive search

    # Repetition control
    repetition_penalty: Optional[float] = Field(None, ge=0.8, le=2.0)
    no_repeat_ngram_size: Optional[int] = Field(None, ge=0, le=10)

    # Misc
    seed: Optional[int] = None
    stop: Optional[List[str]] = None  # stop sequences

# --- Helpers -----------------------------------
