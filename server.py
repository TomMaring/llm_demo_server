from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    __version__ as hf_version,
    StoppingCriteriaList,
    StoppingCriteria,
)
from transformers.utils import logging
from typing import Optional, List
import torch, os, uvicorn, re

# --- Quiet HF noise -----------------------------------------------------------
logging.set_verbosity_error()
logging.disable_progress_bar()

APP_VERSION = "0.5"

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

# Prefer half precision on CUDA for speed/VRAM. CPU stays float32.
load_dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=load_dtype).to(device).eval()

# eos/pad id sanity
eos_id = tokenizer.eos_token_id
if eos_id is None:
    # very defensive; most LMs have an eos
    tokenizer.eos_token = tokenizer.eos_token or "</s>"
    eos_id = tokenizer.eos_token_id
model.config.pad_token_id = eos_id
tokenizer.pad_token_id = eos_id

SYSTEM = os.environ.get(
    "SYSTEM_PROMPT",
    "You are Eliara, a Guide AI from Veil. Answer in 2–4 concrete sentences. "
    "If the question lacks Veil context, reply exactly: 'Not enough context.'\n",
)

# --- Demo fallbacks --------------------------------
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
    return 8 <= words <= 60 and ratio > 0.55 and end_punct

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

# --- Helpers ------------------------------------------------------------------
def _clean(text: str, prompt: str) -> str:
    # Remove echoed prompt
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Stop if the model tries to begin a new user turn
    for m in ["\nUser:", "\nYou:", "\nHuman:", "\nQ:"]:
        i = text.find(m)
        if i != -1:
            text = text[:i].strip()
            break

    # Trim to the last sentence end
    m = re.search(r'([.?!])[^.?!]*$', text)
    if m:
        text = text[:m.end(1)]
    return re.sub(r"\s+", " ", text).strip()

def _apply_stop(s: str, stops: Optional[List[str]]) -> str:
    if not stops:
        return s
    idx = min((s.find(t) for t in stops if t in s), default=-1)
    return s[:idx] if idx >= 0 else s

class StopOnTokenSeq(StoppingCriteria):
    """Stops when any encoded stop sequence occurs at the end of the output."""
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        # filter out empties
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequences:
            return False
        seq = input_ids[0].tolist()
        for stop in self.stop_sequences:
            n = len(stop)
            if n <= len(seq) and seq[-n:] == stop:
                return True
        return False

def _make_stoppers(stops: Optional[List[str]]) -> Optional[StoppingCriteriaList]:
    if not stops:
        return None
    # encode each stop without special tokens
    encoded = [tokenizer.encode(s, add_special_tokens=False) for s in stops if s]
    if not encoded:
        return None
    return StoppingCriteriaList([StopOnTokenSeq(encoded)])

# --- Endpoints ----------------------------------------------------------------
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

    # Effective knobs with defaults (mirrors UI)
    max_new = (
        prompt.max_new_tokens
        or prompt.tokens
        or DEFAULT_MAX_NEW
    )
    max_new = max(1, min(HARD_MAX_NEW, int(max_new)))

    mode = (prompt.mode or ("sample" if prompt.sample else None) or "contrastive").lower().strip()
    # do_sample is true if you asked for 'sample' mode or any sampling knobs were provided
    do_sample = bool(
        prompt.do_sample
        or prompt.sample
        or mode.startswith("samp")
        or any(v is not None for v in (prompt.temperature, prompt.top_p))
    )

    # explicit defaults, matching the UI defaults you send
    penalty_alpha = 0.25 if prompt.penalty_alpha is None else float(prompt.penalty_alpha)
    top_k = 4 if prompt.top_k is None else int(prompt.top_k)
    temperature = 0.7 if prompt.temperature is None else float(prompt.temperature)
    top_p = 0.9 if prompt.top_p is None else float(prompt.top_p)
    repetition_penalty = float(prompt.repetition_penalty or 1.12)
    no_repeat = int(prompt.no_repeat_ngram_size or 4)
    no_repeat = max(0, min(10, no_repeat))

    if prompt.seed is not None:
        torch.manual_seed(int(prompt.seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(prompt.seed))

    # Build stopping criteria from 'stop' tokens (and keep post-trim just in case)
    stoppers = _make_stoppers(prompt.stop)

    # Base kwargs shared by both strategies
    base_kwargs = dict(
        max_new_tokens=max_new,
        no_repeat_ngram_size=no_repeat,
        repetition_penalty=repetition_penalty,
        renormalize_logits=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        stopping_criteria=stoppers,
        use_cache=True,
    )

    if do_sample:
        # nucleus/top-k sampling
        gen_kwargs = dict(
            **base_kwargs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,  # your UI always sends a value; honor it
        )
    else:
        # contrastive search requires top_k >= 2
        gen_kwargs = dict(
            **base_kwargs,
            do_sample=False,
            penalty_alpha=max(1e-6, penalty_alpha),
            top_k=max(2, top_k),
        )

    # Generate
    with torch.inference_mode():
        if device == "cuda":
            # modest perf boost on consumer GPUs, no API changes
            with torch.cuda.amp.autocast(dtype=torch.float16):
                try:
                    out = model.generate(**enc, **gen_kwargs)
                except Exception:
                    # Safe fallback to sampling if contrastive path misbehaves
                    fallback = dict(
                        **base_kwargs,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        top_k=top_k,
                    )
                    out = model.generate(**enc, **fallback)
        else:
            try:
                out = model.generate(**enc, **gen_kwargs)
            except Exception:
                fallback = dict(
                    **base_kwargs,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=top_k,
                )
                out = model.generate(**enc, **fallback)

    # Decode
    ilen = enc["input_ids"].shape[-1]
    gen_tokens = out[0][ilen:]
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    clean = _clean(raw, input_text)
    clean = _apply_stop(clean, prompt.stop)

    # Demo guardrails
    if DEMO_MODE:
        fb = _fallback(user_q)
        if fb:
            clean = fb
    if not _looks_ok(clean):
        clean = _fallback(user_q) or clean or "Not enough context."

    # Usage stats
    prompt_tokens = int(ilen)
    completion_tokens = int(len(gen_tokens))
    total_tokens = prompt_tokens + completion_tokens

    return JSONResponse(
        content={
            "response": clean,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "params": {
                # echo back the effective knobs so the UI debug panel matches reality
                "mode": "sample" if do_sample else "contrastive",
                "max_new_tokens": max_new,
                "temperature": temperature if do_sample else None,
                "top_p": top_p if do_sample else None,
                "top_k": top_k if do_sample else max(2, top_k),
                "penalty_alpha": None if do_sample else max(1e-6, penalty_alpha),
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat,
                "seed": prompt.seed,
                "stop": prompt.stop or [],
            },
            "model": os.path.basename(MODEL_DIR),
            "transformers": hf_version,
        },
        media_type="application/json; charset=utf-8",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
