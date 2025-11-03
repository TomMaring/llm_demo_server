from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import torch, os, uvicorn, re
from typing import Optional

# Quiet HF logs
logging.set_verbosity_error()
logging.disable_progress_bar()

app = FastAPI(title="Veil Mini LLM v2", version="0.3")

# -------- Model load (once) --------
MODEL_DIR = os.environ.get("MODEL_DIR", "./veil_mini_model_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, int(os.environ.get("CPU_THREADS", "1"))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()

# pad/eos sanity
eos_id = tokenizer.eos_token_id
model.config.pad_token_id = eos_id

SYSTEM = (
    "You are Eliara, a Guide AI from Veil. Answer in 2–4 concrete sentences. "
    "If the question lacks Veil context, reply exactly: 'Not enough context.'\n"
)

class Prompt(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None
    sample: Optional[bool] = None  # optional: enable light sampling

def _clean(text: str, prompt: str) -> str:
    # Remove echoed prompt
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Stop if the model tries to start a new user turn
    for m in ["\nUser:", "\nYou:", "\nHuman:", "\nQ:"]:
        i = text.find(m)
        if i != -1:
            text = text[:i].strip()

    # Trim to the last sentence end
    m = re.search(r'([.?!])[^.?!]*$', text)
    if m:
        text = text[:m.end(1)]
    return re.sub(r"\s+", " ", text).strip()

@app.get("/health")
def health():
    p = os.path.join(MODEL_DIR, "model.safetensors")
    size = os.path.getsize(p) if os.path.exists(p) else 0
    return {"device": device, "model_dir": MODEL_DIR, "model_bytes": size}

@app.post("/chat")
def chat(prompt: Prompt):
    user_q = prompt.text.strip()
    preface = SYSTEM
    input_text = f"{preface}User: {user_q}\nAI:"
    enc = tokenizer(input_text, return_tensors="pt").to(device)

    # --- Decoding ---
    # Use native contrastive search (no 'decoding=' param!)
    gen_kwargs = dict(
        max_new_tokens=prompt.max_new_tokens or 160,
        do_sample=False,               # contrastive is deterministic by default
        penalty_alpha=0.25,            # tune if needed (0.3–0.6 increases contrast)
        top_k=4,                       # small k for a tiny model
        no_repeat_ngram_size=4,
        repetition_penalty=1.12,
        renormalize_logits=True,
        eos_token_id=eos_id,
    )

    # Optional light sampling if explicitly requested
    if prompt.sample:
        gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9))

    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)

    # Slice by token count, not chars
    ilen = enc["input_ids"].shape[-1]
    gen_tokens = out[0][ilen:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {"response": _clean(text, input_text)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
