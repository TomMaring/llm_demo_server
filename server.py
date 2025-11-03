from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import torch, os, uvicorn, re
from typing import Optional

# Shut up, HF.
logging.set_verbosity_error()
logging.disable_progress_bar()

app = FastAPI(title="Veil Mini LLM v2", version="0.2")

# -------- Model load (once) --------
MODEL_DIR = os.environ.get("MODEL_DIR", "./veil_mini_model_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, int(os.environ.get("CPU_THREADS", "1"))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
)
model.to(device).eval()

# pad/eos sanity
eos_id = tokenizer.eos_token_id
model.config.pad_token_id = eos_id

# Minimal instruction. No roleplay theater.
INSTR = (
    "Answer clearly in 2–4 sentences about the Veil setting. "
    "If there isn't enough context, reply exactly: Not enough context.\n"
)

class Prompt(BaseModel):
    text: str
    max_new_tokens: Optional[int] = None
    sample: Optional[bool] = None  # if true -> sampling; else -> contrastive search

def _clean(text: str, prompt: str) -> str:
    # Drop echoed prompt
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Chop off invented turns
    for tag in ("\nQ:", "\nA:", "\nUser:", "\nYou:"):
        i = text.find(tag)
        if i != -1:
            text = text[:i].strip()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Trim to last sentence ending
    m = re.search(r"([.?!])[^.?!]*$", text)
    if m:
        text = text[: m.end(1)].strip()

    # Guard against degenerate loops: cut if a 2–5 word chunk repeats 3+ times
    text = re.sub(r"(\b[\w’'-]{2,}\b(?:\s+\b[\w’'-]{2,}\b){1,5})(?:\s+\1){2,}", r"\1", text)

    return text if text else "Not enough context."

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(prompt: Prompt):
    q = prompt.text.strip()
    # Keep it tiny: one-shot-ish pattern that small models handle
    input_text = f"{INSTR}Q: {q}\nA:"

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=prompt.max_new_tokens or 140,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        repetition_penalty=1.15,
    )

    # Default: contrastive search (works better than greedy for baby models)
    if prompt.sample:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                typical_p=0.95,
                no_repeat_ngram_size=3,
                renormalize_logits=True,
            )
        )
    else:
        # contrastive search triggers when penalty_alpha & top_k are set
        gen_kwargs.update(dict(penalty_alpha=0.6, top_k=4))

    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)

    # Slice by token length, not characters
    inp_len = enc["input_ids"].shape[-1]
    gen_tokens = out[0, inp_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return {"response": _clean(text, input_text)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
