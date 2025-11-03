from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
import torch, os, uvicorn, re

# Quiet HF logs
logging.set_verbosity_error()
logging.disable_progress_bar()

app = FastAPI(title="Veil Mini LLM v2", version="0.1")

# -------- Model load (once) --------
MODEL_DIR = os.environ.get("MODEL_DIR", "./veil_mini_model_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, int(os.environ.get("CPU_THREADS", "1"))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(device).eval()

# pad/eos sanity
eos_id = tokenizer.eos_token_id
model.config.pad_token_id = eos_id

SYSTEM = (
    "You are Eliara, a Guide AI from Veil. Answer in 2–4 clear sentences. "
    "Be concrete. If you don't have enough context, say: 'Not enough context.'\n"
)

class Prompt(BaseModel):
    text: str
    max_new_tokens: int | None = None
    sample: bool | None = None  # optional: turn on light sampling

def _clean(text: str, prompt: str) -> str:
    # Remove echoed prompt
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    # Cut at next user marker if it tried to invent a turn
    for m in ["\nUser:", "\nYou:"]:
        i = text.find(m)
        if i != -1:
            text = text[:i].strip()
    # Trim to last sentence end
    m = re.search(r'([.?!])[^.?!]*$', text)
    if m:
        text = text[:m.end(1)]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/chat")
def chat(prompt: Prompt):
    user_q = prompt.text.strip()
    preface = SYSTEM
    input_text = f"{preface}User: {user_q}\nAI:"

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Deterministic by default; sampling is opt-in
    decode = dict(
        do_sample=False,
        max_new_tokens=prompt.max_new_tokens or 160,
        repetition_penalty=1.12,
        no_repeat_ngram_size=4,
        renormalize_logits=True,
        eos_token_id=eos_id
    )
    if prompt.sample:
        decode.update(dict(do_sample=True, temperature=0.7, top_p=0.9))

    with torch.inference_mode():
        outputs = model.generate(**inputs, **decode)

    # Slice by token length, not characters
    input_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return {"response": _clean(text, input_text)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
