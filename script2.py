import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

CHAT_TEMPLATE_ID = "HuggingFaceH4/zephyr-7b-beta"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "clibrain/mamba-2.8b-instruct-openhermes"

eos_token = "<|endoftext|>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.eos_token = eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_ID).chat_template

model = MambaLMHeadModel.from_pretrained(
        model_name, device=device, dtype=torch.float16)

messages = []
prompt = "Tell me 5 sites to visit in Spain"
messages.append(dict(role="user", content=prompt))

input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
).to(device)

out = model.generate(
    input_ids=input_ids,
    max_length=2000,
    temperature=0.9,
    top_p=0.7,
    eos_token_id=tokenizer.eos_token_id,
)

decoded = tokenizer.batch_decode(out)
assistant_message = (
    decoded[0].split("<|assistant|>\n")[-1].replace(eos_token, "")
)

print(assistant_message)
