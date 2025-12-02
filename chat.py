

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your fine-tuned model folder
MODEL_PATH = "/tmp/Qwen3-30B-A3B-Edgebot-Merged"

def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",       # uses bf16/float16 if available
        device_map="auto",        # put it on GPU automatically
        trust_remote_code=True,
    )

    model.eval()
    return tokenizer, model


def generate_reply(tokenizer, model, user_text, max_new_tokens=3000):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information",
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    # Uses the chat template that came with your checkpoint (chat_template.jinja)
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
    )

    # send tensors to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt part and decode only the answer
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def main():
    tokenizer, model = load_model()

    print("AlphaCoder CLI ready.")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_text = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if not user_text:
            continue

        reply = generate_reply(tokenizer, model, user_text)
        print(f"AlphaCoder: {reply}\n")


if __name__ == "__main__":
    main()
