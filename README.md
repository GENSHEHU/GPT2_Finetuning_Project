Fine-Tuned GPT-2 Model on OpenWebText Dataset
This project contains a fine-tuned GPT-2 model using the OpenWebText dataset for text generation tasks.

Overview
The Original_Finetuned_GPT2_Using_Openwebtext.ipynb notebook contains the code for loading the fine-tuned model and generating text based on user-provided prompts.

Usage
1. Load the Model and Tokenizer
Ensure that the fine-tuned model and tokenizer are located in the ./fine_tuned_gpt2_openwebtext directory.


from transformers import GPT2LMHeadModel, GPT2Tokenizer

finetuned_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2_openwebtext")
finetuned_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2_openwebtext")


2. Generate Text
The generate_text function takes a prompt as input and generates text using the fine-tuned model.

def generate_text(prompt, max_length=100):
    inputs = finetuned_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = finetuned_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,  # Enables sampling for temperature and top_p to work
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=finetuned_tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Helps reduce repetitive outputs
    )
    return finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
prompt: The input text prompt.

max_length: Maximum length of the generated text (default: 100).

The function uses sampling with temperature and top_p to generate more diverse and creative text.

repetition_penalty reduces repetitive outputs.

3. Test with Prompts
The notebook includes test prompts to demonstrate the model's text generation capabilities.

test_prompts = [
    "The future of artificial intelligence is",
    "Once upon a time,",
    "The latest advancements in machine learning have",
    "The meaning of love is",
    "In a world where honesty rules,"
]

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generate_text(prompt)}\n")


Requirements
transformers
torch
The fine-tuned model should be in the ./fine_tuned_gpt2_openwebtext directory.

Notes
The pad_token_id in the generate function is set to finetuned_tokenizer.eos_token_id.
The repetition_penalty parameter can be adjusted to control text repetitiveness.
The temperature, top_k, and top_p parameters can be adjusted to control diversity and randomness.

