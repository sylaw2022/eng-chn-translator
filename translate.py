from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import sys

def main():
    # Use the fine-tuned model if available, otherwise the base model
    model_path = "./final_model"
    base_model = "Helsinki-NLP/opus-mt-en-zh"
    
    try:
        print(f"Attempting to load model from {model_path}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Loaded fine-tuned model.")
    except Exception as e:
        print(f"Fine-tuned model not found or invalid ({e}). Loading base model {base_model}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=device)

    print("\nEnglish to Chinese Translator (Type 'q' to exit)")
    print("-" * 50)

    while True:
        text = input("Enter English text: ")
        if text.lower() in ['q', 'quit', 'exit']:
            break
        
        if not text.strip():
            continue

        translation = translator(text)[0]['translation_text']
        print(f"Chinese: {translation}\n")

if __name__ == "__main__":
    main()













