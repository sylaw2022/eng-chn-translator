import os
import torch
import torch.nn as nn
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
import evaluate
import numpy as np
import sys
import time
from torch.utils.data import DataLoader
from custom_transformer import Transformer, create_causal_mask, create_padding_mask
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# 1. Configuration
MODEL_CHECKPOINT = "./untrained_model"
DATA_PATH = "./data/opus100_en_zh"
OUTPUT_DIR = "./results"
BATCH_SIZE = 8
ACCUM_STEPS = 1  # Standard batch size for custom loop, adjust if OOM
LR = 2e-4
EPOCHS = 5
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"Using device: {DEVICE}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

    # 2. Load Data
    print("Loading dataset...")
    raw_datasets = load_from_disk(DATA_PATH)
    
    train_subset_size = 100000 
    print(f"Using {train_subset_size} samples for training from scratch.")
    train_subset = raw_datasets["train"].select(range(train_subset_size))
    eval_subset = raw_datasets["validation"]
    
    datasets_to_process = DatasetDict({
        "train": train_subset,
        "validation": eval_subset
    })

    # 3. Tokenizer
    print(f"Loading tokenizer from {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    PAD_IDX = tokenizer.pad_token_id
    
    # Preprocessing
    source_lang = "en"
    target_lang = "zh"

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        
        model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_LEN, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing data...")
    tokenized_datasets = datasets_to_process.map(
        preprocess_function, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names
    )

    # 4. Collate Function for DataLoader
    def collate_fn(batch):
        # batch is a list of dicts: {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
        
        src_batch = [item['input_ids'] for item in batch]
        tgt_batch = [item['labels'] for item in batch]
        
        # Pad sequences
        src_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in src_batch], 
            batch_first=True, 
            padding_value=PAD_IDX
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in tgt_batch], 
            batch_first=True, 
            padding_value=PAD_IDX
        )
        
        return src_padded, tgt_padded

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 5. Model Initialization
    print("Initializing Custom Transformer...")
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Optimization
    # Lower learning rate to 5e-5 for stability
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = OneCycleLR(optimizer, max_lr=5e-5, total_steps=total_steps, pct_start=0.1)

    start_epoch = 0

    # Resume capability
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint-epoch-") and f.endswith(".pt")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[2].split(".")[0]))
            checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("No checkpoints found. Starting from scratch.")

    # 7. Training Loop
    print("Starting training...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            # Prepare inputs
            # tgt_input: remove last token (teacher forcing)
            # tgt_output: remove first token (prediction target)
            
            # Wait, tokenized output includes EOS but usually not BOS?
            # Let's inspect typical Marian tokenizer output.
            # Usually: input_ids ends with EOS.
            # Decoder input: needs start token.
            # If labels = [A, B, EOS], decoder_input = [PAD/START, A, B], target = [A, B, EOS]
            
            # Assuming tokenizer output: [tok1, tok2, ..., EOS]
            # Decoder Input: [PAD, tok1, tok2, ...]
            # Target: [tok1, tok2, ..., EOS]
            
            tgt_input = tgt.clone()
            # Shift right: add PAD at start, remove last
            # Create a column of PADs
            pad_col = torch.full((tgt.shape[0], 1), PAD_IDX, device=DEVICE, dtype=tgt.dtype)
            tgt_input = torch.cat((pad_col, tgt[:, :-1]), dim=1)
            
            tgt_output = tgt # The labels
            
            # Create Masks
            # Source mask (padding mask)
            # Shape: [batch, 1, 1, src_len]
            src_mask = create_padding_mask(src, PAD_IDX)
            
            # Target mask
            # 1. Causal mask
            tgt_len = tgt_input.shape[1]
            causal_mask = create_causal_mask(tgt_len, DEVICE) # [tgt_len, tgt_len]
            # 2. Padding mask
            tgt_pad_mask = create_padding_mask(tgt_input, PAD_IDX) # [batch, 1, 1, tgt_len]
            
            # Combine
            # Broadcast causal: [1, 1, tgt_len, tgt_len]
            # Combine: tgt_mask = causal & pad
            tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0) & tgt_pad_mask
            
            # Forward
            logits = model(src, tgt_input, src_mask, tgt_mask)
            
            # Reshape for loss: [batch*seq_len, vocab_size] vs [batch*seq_len]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            
            # Backward
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")
        
        # Evaluation (Loss only for speed)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in eval_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                
                tgt_input = torch.cat((torch.full((tgt.shape[0], 1), PAD_IDX, device=DEVICE), tgt[:, :-1]), dim=1)
                tgt_output = tgt
                
                src_mask = create_padding_mask(src, PAD_IDX)
                tgt_len = tgt_input.shape[1]
                tgt_mask = create_causal_mask(tgt_len, DEVICE).unsqueeze(0).unsqueeze(0) & create_padding_mask(tgt_input, PAD_IDX)
                
                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
                val_loss += loss.item()
                
        print(f"Validation Loss: {val_loss / len(eval_loader):.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    # Final Save
    print("Saving final model...")
    final_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model_tokenizer"))
    print("Done!")

if __name__ == "__main__":
    main()
