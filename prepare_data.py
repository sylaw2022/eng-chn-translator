from datasets import load_dataset
import os

def download_and_prepare_data():
    print("Downloading dataset opus100 (en-zh)...")
    # Load the dataset (this downloads it to the cache dir by default)
    dataset = load_dataset("opus100", "en-zh")
    
    print("Dataset structure:")
    print(dataset)
    
    # Save to disk for easy access
    save_path = "./data/opus100_en_zh"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    print(f"Saving dataset to {save_path}...")
    dataset.save_to_disk(save_path)
    print("Done!")

    # Show a few examples
    print("\nExamples:")
    for i in range(3):
        print(dataset['train'][i])

if __name__ == "__main__":
    download_and_prepare_data()













