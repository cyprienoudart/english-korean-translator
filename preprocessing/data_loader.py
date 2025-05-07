import os
import pandas as pd
import numpy as np
from .tokenizer import Tokenizer
from .dataset import TranslationDataset

def download_dataset(url, target_path):
    """
    Download dataset from URL if not already available
    
    Args:
        url (str): URL to download from
        target_path (str): Path to save the dataset
    """
    if not os.path.exists(target_path):
        print(f"Downloading dataset from {url}...")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            # Using pandas to download and save CSV
            df = pd.read_csv(url)
            df.to_csv(target_path, index=False)
            print(f"Dataset saved to {target_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    return True

def prepare_dataset(data_path, 
                    eng_tokenizer_path=None, 
                    kor_tokenizer_path=None,
                    vocab_size_eng=10000,
                    vocab_size_kor=10000,
                    max_length_eng=50,
                    max_length_kor=50,
                    batch_size=64,
                    buffer_size=20000):
    """
    Prepare dataset for training
    
    Args:
        data_path (str): Path to dataset CSV
        eng_tokenizer_path (str): Path to save/load English tokenizer
        kor_tokenizer_path (str): Path to save/load Korean tokenizer
        vocab_size_eng (int): English vocabulary size
        vocab_size_kor (int): Korean vocabulary size
        max_length_eng (int): Maximum English sequence length
        max_length_kor (int): Maximum Korean sequence length
        batch_size (int): Batch size for training
        buffer_size (int): Buffer size for shuffling
        
    Returns:
        tuple: Dataset, English tokenizer, Korean tokenizer, vocabulary sizes
    """
    # Create or load tokenizers
    if eng_tokenizer_path and os.path.exists(eng_tokenizer_path):
        print(f"Loading English tokenizer from {eng_tokenizer_path}")
        eng_tokenizer = Tokenizer.load(eng_tokenizer_path)
    else:
        print(f"Creating new English tokenizer")
        eng_tokenizer = Tokenizer(
            vocab_size=vocab_size_eng,
            max_length=max_length_eng,
            language="eng"
        )
    
    if kor_tokenizer_path and os.path.exists(kor_tokenizer_path):
        print(f"Loading Korean tokenizer from {kor_tokenizer_path}")
        kor_tokenizer = Tokenizer.load(kor_tokenizer_path)
    else:
        print(f"Creating new Korean tokenizer")
        kor_tokenizer = Tokenizer(
            vocab_size=vocab_size_kor,
            max_length=max_length_kor,
            language="kor"
        )
    
    # Create dataset handler
    dataset_handler = TranslationDataset(
        eng_tokenizer=eng_tokenizer,
        kor_tokenizer=kor_tokenizer,
        max_length_eng=max_length_eng,
        max_length_kor=max_length_kor,
        batch_size=batch_size,
        buffer_size=buffer_size
    )
    
    # Load data
    eng_sentences, kor_sentences = dataset_handler.load_data(data_path)
    
    if not eng_sentences or not kor_sentences:
        print("Failed to load data")
        return None, None, None, None, None
    
    # Preprocess data
    dataset, vocab_size_eng, vocab_size_kor = dataset_handler.preprocess_data(
        eng_sentences, kor_sentences
    )
    
    # Split into train and validation
    train_dataset, val_dataset = dataset_handler.get_validation_split(dataset)
    
    # Save tokenizers if paths provided
    if eng_tokenizer_path:
        os.makedirs(os.path.dirname(eng_tokenizer_path), exist_ok=True)
        eng_tokenizer.save(eng_tokenizer_path)
    
    if kor_tokenizer_path:
        os.makedirs(os.path.dirname(kor_tokenizer_path), exist_ok=True)
        kor_tokenizer.save(kor_tokenizer_path)
    
    return train_dataset, val_dataset, eng_tokenizer, kor_tokenizer, vocab_size_eng, vocab_size_kor 