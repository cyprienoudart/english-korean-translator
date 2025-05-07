#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from .tokenizer import Tokenizer

class TranslationDataset:
    def __init__(self, 
                 eng_tokenizer,
                 kor_tokenizer,
                 max_length_eng=50,
                 max_length_kor=50,
                 batch_size=64,
                 buffer_size=20000):
        """
        Initialize dataset class for translation
        
        Args:
            eng_tokenizer (Tokenizer): English tokenizer
            kor_tokenizer (Tokenizer): Korean tokenizer
            max_length_eng (int): Maximum length of English sequences
            max_length_kor (int): Maximum length of Korean sequences
            batch_size (int): Batch size for training
            buffer_size (int): Buffer size for shuffling
        """
        self.eng_tokenizer = eng_tokenizer
        self.kor_tokenizer = kor_tokenizer
        self.max_length_eng = max_length_eng
        self.max_length_kor = max_length_kor
        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def load_data(self, filepath):
        """
        Load data from CSV file
        
        Args:
            filepath (str): Path to CSV file with English-Korean pairs
            
        Returns:
            tuple: Lists of English and Korean sentences
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset columns: {df.columns.tolist()}")
            
            # Map column names to expected names
            eng_col = 'english_sentence' if 'english_sentence' in df.columns else 'english'
            kor_col = 'korean_sentence' if 'korean_sentence' in df.columns else 'korean'
            
            english_sentences = df[eng_col].tolist()
            korean_sentences = df[kor_col].tolist()
            
            print(f"Loaded {len(english_sentences)} sentence pairs")
            print(f"Sample: {english_sentences[0]} -> {korean_sentences[0]}")
            
            return english_sentences, korean_sentences
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []
    
    def preprocess_data(self, eng_sentences, kor_sentences):
        """
        Preprocess and tokenize the data
        
        Args:
            eng_sentences (list): List of English sentences
            kor_sentences (list): List of Korean sentences
            
        Returns:
            tuple: TensorFlow dataset and vocabulary sizes
        """
        # Fit tokenizers if not already fitted
        if len(self.eng_tokenizer.word_to_index) <= len(self.eng_tokenizer.special_tokens):
            print("Fitting English tokenizer...")
            self.eng_tokenizer.fit(eng_sentences)
        
        if len(self.kor_tokenizer.word_to_index) <= len(self.kor_tokenizer.special_tokens):
            print("Fitting Korean tokenizer...")
            self.kor_tokenizer.fit(kor_sentences)
        
        # Encode sentences
        eng_sequences = [self.eng_tokenizer.encode(sentence) for sentence in eng_sentences]
        kor_sequences = [self.kor_tokenizer.encode(sentence) for sentence in kor_sentences]
        
        # Create tensor datasets
        dataset = tf.data.Dataset.from_tensor_slices((
            np.array(eng_sequences),
            np.array(kor_sequences)
        ))
        
        # Process dataset
        dataset = self._process_dataset(dataset)
        
        return dataset, self.eng_tokenizer.vocab_size, self.kor_tokenizer.vocab_size
    
    def _process_dataset(self, dataset):
        """
        Process dataset for training
        
        Args:
            dataset (tf.data.Dataset): Input dataset
            
        Returns:
            tf.data.Dataset: Processed dataset
        """
        # Create decoder inputs and targets
        def create_decoder_data(eng, kor):
            # Decoder input is target with START token, without END token
            decoder_input = kor
            # Target is without START token but with END token
            target = kor[1:]
            
            # Add padding to target
            padding = tf.constant([0], dtype=tf.int64)
            target = tf.concat([target, padding], axis=0)
            
            return eng, decoder_input, target
        
        dataset = dataset.map(create_decoder_data)
        
        # Shuffle and batch
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    def get_validation_split(self, dataset, validation_split=0.2):
        """
        Split dataset into training and validation
        
        Args:
            dataset (tf.data.Dataset): Input dataset
            validation_split (float): Validation split ratio
            
        Returns:
            tuple: Training and validation datasets
        """
        # Calculate split size
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        train_size = int(dataset_size * (1 - validation_split))
        
        # Split dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        return train_dataset, val_dataset

# Read the original file content
with open('preprocessing/dataset.py', 'r') as file:
    content = file.read()

# Replace the load_data method
original_method = """    def load_data(self, filepath):
        \"\"\"
        Load data from CSV file
        
        Args:
            filepath (str): Path to CSV file with English-Korean pairs
            
        Returns:
            tuple: Lists of English and Korean sentences
        \"\"\"
        try:
            df = pd.read_csv(filepath)
            # Assuming columns are named 'english' and 'korean'
            english_sentences = df['english_sentence' if 'english_sentence' in df.columns else 'english_sentence' if 'english_sentence' in df.columns else 'english'].tolist()
            korean_sentences = df['korean_sentence' if 'korean_sentence' in df.columns else 'korean_sentence' if 'korean_sentence' in df.columns else 'korean'].tolist()
            
            print(f"Loaded {len(english_sentences)} sentence pairs")
            
            return english_sentences, korean_sentences
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []"""

# New method with correct column names
new_method = """    def load_data(self, filepath):
        \"\"\"
        Load data from CSV file
        
        Args:
            filepath (str): Path to CSV file with English-Korean pairs
            
        Returns:
            tuple: Lists of English and Korean sentences
        \"\"\"
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset columns: {df.columns.tolist()}")
            
            # Map column names to expected names
            eng_col = 'english_sentence' if 'english_sentence' in df.columns else 'english'
            kor_col = 'korean_sentence' if 'korean_sentence' in df.columns else 'korean'
            
            english_sentences = df[eng_col].tolist()
            korean_sentences = df[kor_col].tolist()
            
            print(f"Loaded {len(english_sentences)} sentence pairs")
            print(f"Sample: {english_sentences[0]} -> {korean_sentences[0]}")
            
            return english_sentences, korean_sentences
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []"""

# Replace the method in the content
modified_content = content.replace(original_method, new_method)

# Write the modified content back to file
with open('preprocessing/dataset.py', 'w') as file:
    file.write(modified_content)

print("File successfully updated!")