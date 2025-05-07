import tensorflow as tf
import numpy as np
import pickle
import os
from collections import Counter
import re

class Tokenizer:
    def __init__(self, 
                 vocab_size=None, 
                 max_length=None, 
                 special_tokens=None,
                 language=""):
        """
        Initialize tokenizer for a specific language
        
        Args:
            vocab_size (int): Maximum size of vocabulary
            max_length (int): Maximum sequence length
            special_tokens (dict): Dictionary of special tokens
            language (str): Language identifier ("eng" or "kor")
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.language = language
        
        # Default special tokens
        self.special_tokens = {
            "PAD": "<PAD>",
            "UNK": "<UNK>",
            "START": "<START>",
            "END": "<END>"
        }
        
        # Update special tokens if provided
        if special_tokens:
            self.special_tokens.update(special_tokens)
        
        # Initialize vocabularies
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()
        
        # Add special tokens to vocabulary
        for idx, token in enumerate(self.special_tokens.values()):
            self.word_to_index[token] = idx
            self.index_to_word[idx] = token
    
    def fit(self, texts):
        """
        Fit tokenizer on texts
        
        Args:
            texts (list): List of text strings
        """
        # Count words
        for text in texts:
            words = self._preprocess_text(text)
            self.word_counts.update(words)
        
        # Get most common words
        most_common = self.word_counts.most_common(
            self.vocab_size - len(self.special_tokens) if self.vocab_size else None
        )
        
        # Create vocabulary
        for word, _ in most_common:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
            
        # Update vocab size
        self.vocab_size = len(self.word_to_index)
        
        print(f"Vocabulary size for {self.language}: {self.vocab_size}")
    
    def _preprocess_text(self, text):
        """
        Preprocess text based on language
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        if self.language == "eng":
            # English preprocessing
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            return text.split()
        elif self.language == "kor":
            # Korean preprocessing
            # For simplicity, we'll just split by space for now
            return text.split()
        else:
            return text.split()
    
    def encode(self, text):
        """
        Encode text to sequence of indices
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of token indices
        """
        words = self._preprocess_text(text)
        sequence = [self.word_to_index.get(word, self.word_to_index[self.special_tokens["UNK"]]) 
                    for word in words]
        
        # Add START and END tokens
        sequence = [self.word_to_index[self.special_tokens["START"]]] + sequence + [self.word_to_index[self.special_tokens["END"]]]
        
        # Pad sequence
        if self.max_length:
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            else:
                sequence += [self.word_to_index[self.special_tokens["PAD"]]] * (self.max_length - len(sequence))
        
        return sequence
    
    def decode(self, sequence):
        """
        Decode sequence of indices to text
        
        Args:
            sequence (list): List of token indices
            
        Returns:
            str: Decoded text
        """
        # Remove PAD, START tokens
        pad_idx = self.word_to_index[self.special_tokens["PAD"]]
        start_idx = self.word_to_index[self.special_tokens["START"]]
        end_idx = self.word_to_index[self.special_tokens["END"]]
        
        words = []
        for idx in sequence:
            if idx == pad_idx or idx == start_idx:
                continue
            if idx == end_idx:
                break
            words.append(self.index_to_word.get(idx, self.special_tokens["UNK"]))
        
        return " ".join(words)
    
    def save(self, path):
        """Save tokenizer to file"""
        data = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "special_tokens": self.special_tokens,
            "word_to_index": self.word_to_index,
            "index_to_word": self.index_to_word,
            "word_counts": self.word_counts,
            "language": self.language
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            max_length=data["max_length"],
            special_tokens=data["special_tokens"],
            language=data["language"]
        )
        
        tokenizer.word_to_index = data["word_to_index"]
        tokenizer.index_to_word = data["index_to_word"]
        tokenizer.word_counts = data["word_counts"]
        
        return tokenizer
