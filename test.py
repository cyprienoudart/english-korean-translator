import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask
import numpy as np
import nltk
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.models import Model

print("TensorFlow version:", tf.__version__)
print("Flask version:", Flask.__version__)
print("Numpy version:", np.__version__)

nltk.download('punkt')  # Make sure NLTK data is available