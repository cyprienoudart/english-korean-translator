from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

def create_tokenizer(sentences):
    """
    Create a Keras Tokenizer and fit it on the given sentences.
    
    :param sentences: A list of sentences to fit the tokenizer.
    :return: Fitted tokenizer.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def save_tokenizer(tokenizer, filepath):
    """
    Save the tokenizer to a file using pickle.
    
    :param tokenizer: The tokenizer object to save.
    :param filepath: Path to the file where the tokenizer will be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(filepath):
    """
    Load a tokenizer from a file.
    
    :param filepath: Path to the file where the tokenizer is saved.
    :return: Loaded tokenizer object.
    """
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
