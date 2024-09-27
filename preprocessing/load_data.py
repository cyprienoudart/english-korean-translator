import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.tokenizer import create_tokenizer, save_tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(csv_file, test_size=0.1):
    """
    Loads and splits the dataset into training and validation sets. 
    It also tokenizes the sentences and converts them into padded sequences of integers.
    
    Args:
    csv_file: Path to the dataset file (CSV).
    test_size: Proportion of the dataset to use as validation (default is 10%).

    Returns:
    input_tensor_train: Padded sequences of English training sentences.
    target_tensor_train: Padded sequences of Korean training sentences.
    input_tensor_val: Padded sequences of English validation sentences.
    target_tensor_val: Padded sequences of Korean validation sentences.
    tokenizer_eng: Tokenizer fitted on the English sentences.
    tokenizer_kor: Tokenizer fitted on the Korean sentences.
    """
    
    # Read the CSV file with English-Korean sentence pairs
    df = pd.read_csv(csv_file)

    # Assuming the columns in your CSV are 'english_sentence' and 'korean_sentence'
    english_sentences = df['english_sentence'].tolist()
    korean_sentences = df['korean_sentence'].tolist()

    # Split the dataset into training and validation sets
    eng_train, eng_val, kor_train, kor_val = train_test_split(
        english_sentences, korean_sentences, test_size=test_size
    )

    # Create tokenizers for English and Korean
    tokenizer_eng = create_tokenizer(eng_train)
    tokenizer_kor = create_tokenizer(kor_train)

    # Save the tokenizers (optional)
    save_tokenizer(tokenizer_eng, 'models/tokenizer_eng.pkl')
    save_tokenizer(tokenizer_kor, 'models/tokenizer_kor.pkl')

    # Convert the sentences into sequences of integers
    input_tensor_train = pad_sequences(tokenizer_eng.texts_to_sequences(eng_train), padding='post')
    target_tensor_train = pad_sequences(tokenizer_kor.texts_to_sequences(kor_train), padding='post')
    input_tensor_val = pad_sequences(tokenizer_eng.texts_to_sequences(eng_val), padding='post')
    target_tensor_val = pad_sequences(tokenizer_kor.texts_to_sequences(kor_val), padding='post')

    return (input_tensor_train, target_tensor_train), (input_tensor_val, target_tensor_val), tokenizer_eng, tokenizer_kor
