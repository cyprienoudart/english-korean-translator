import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(csv_file):
    # Read the original CSV file
    df = pd.read_csv(csv_file)

    # Select the English-Korean sentence pairs from the CSV file
    columns = ['E1', 'K1', 'E2', 'K2']
    df = df[columns]

    # Rename columns for simplicity
    df.columns = ['english_1', 'korean_1', 'english_2', 'korean_2']

    # Combine the two sets of sentence pairs into one DataFrame
    english_sentences = pd.concat([df['english_1'], df['english_2']]).dropna().tolist()
    korean_sentences = pd.concat([df['korean_1'], df['korean_2']]).dropna().tolist()

    # Split into train and validation sets
    eng_train, eng_val, kor_train, kor_val = train_test_split(english_sentences, korean_sentences, test_size=0.1)

    # Create and fit the tokenizer on both the English and Korean sentences
    tokenizer_eng = Tokenizer()
    tokenizer_kor = Tokenizer()

    tokenizer_eng.fit_on_texts(eng_train)
    tokenizer_kor.fit_on_texts(kor_train)

    # Convert sentences to sequences of integers
    input_tensor_train = pad_sequences(tokenizer_eng.texts_to_sequences(eng_train), padding='post')
    target_tensor_train = pad_sequences(tokenizer_kor.texts_to_sequences(kor_train), padding='post')
    input_tensor_val = pad_sequences(tokenizer_eng.texts_to_sequences(eng_val), padding='post')
    target_tensor_val = pad_sequences(tokenizer_kor.texts_to_sequences(kor_val), padding='post')

    return eng_train, eng_val, kor_train, kor_val
