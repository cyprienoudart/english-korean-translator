import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(csv_file, test_size=0.1):
    """
    Loads and splits the dataset into training and validation sets.
    
    Args:
    csv_file: Path to the dataset file (CSV).
    test_size: Proportion of the dataset to use as validation (default is 10%).

    Returns:
    eng_train: List of English training sentences.
    eng_val: List of English validation sentences.
    kor_train: List of Korean training sentences.
    kor_val: List of Korean validation sentences.
    """
    
    # Read the CSV file with English-Korean sentence pairs
    df = pd.read_csv(csv_file)

    # Assuming the columns in your CSV are 'english_sentence' and 'korean_sentence'
    # No need to handle E1/K1/E2/K2 as it simplifies the data structure
    english_sentences = df['english_sentence'].tolist()
    korean_sentences = df['korean_sentence'].tolist()

    # Split the dataset into training and validation sets
    eng_train, eng_val, kor_train, kor_val = train_test_split(
        english_sentences, korean_sentences, test_size=test_size
    )

    return eng_train, eng_val, kor_train, kor_val
