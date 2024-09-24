from preprocessing.load_data import load_dataset

def test_load_data():
    # Load the sample dataset
    eng_train, eng_val, kor_train, kor_val = load_dataset('data/sample_data.csv')

    # Print the results to verify
    print("English Training Sentences:", eng_train)
    print("English Validation Sentences:", eng_val)
    print("Korean Training Sentences:", kor_train)
    print("Korean Validation Sentences:", kor_val)

if __name__ == "__main__":
    test_load_data()
