import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.load_data import load_dataset

def test_load_data():
    """
    Function to test the data loading, tokenization, and padding process.
    """
    
    # Load dataset and split into training/validation sets
    (input_tensor_train, target_tensor_train), (input_tensor_val, target_tensor_val), tokenizer_eng, tokenizer_kor = load_dataset('sample_data.csv')

    # Print results for verification
    print("Input Tensor (English) - Training Set:")
    print(input_tensor_train)
    
    print("\nTarget Tensor (Korean) - Training Set:")
    print(target_tensor_train)
    
    print("\nInput Tensor (English) - Validation Set:")
    print(input_tensor_val)
    
    print("\nTarget Tensor (Korean) - Validation Set:")
    print(target_tensor_val)
    
    # Print tokenizers to check if they worked
    print("\nEnglish Tokenizer Word Index:")
    print(tokenizer_eng.word_index)
    
    print("\nKorean Tokenizer Word Index:")
    print(tokenizer_kor.word_index)

# Run the test
if __name__ == "__main__":
    test_load_data()


"""#### Example output : #####
Input Tensor (English) - Training Set:
[[ 1  2  3]
 [ 4  5]]

Target Tensor (Korean) - Training Set:
[[ 6  7  8]
 [ 9 10]]

Input Tensor (English) - Validation Set:
[[ 1 11]]

Target Tensor (Korean) - Validation Set:
[[12 13]]

English Tokenizer Word Index:
{'hello': 1, 'how': 2, 'are': 3, 'you': 4, 'thank': 5}

Korean Tokenizer Word Index:
{'안녕하세요': 1, '어떻게': 2, '지내세요': 3, '감사합니다': 4}"""