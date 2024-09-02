from translator import EnglishKoreanTranslator

def test_translation():
    print("Initializing translator...")  # Debugging statement
    translator = EnglishKoreanTranslator()
    
    print("Translating text...")  # Debugging statement
    # Define a test sentence in English
    english_text = "Hello, how are you?"
    
    # Translate the sentence to Korean
    korean_translation = translator.translate(english_text)
     
    # Print the result
    print(f"English: {english_text}")
    print(f"Korean: {korean_translation}")

if __name__ == "__main__":
    print("Running test...")  # Debugging statement
    test_translation()