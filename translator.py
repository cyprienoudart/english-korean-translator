from transformers import MarianMTModel, MarianTokenizer

class EnglishKoreanTranslator:
    def __init__(self):
        # Use the pre-trained model for English to Korean translation
        model_name = 'Helsinki-NLP/opus-mt-en-ko'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        # Tokenize the input text and prepare for model input
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        # Generate translation output using the model
        outputs = self.model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
        # Decode the output and return the translated text
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

if __name__ == "__main__":
    # Example usage
    translator = EnglishKoreanTranslator()
    english_text = "Hello, how are you?"
    korean_translation = translator.translate(english_text)
    print(f"English: {english_text}")
    print(f"Korean: {korean_translation}")