from transformers import MarianMTModel, MarianTokenizer

class EnglishKoreanTranslator:
    def __init__(self, model_name: str = 'Helsinki-NLP/opus-mt-en-ko'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text