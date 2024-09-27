import re
# from nltk.tokenize import word_tokenize
import nltk
nltk.download('all')

# Clean and tokenize English and Korean sentences
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9가-힣\s]", "", sentence)  # Remove special characters
    sentence = word_tokenize(sentence)  # Tokenize the sentence
    return sentence

df['english_sentence'] = df['english_sentence'].apply(preprocess_sentence)
df['korean_sentence'] = df['korean_sentence'].apply(preprocess_sentence)

# Verify preprocessing
print(df.head())
