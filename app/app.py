from flask import Flask, request, jsonify
import tensorflow as tf
from preprocessing.load_data import load_dataset
from models.seq2seq import Seq2SeqModel

app = Flask(__name__)

# Load the model and tokenizers
model = Seq2SeqModel()
(input_tensor_train, target_tensor_train), (input_tensor_val, target_tensor_val), tokenizer_eng, tokenizer_kor = load_dataset('data/korean_english_dataset.csv')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    english_sentence = data['sentence']

    # Tokenize and pad the English sentence
    input_seq = tokenizer_eng.texts_to_sequences([english_sentence])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=input_tensor_train.shape[1], padding='post')

    # Predict the translation using the model
    prediction = model.evaluate(input_seq)
    predicted_seq = tf.argmax(prediction, axis=-1)

    # Convert the prediction back to words
    korean_translation = ' '.join([tokenizer_kor.index_word[i] for i in predicted_seq[0] if i != 0])

    return jsonify({'translation': korean_translation})

if __name__ == '__main__':
    app.run(debug=True)
