import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.seq2seq import BasicDecoder, AttentionWrapper
from tensorflow.keras.layers import Embedding, GRU, Dense
from preprocessing.load_data import load_dataset

# Load data
(input_tensor_train, target_tensor_train), (input_tensor_val, target_tensor_val), tokenizer_eng, tokenizer_kor = load_dataset('data/korean_english_dataset.csv')

# Hyperparameters
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer_eng.word_index) + 1
vocab_tar_size = len(tokenizer_kor.word_index) + 1

# Define Encoder using TensorFlow/Keras API
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# Define Decoder using TensorFlow/Keras API
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)

        # For attention mechanism
        self.attention = tfa.seq2seq.BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # Apply attention
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # Embedding
        x = self.embedding(x)

        # Concatenate context vector and embedding input
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # GRU
        output, state = self.gru(x)

        # Dense output layer
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state

# Instantiate Encoder and Decoder
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# Optimizer and Loss function
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Training loop
EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (input_tensor, target_tensor)) in enumerate(zip(input_tensor_train, target_tensor_train)):
        with tf.GradientTape() as tape:
            enc_hidden = encoder.initialize_hidden_state()
            enc_output, enc_hidden = encoder(input_tensor, enc_hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([tokenizer_kor.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing: Feed the target as the next input
            for t in range(1, target_tensor.shape[1]):
                predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
                loss = loss_function(target_tensor[:, t], predictions)

                total_loss += loss
                dec_input = tf.expand_dims(target_tensor[:, t], 1)  # Teacher forcing

        gradients = tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))

    print(f'Epoch {epoch+1}, Loss: {total_loss.numpy():.4f}')
