import tensorflow as tf
from seq2seq import Encoder, Decoder
from preprocessing.load_data import load_dataset

# Load data
(input_tensor_train, target_tensor_train), (input_tensor_val, target_tensor_val), tokenizer_eng, tokenizer_kor = load_dataset('data/korean_english_dataset.csv')

# Hyperparameters
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer_eng.word_index) + 1
vocab_tar_size = len(tokenizer_kor.word_index) + 1

# Create the encoder and decoder
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# Optimizer and Loss
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
