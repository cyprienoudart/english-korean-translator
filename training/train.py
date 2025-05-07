import tensorflow as tf
from models.seq2seq_model import Seq2SeqModel
import numpy as np

# Load data
(input_tensor_train, target_tensor_train), (input_tensor_val, target_tensor_val), tokenizer_eng, tokenizer_kor = load_dataset('data/korean_english_dataset.csv')

# Hyperparameters
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer_eng.word_index) + 1
vocab_tar_size = len(tokenizer_kor.word_index) + 1

class TranslationTrainer:
    def __init__(self,
                 model,
                 learning_rate=0.001,
                 batch_size=64):
        """
        Initialize the trainer
        
        Args:
            model: Seq2Seq model instance
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
        """
        self.model = model
        self.batch_size = batch_size
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Initialize loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        
    def loss_function(self, real, pred):
        """Calculate loss with masking for padding"""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    
    @tf.function
    def train_step(self, encoder_input, decoder_input, target):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model([encoder_input, decoder_input])
            
            # Calculate loss
            loss = self.loss_function(target, predictions)
            
        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        return loss
    
    def train(self, dataset, epochs):
        """Train the model"""
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch, (encoder_input, decoder_input, target) in enumerate(dataset):
                loss = self.train_step(encoder_input, decoder_input, target)
                total_loss += loss
                num_batches += 1
                
                if batch % 100 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch}, Loss: {loss:.4f}')
            
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

# Instantiate Seq2Seq model
model = Seq2SeqModel(vocab_inp_size, vocab_tar_size, embedding_dim, units)

# Create TranslationTrainer instance
trainer = TranslationTrainer(model)

# Train the model
EPOCHS = 10
trainer.train((input_tensor_train, target_tensor_train), EPOCHS)
