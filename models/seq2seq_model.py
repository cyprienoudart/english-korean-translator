import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Embedding, Input, Concatenate, Dot, Activation
from tensorflow.keras.models import Model

class Seq2SeqModel:
    def __init__(self, 
                 vocab_size_eng,
                 vocab_size_kor,
                 embedding_dim=256,
                 units=1024,
                 max_length_eng=50,
                 max_length_kor=50):
        """
        Initialize the Seq2Seq model with attention mechanism
        
        Args:
            vocab_size_eng (int): Size of English vocabulary
            vocab_size_kor (int): Size of Korean vocabulary
            embedding_dim (int): Dimension of word embeddings
            units (int): Number of units in GRU layers
            max_length_eng (int): Maximum length of English sentences
            max_length_kor (int): Maximum length of Korean sentences
        """
        self.vocab_size_eng = vocab_size_eng
        self.vocab_size_kor = vocab_size_kor
        self.embedding_dim = embedding_dim
        self.units = units
        self.max_length_eng = max_length_eng
        self.max_length_kor = max_length_kor
    
    # Custom attention mechanism
    def _apply_attention(self, encoder_outputs, decoder_outputs):
        """Apply Bahdanau attention mechanism"""
        # Create attention weights
        W1 = Dense(self.units)(encoder_outputs)
        W2 = Dense(self.units)(decoder_outputs)
        
        # Calculate attention scores
        score = Dense(1)(tf.nn.tanh(W1 + tf.expand_dims(W2, 1)))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Calculate context vector
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    
    def build_model(self):
        """Build the complete encoder-decoder model with attention"""
        # Encoder inputs
        encoder_inputs = Input(shape=(self.max_length_eng,))
        
        # Encoder embedding
        encoder_embedding = Embedding(
            input_dim=self.vocab_size_eng,
            output_dim=self.embedding_dim
        )(encoder_inputs)
        
        # Bidirectional GRU for encoder to capture more context
        encoder_gru = GRU(
            units=self.units,
            return_sequences=True,
            return_state=True
        )
        
        encoder_outputs, encoder_state = encoder_gru(encoder_embedding)
        
        # Decoder inputs
        decoder_inputs = Input(shape=(self.max_length_kor,))
        
        # Decoder embedding
        decoder_embedding = Embedding(
            input_dim=self.vocab_size_kor,
            output_dim=self.embedding_dim
        )(decoder_inputs)
        
        # Decoder GRU
        decoder_gru = GRU(
            units=self.units,
            return_sequences=True,
            return_state=True
        )
        
        # Initial decoder call
        decoder_outputs, _ = decoder_gru(
            decoder_embedding,
            initial_state=encoder_state
        )
        
        # Attention mechanism
        context_vector, attention_weights = self._apply_attention(encoder_outputs, decoder_outputs)
        
        # Expand context vector for concatenation
        context_vector_expanded = tf.expand_dims(context_vector, 1)
        context_vector_tiled = tf.tile(
            context_vector_expanded, 
            [1, tf.shape(decoder_outputs)[1], 1]
        )
        
        # Concatenate decoder outputs with context vector
        decoder_combined_context = Concatenate(axis=-1)(
            [decoder_outputs, context_vector_tiled]
        )
        
        # Final output layer
        outputs = Dense(
            units=self.vocab_size_kor,
            activation='softmax'
        )(decoder_combined_context)
        
        # Create the full model
        model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=outputs
        )
        
        return model 