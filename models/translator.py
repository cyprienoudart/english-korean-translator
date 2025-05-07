import tensorflow as tf
import numpy as np
from models.seq2seq_model import Seq2SeqModel
from preprocessing.tokenizer import Tokenizer

class Translator:
    def __init__(self, 
                 model_path,
                 eng_tokenizer_path,
                 kor_tokenizer_path,
                 max_length_eng=50,
                 max_length_kor=50):
        """
        Initialize translator with model and tokenizers
        
        Args:
            model_path (str): Path to saved model
            eng_tokenizer_path (str): Path to English tokenizer
            kor_tokenizer_path (str): Path to Korean tokenizer
            max_length_eng (int): Maximum English sequence length
            max_length_kor (int): Maximum Korean sequence length
        """
        # Load tokenizers
        self.eng_tokenizer = Tokenizer.load(eng_tokenizer_path)
        self.kor_tokenizer = Tokenizer.load(kor_tokenizer_path)
        
        # Set max lengths
        self.max_length_eng = max_length_eng
        self.max_length_kor = max_length_kor
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Extract encoder and decoder
        self._extract_encoder_decoder()
    
    def _extract_encoder_decoder(self):
        """Extract encoder and decoder from full model"""
        # This is a simplified version - in a real implementation,
        # you might need to reconstruct the encoder and decoder based on your model architecture
        
        # Placeholder implementation - this would need to be adapted to your specific model structure
        self.encoder = self.model.layers[0]
        self.decoder = self.model.layers[1]
    
    def translate(self, text, beam_width=3):
        """
        Translate English text to Korean
        
        Args:
            text (str): English text to translate
            beam_width (int): Width for beam search
            
        Returns:
            str: Translated Korean text
        """
        # Encode input text
        encoder_input = np.array([self.eng_tokenizer.encode(text)])
        
        # Initialize states
        encoder_outputs, encoder_state = self.encoder(encoder_input)
        
        # Initialize decoder input
        decoder_input = np.array([[self.kor_tokenizer.word_to_index[self.kor_tokenizer.special_tokens["START"]]]])
        
        # Initialize beam search
        beams = [(0.0, decoder_input, [])]  # (score, decoder_input, output_tokens)
        
        # Beam search
        for _ in range(self.max_length_kor):
            new_beams = []
            
            for score, decoder_input, output_tokens in beams:
                # Get decoder output
                decoder_output, decoder_state = self.decoder([decoder_input, encoder_outputs, encoder_state])
                
                # Get top k predictions
                top_k_indices = tf.argsort(
                    decoder_output[0, -1], direction='DESCENDING'
                )[:beam_width].numpy()
                
                top_k_probs = tf.gather(
                    decoder_output[0, -1], top_k_indices
                ).numpy()
                
                # Add new beams
                for idx, prob in zip(top_k_indices, top_k_probs):
                    new_score = score - np.log(prob + 1e-10)  # Negative log probability
                    new_output = output_tokens + [idx]
                    
                    # If END token, add to final beams
                    if idx == self.kor_tokenizer.word_to_index[self.kor_tokenizer.special_tokens["END"]]:
                        new_beams.append((new_score, decoder_input, new_output))
                    else:
                        # Update decoder input
                        new_decoder_input = np.array([[idx]])
                        new_beams.append((new_score, new_decoder_input, new_output))
            
            # Sort and keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[0])[:beam_width]
            
            # Stop if all beams end with END token
            if all(beams[i][2][-1] == self.kor_tokenizer.word_to_index[self.kor_tokenizer.special_tokens["END"]] 
                  for i in range(len(beams))):
                break
        
        # Get best translation
        best_translation = beams[0][2]
        
        # Decode
        return self.kor_tokenizer.decode(best_translation) 