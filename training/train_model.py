import os
import sys
import tensorflow as tf
import numpy as np
import argparse
import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq_model import Seq2SeqModel
from training.train import TranslationTrainer
from preprocessing.data_loader import prepare_dataset

def train_model(model, train_dataset, val_dataset, epochs=20, learning_rate=0.001, checkpoint_dir="checkpoints"):
    """
    Train the translation model
    
    Args:
        model: TensorFlow model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        checkpoint_dir: Directory to save checkpoints
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(checkpoint_dir, 
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'ckpt-{epoch}'),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )
    
    # Create TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S')),
        histogram_freq=1
    )
    
    # Train model
    print(f"Starting training for {epochs} epochs")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
    
    # Save final model
    model_save_path = os.path.join(checkpoint_dir, 'final_model')
    model.save(model_save_path)
    
    print(f"Model saved to {model_save_path}")
    
    return model_save_path, history

def main(args):
    """Main training function"""
    # Prepare dataset
    train_dataset, val_dataset, eng_tokenizer, kor_tokenizer, vocab_size_eng, vocab_size_kor = prepare_dataset(
        data_path=args.data_path,
        eng_tokenizer_path=args.eng_tokenizer_path,
        kor_tokenizer_path=args.kor_tokenizer_path,
        vocab_size_eng=args.vocab_size_eng,
        vocab_size_kor=args.vocab_size_kor,
        max_length_eng=args.max_length_eng,
        max_length_kor=args.max_length_kor,
        batch_size=args.batch_size
    )
    
    if train_dataset is None:
        print("Failed to prepare dataset")
        return
    
    # Create model
    print(f"Creating model with vocab_size_eng={vocab_size_eng}, vocab_size_kor={vocab_size_kor}")
    model = Seq2SeqModel(
        vocab_size_eng=vocab_size_eng,
        vocab_size_kor=vocab_size_kor,
        embedding_dim=args.embedding_dim,
        units=args.units,
        max_length_eng=args.max_length_eng,
        max_length_kor=args.max_length_kor
    )
    
    # Build full model
    full_model = model.build_model()
    
    # Train model
    train_model(
        model=full_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save tokenizers
    if not args.eng_tokenizer_path:
        eng_tokenizer.save(os.path.join(args.checkpoint_dir, 'eng_tokenizer.pkl'))
    
    if not args.kor_tokenizer_path:
        kor_tokenizer.save(os.path.join(args.checkpoint_dir, 'kor_tokenizer.pkl'))
    
    print(f"Model saved to {args.checkpoint_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train English-Korean translation model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with English-Korean pairs')
    parser.add_argument('--eng_tokenizer_path', type=str, default=None,
                        help='Path to save/load English tokenizer')
    parser.add_argument('--kor_tokenizer_path', type=str, default=None,
                        help='Path to save/load Korean tokenizer')
    
    # Model parameters
    parser.add_argument('--vocab_size_eng', type=int, default=10000,
                        help='English vocabulary size')
    parser.add_argument('--vocab_size_kor', type=int, default=10000,
                        help='Korean vocabulary size')
    parser.add_argument('--max_length_eng', type=int, default=50,
                        help='Maximum English sequence length')
    parser.add_argument('--max_length_kor', type=int, default=50,
                        help='Maximum Korean sequence length')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--units', type=int, default=1024,
                        help='Number of units in GRU layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    main(args) 