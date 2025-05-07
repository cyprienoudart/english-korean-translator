import argparse
import os
import sys
import tensorflow as tf

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.seq2seq_model import Seq2SeqModel
from preprocessing.data_loader import prepare_dataset, download_dataset
from training.train_model import train_model
from models.translator import Translator

def main():
    """Main entry point for the English-Korean Translator"""
    parser = argparse.ArgumentParser(description="English-Korean Neural Machine Translation System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download dataset command
    download_parser = subparsers.add_parser("download", help="Download and prepare dataset")
    download_parser.add_argument("--url", type=str, required=True, help="URL to download dataset from")
    download_parser.add_argument("--output", type=str, default="data/eng_kor_dataset.csv", 
                               help="Path to save the downloaded dataset")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train translation model")
    train_parser.add_argument("--data_path", type=str, required=True, 
                             help="Path to CSV file with English-Korean pairs")
    train_parser.add_argument("--vocab_size_eng", type=int, default=10000, 
                             help="English vocabulary size")
    train_parser.add_argument("--vocab_size_kor", type=int, default=10000, 
                             help="Korean vocabulary size")
    train_parser.add_argument("--max_length_eng", type=int, default=50, 
                             help="Maximum English sequence length")
    train_parser.add_argument("--max_length_kor", type=int, default=50, 
                             help="Maximum Korean sequence length")
    train_parser.add_argument("--embedding_dim", type=int, default=256, 
                             help="Embedding dimension")
    train_parser.add_argument("--units", type=int, default=1024, 
                             help="Number of units in GRU layers")
    train_parser.add_argument("--batch_size", type=int, default=64, 
                             help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=20, 
                             help="Number of epochs for training")
    train_parser.add_argument("--learning_rate", type=float, default=0.001, 
                             help="Learning rate for optimizer")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                             help="Directory to save checkpoints")
    
    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate English text to Korean")
    translate_parser.add_argument("--model_path", type=str, required=True, 
                                help="Path to saved model")
    translate_parser.add_argument("--eng_tokenizer_path", type=str, required=True, 
                                help="Path to English tokenizer")
    translate_parser.add_argument("--kor_tokenizer_path", type=str, required=True, 
                                help="Path to Korean tokenizer")
    translate_parser.add_argument("--text", type=str, help="English text to translate")
    translate_parser.add_argument("--interactive", action="store_true", 
                                help="Run in interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "download":
        print(f"Downloading dataset from {args.url} to {args.output}")
        success = download_dataset(args.url, args.output)
        if success:
            print("Dataset downloaded successfully")
        else:
            print("Failed to download dataset")
            
    elif args.command == "train":
        print(f"Training model with data from {args.data_path}")
        # Set up tokenizer paths
        eng_tokenizer_path = os.path.join(args.checkpoint_dir, "eng_tokenizer.pkl")
        kor_tokenizer_path = os.path.join(args.checkpoint_dir, "kor_tokenizer.pkl")
        
        # Prepare dataset
        train_dataset, val_dataset, eng_tokenizer, kor_tokenizer, vocab_size_eng, vocab_size_kor = prepare_dataset(
            data_path=args.data_path,
            eng_tokenizer_path=eng_tokenizer_path,
            kor_tokenizer_path=kor_tokenizer_path,
            vocab_size_eng=args.vocab_size_eng,
            vocab_size_kor=args.vocab_size_kor,
            max_length_eng=args.max_length_eng,
            max_length_kor=args.max_length_kor,
            batch_size=args.batch_size
        )
        
        if train_dataset is not None:
            # Create and train model
            model = Seq2SeqModel(
                vocab_size_eng=vocab_size_eng,
                vocab_size_kor=vocab_size_kor,
                embedding_dim=args.embedding_dim,
                units=args.units,
                max_length_eng=args.max_length_eng,
                max_length_kor=args.max_length_kor
            )
            
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
            
    elif args.command == "translate":
        # Load translator
        translator = Translator(
            model_path=args.model_path,
            eng_tokenizer_path=args.eng_tokenizer_path,
            kor_tokenizer_path=args.kor_tokenizer_path
        )
        
        if args.interactive:
            print("Interactive translation mode. Type 'exit' to quit.")
            while True:
                text = input("Enter English text: ")
                if text.lower() == 'exit':
                    break
                
                translation = translator.translate(text)
                print(f"Korean translation: {translation}")
        elif args.text:
            translation = translator.translate(args.text)
            print(f"English: {args.text}")
            print(f"Korean: {translation}")
        else:
            print("Error: Either --text or --interactive must be specified")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 