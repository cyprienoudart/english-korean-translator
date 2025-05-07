from flask import Flask, request, jsonify, render_template
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.translator import Translator

app = Flask(__name__)

# Load translator
translator = None

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """API endpoint for translation"""
    # Get text from request
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Translate text
    try:
        translation = translator.translate(text)
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

def load_translator(model_path, eng_tokenizer_path, kor_tokenizer_path):
    """Load translator model"""
    global translator
    translator = Translator(
        model_path=model_path,
        eng_tokenizer_path=eng_tokenizer_path,
        kor_tokenizer_path=kor_tokenizer_path
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Start translation API server')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--eng_tokenizer_path', type=str, required=True,
                        help='Path to English tokenizer')
    parser.add_argument('--kor_tokenizer_path', type=str, required=True,
                        help='Path to Korean tokenizer')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    
    args = parser.parse_args()
    
    # Load translator
    load_translator(
        model_path=args.model_path,
        eng_tokenizer_path=args.eng_tokenizer_path,
        kor_tokenizer_path=args.kor_tokenizer_path
    )
    
    # Start server
    app.run(host=args.host, port=args.port)
