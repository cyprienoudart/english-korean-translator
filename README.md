# English-Korean Translator

A powerful and accurate English to Korean translation system built with state-of-the-art deep learning techniques. This project implements a sequence-to-sequence (Seq2Seq) model using TensorFlow and provides a user-friendly web interface through Flask.

## 🌟 Features

- High-accuracy English to Korean translation
- Web-based interface for easy access
- Built on TensorFlow 2.17.0 for optimal performance
- Utilizes advanced neural network architectures
- Real-time translation capabilities
- Support for both short phrases and longer texts

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone hhttps://github.com/cyprienoudart/english-korean-translator.git
cd english-korean-translator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Overview

This project implements a deep learning-based translation system that converts English sentences to Korean with high accuracy. It uses a sequence-to-sequence architecture with attention mechanisms to handle the complexities of translation between these structurally different languages.

## Features

- Advanced neural machine translation using seq2seq with attention
- RESTful API for easy integration with web applications
- Pre-trained models for immediate use
- Customizable training pipeline for fine-tuning on specific domains
- Efficient tokenization and preprocessing for both languages
- Support for batch translation

## 🛠️ Technical Stack

- **Backend Framework**: Flask 2.0.0+
- **Deep Learning Framework**: TensorFlow 2.17.0
- **Natural Language Processing**: NLTK 3.6.7+
- **Data Processing**: NumPy 1.23.0+, Pandas 1.5.0+
- **Machine Learning Utilities**: scikit-learn 0.24.2+
- **Additional Features**: tensorflow-addons 0.18.0+

## 📚 Model Architecture

The translation system uses a sophisticated Seq2Seq model with:
- GRU (Gated Recurrent Unit) layers for sequence processing
- Embedding layers for word representation
- Attention mechanisms for improved translation accuracy
- Custom tokenization and preprocessing pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- NLTK team for the natural language processing tools
- The open-source community for various libraries and tools used in this project

## 📧 Contact

For any questions or suggestions, please open an issue in the GitHub repository.

---

Made with ❤️ by Cyprien Oudart