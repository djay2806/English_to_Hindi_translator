# English to Hindi Neural Machine Translation

A Neural Machine Translation (NMT) system that translates English sentences into Hindi using a Sequence-to-Sequence (Seq2Seq) Encoder–Decoder architecture built with TensorFlow and the Keras Functional API.

## Features

- English to Hindi translation
- Seq2Seq Encoder–Decoder architecture
- Bidirectional GRU encoder
- GRU decoder
- Teacher Forcing during training
- Tokenization and sequence padding
- Functional API implementation
- Custom inference pipeline for sentence translation
- Model trained on a large English–Hindi parallel corpus

---

## Tech Stack

- Python
- TensorFlow
- Keras Functional API
- NumPy
- Pandas
- Scikit-learn
- Google Colab

---

## Dataset

- English–Hindi Parallel Corpus
- Source: Kaggle
- Over **1.3 Million** parallel sentence pairs

---

## Model Architecture

```
English Sentence
        │
        ▼
Embedding Layer
        │
        ▼
Bidirectional GRU Encoder
        │
        ▼
Encoder Hidden State
        │
        ▼
GRU Decoder
        │
        ▼
Dense + Softmax
        │
        ▼
Hindi Translation
```

---

## Training Details

- Embedding Dimension: **256**
- Encoder Units: **256**
- Decoder Units: **256**
- Batch Size: **16**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Teacher Forcing used during training

---

## Project Structure

```
English-Hindi-Translator/
│── Translation_Model.ipynb
│── app.py
│── requirements.txt
│── README.md
│── model/
│── screenshots/
```

---

## Installation

```bash
git clone https://github.com/yourusername/English-Hindi-Translator.git

cd English-Hindi-Translator

pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

---

## Future Improvements

- Implement Bahdanau Attention
- Replace GRU with Transformer architecture
- Beam Search decoding
- Improve translation quality using larger vocabulary
- Deploy on Streamlit Cloud

---

## 👨‍💻 Author

**Dhananjay Patil**

- GitHub: https://github.com/djay2806
