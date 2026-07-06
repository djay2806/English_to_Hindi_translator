import gradio as gr
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# 1. Load Tokenizers
with open('english_tokenizer.pkl', 'rb') as f:
    english_tokenizer = pickle.load(f)
with open('hindi_tokenizer.pkl', 'rb') as f:
    hindi_tokenizer = pickle.load(f)

reverse_hindi_word_index = {index: word for word, index in hindi_tokenizer.word_index.items()}

# 2. Define Constants
MAX_ENGLISH_LEN = 41
MAX_HINDI_LEN = 48
ENCODER_UNITS = 128
DECODER_UNITS = 256

# 3. Load Model & Extract Layers
training_model = load_model("best_model.keras")

# Get the actual input from the loaded model (index 0 is encoder_inputs)
encoder_inputs = training_model.inputs[0] 

# Get the outputs from the layers by name
encoder_outputs = training_model.get_layer("encoder_bigru").output[0]
encoder_state_proj = training_model.get_layer("state_projection").output

inference_encoder_model = Model(encoder_inputs, [encoder_outputs, encoder_state_proj], name="inference_encoder")
inference_encoder_model = Model(encoder_inputs, [encoder_outputs, encoder_state_proj], name="inference_encoder")

# Extract individual trained layers for the decoder step
decoder_embedding_layer = training_model.get_layer("decoder_embedding")
decoder_gru = training_model.get_layer("decoder_gru")
attention_layer = training_model.get_layer("attention_layer")
context_concat_layer = training_model.get_layer("context_concat")
pre_output_dense = training_model.get_layer("pre_output_dense")
output_layer = training_model.get_layer("output_layer")

# Re-wire them into the single-step inference decoder
decoder_state_input = Input(shape=(DECODER_UNITS,), name="decoder_state_input")
encoder_outputs_input = Input(shape=(MAX_ENGLISH_LEN, 2 * ENCODER_UNITS), name="encoder_outputs_input")
decoder_input_single = Input(shape=(1,), name="decoder_input_single")

dec_emb_inf = decoder_embedding_layer(decoder_input_single)
dec_seq_out_inf, dec_state_inf = decoder_gru(dec_emb_inf, initial_state=decoder_state_input)

context_inf = attention_layer([dec_seq_out_inf, encoder_outputs_input])
combined_inf = context_concat_layer([dec_seq_out_inf, context_inf])
pre_out_inf = pre_output_dense(combined_inf)
dec_output_inf = output_layer(pre_out_inf)

inference_decoder_model = Model(
    [decoder_input_single, decoder_state_input, encoder_outputs_input],
    [dec_output_inf, dec_state_inf],
    name="inference_decoder"
)

# 4. Translation Functions
def clean_english(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate(sentence, max_len=MAX_HINDI_LEN, verbose=False):
    cleaned = clean_english(sentence)
    sequence = english_tokenizer.texts_to_sequences([cleaned])
    sequence = pad_sequences(sequence, maxlen=MAX_ENGLISH_LEN, padding="post", truncating="post")

    enc_outputs_val, state_val = inference_encoder_model.predict(sequence, verbose=0)

    current_word = np.array([[hindi_tokenizer.word_index["<start>"]]])
    translated_words = []
    
    # Get the OOV index for Hindi
    oov_index = hindi_tokenizer.word_index.get("<OOV>", 1)

    for step in range(max_len):
        prediction, state_val = inference_decoder_model.predict(
            [current_word, state_val, enc_outputs_val],
            verbose=0
        )

        # FORCE NO OOV: Set the probability of <OOV> to absolute zero
        prediction[0, 0, oov_index] = 0.0
        
        # Now get the predicted word
        predicted_index = int(np.argmax(prediction[0, 0]))
        predicted_word = reverse_hindi_word_index.get(predicted_index, "")

        if verbose:
            print(f"step {step+1}: token={predicted_index} word={predicted_word} "
                  f"confidence={float(np.max(prediction[0, 0])):.3f}")

        if predicted_index == 0 or predicted_word == "<end>":
            break

        translated_words.append(predicted_word)
        current_word = np.array([[predicted_index]])

    return " ".join(translated_words)

# 5. Build the Gradio Website Interface
interface = gr.Interface(
    fn=translate,                           
    inputs=gr.Textbox(label="English", placeholder="Type a formal English sentence here..."),     
    outputs=gr.Textbox(label="Hindi"),      
    title="English to Hindi Translator",
    description="Translation engine powered by a custom bidirectional GRU with attention. Try to give formal Sentences "

)

# Launch the site!
interface.launch()