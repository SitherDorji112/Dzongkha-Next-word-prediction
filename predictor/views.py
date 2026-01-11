from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, Bidirectional, Embedding
import os

# Register compatible GRU layer that ignores time_major parameter
class CompatibleGRU(GRU):
    """GRU layer that ignores deprecated time_major parameter."""
    
    def __init__(self, units, **kwargs):
        # Remove deprecated time_major parameter
        kwargs.pop('time_major', None)
        super().__init__(units, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config_copy = config.copy()
        config_copy.pop('time_major', None)
        return cls(**config_copy)

# Base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Updated relative paths
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'dzongkha_gru_model.h5')
MAPPING_PATH = os.path.join(BASE_DIR, 'model', 'gru_syllable_mappings.pkl')

# Create custom objects - include both CompatibleGRU and GRU with the fix
custom_objects = {
    'GRU': CompatibleGRU,  # Map GRU to our compat version
    'CompatibleGRU': CompatibleGRU,
    'Bidirectional': Bidirectional,
    'Embedding': Embedding,
}

# Load GRU model
model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)

# Load syllable mappings
with open(MAPPING_PATH, "rb") as f:
    data = pickle.load(f)

syllable_to_index = data["syllable_to_index"]
index_to_syllable = data["index_to_syllable"]
sequence_length = data["sequence_length"]

def predict_top_syllables(seed_text, top_k=10):
    """
    Predict top-k next syllables based on given Dzongkha text.
    Also supports partial syllable suggestions and handles Tshag markers.
    """
    seed_text = seed_text.strip()

    # 🔹 Convert Dzongkha syllable markers to spaces
    seed_text = seed_text.replace("་", " ")

    if not seed_text:
        return []

    seed_list = seed_text.split()
    partial_input = None

    # Detect if last token might be incomplete (no space at end)
    if not seed_text.endswith(" "):
        partial_input = seed_list[-1]
        seed_list = seed_list[:-1]  # remove partial from prediction context

    # Convert syllables to indices
    token_list = [syllable_to_index.get(s, syllable_to_index.get("<UNK>", 0))
                  for s in seed_list[-(sequence_length - 1):]]

    # Pad input sequence
    token_list = np.pad(token_list, (sequence_length - 1 - len(token_list), 0), 'constant')
    token_list = np.array(token_list).reshape(1, sequence_length - 1)

    # Predict next-syllable probabilities
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    predictions = [index_to_syllable[i].replace("<EOS>", "།") for i in top_indices]

    # Filter predictions by partial syllable if user is typing
    if partial_input:
        predictions = [p for p in predictions if p.startswith(partial_input)] + predictions
        seen = set()
        predictions = [x for x in predictions if not (x in seen or seen.add(x))]

    return predictions


def index(request):
    """Render main prediction page"""
    return render(request, "predictor/index.html")


def about(request):
    """Render about page"""
    return render(request, "predictor/about.html")


def predict_ajax(request):
    """Handle AJAX requests for predictions"""
    text = request.GET.get("text", "")
    predictions = predict_top_syllables(text, top_k=10)
    return JsonResponse({"predictions": predictions})
