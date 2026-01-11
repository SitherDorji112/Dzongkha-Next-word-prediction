
<img width="1834" height="899" alt="image" src="https://github.com/user-attachments/assets/c75d60ba-a1ba-4f6e-8a6d-0d3eeca75f28" />


# Project Overview

Dzongkha Next Word Prediction Using Deep Learning is a Natural Language Processing (NLP) project aimed at predicting the next Dzongkha syllable based on a given sequence of preceding syllables. The system is designed to support Dzongkha language computing and contribute to research on low-resource languages by modeling linguistic patterns from curated Dzongkha text data.

# Technologies Used

- Programming Language: Python

- Deep Learning Frameworks: TensorFlow, Keras

- NLP Techniques: Syllable-level tokenization, sequence modeling

- Development Environment: Jupyter Notebook (.ipynb)

- Selected Model Architecture: GRU-based Neural Language Model

# Dataset

The dataset was manually collected from several Dzongkha textbooks and supplemented with language resources from the Dzongkha Development Commission (DDC). Due to the limited availability of digital Dzongkha corpora, the dataset preparation involved careful manual collection, cleaning, and normalization to preserve linguistic accuracy.

- Source: Dzongkha textbooks and DDC resources

- Collection Method: Manual text collection and verification

- Text Processing:

    - Removal of non-Dzongkha characters and punctuation

    - Sentence segmentation using Dzongkha delimiters

    - Syllable-level tokenization using tsheg (་)

- Dataset Type: Syllable sequences for next-word (next-syllable) prediction

# Model Details

- Model Type: Neural Language Model for Next-Word Prediction

- Selected Architecture: Gated Recurrent Unit (GRU)

- Input Representation: Syllable index sequences generated from a fixed context window

- Embedding Layer: Learns dense syllable representations

- Training Strategy:

    - Train–validation–test split (80% / 10% / 10%)

    - Early stopping and learning rate reduction to prevent overfitting
 
# Evaluation Metrics

Perplexity was used as the primary evaluation metric, as lower perplexity indicates a stronger ability to model Dzongkha syllable sequence patterns. Training and validation loss were monitored to ensure proper convergence, and test perplexity was used to evaluate generalization performance.

The GRU model was selected based on its lower perplexity compared to BiLSTM and Transformer-based models, demonstrating better efficiency and sequence modeling capability for Dzongkha text.

# Use Cases

- Dzongkha text prediction and auto-completion systems

- Educational tools for Dzongkha language learning

- Assistive writing applications for Dzongkha users

- Foundation for advanced Dzongkha NLP tasks such as chatbots and machine translation

# Results

- The GRU-based model effectively learned Dzongkha syllable sequences from manually curated data

- Demonstrated reliable next-syllable prediction performance on unseen test data

<img width="1829" height="905" alt="image" src="https://github.com/user-attachments/assets/059ab184-86a7-490c-a15d-aa80f5bffee5" />
  

- Confirms the suitability of recurrent neural networks for low-resource language modeling
