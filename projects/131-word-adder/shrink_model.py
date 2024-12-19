from gensim.models import KeyedVectors
import os
import numpy as np

directory = './models'

if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_word(word):
    """
    Preprocess a word by lowercasing and removing non-alphabetic characters.
    """
    word = word.lower()  # Convert to lowercase
    word = ''.join(filter(str.isalpha, word))  # Keep only alphabetic characters
    return word if word else None  # Return None if the word becomes empty

print("Loading the model...")
model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
print('loaded')

for size in [10_000, 50_000, 100_000]:
    common_words = model.index_to_key[:size]

    processed_words = {}
    vectors = []

    for word in common_words:
        processed_word = preprocess_word(word)
        if processed_word and processed_word not in processed_words:
            processed_words[processed_word] = len(vectors)  # Map to index
            vectors.append(model[word])

    smaller_model = KeyedVectors(vector_size=model.vector_size)
    smaller_model.add_vectors(list(processed_words.keys()), np.array(vectors))

    model_name = f"./models/{size//1000}k_model.kv"
    smaller_model.save(model_name)
    print(f"Saved {model_name}")
