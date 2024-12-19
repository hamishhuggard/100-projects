from flask import Flask, request, jsonify, render_template
from gensim.models import KeyedVectors
import numpy as np

import gensim.downloader

# Show all available models in gensim-data
#print("Models available:", list(gensim.downloader.info()['models'].keys()))

# Download the "glove-twitter-25" embeddings
#model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
model = KeyedVectors.load("models/100k_model.kv")
print('loaded the model')
#model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

app = Flask(__name__)

def filter_similar_words(results, query, threshold=0.6):
    """
    Filters out words from results that are too similar to the query based on character overlap.

    :param results: List of tuples (word, similarity_score) from model.most_similar.
    :param query: Original query word as a string.
    :param threshold: Jaccard similarity threshold for filtering (default: 0.6).
    :return: Filtered list of results.
    """
    def jaccard_similarity(word1, word2):
        set1, set2 = set(word1.lower()), set(word2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    filtered_results = [
        (word, score) for word, score in results
        if jaccard_similarity(query, word) < threshold
    ]
    return filtered_results

@app.route('/')
def home():
    return render_template('index.html')

# 1. Closest Words
@app.route('/closest_words', methods=['GET', 'POST'])
def closest_words():
    if request.method == 'POST':
        word = request.form['word'].strip().lower()
        if word not in model.key_to_index:
            return jsonify({'error': 'Word is not in the vocabulary.'}), 400

        closest = model.most_similar(positive=[word], topn=10)
        closest = filter_similar_words(closest, query, threshold=0.6)
        return jsonify({'closest': closest})

    return render_template('closest_words.html')

# 2. Closest Word to Addition/Subtraction
@app.route('/closest_math', methods=['GET', 'POST'])
def closest_math():
    if request.method == 'POST':
        word1 = request.form['word1'].strip().lower()
        word2 = request.form['word2'].strip().lower()
        operation = request.form.get('operation', 'add').strip().lower()

        if word1 not in model.key_to_index or word2 not in model.key_to_index:
            return jsonify({'error': 'One or both words are not in the vocabulary.'}), 400

        result_vec = model[word1] + model[word2] if operation == 'add' else model[word1] - model[word2]
        closest = model.most_similar(positive=[result_vec], topn=10)
        return jsonify({'closest': closest})

    return render_template('closest_math.html')

# 3. Similarity Between Words (Coords)
@app.route('/similarity_coords', methods=['GET', 'POST'])
def similarity_coords():
    if request.method == 'POST':
        word_x = request.form['word_x'].strip().lower()
        word_y = request.form['word_y'].strip().lower()
        word_list = request.form.getlist('word_list[]')

        if word_x not in model.key_to_index or word_y not in model.key_to_index:
            return jsonify({'error': 'Base words are not in the vocabulary.'}), 400

        results = {}
        for word in word_list:
            if word in model.key_to_index:
                sim_x = model.similarity(word, word_x)
                sim_y = model.similarity(word, word_y)
                results[word] = {'x': sim_x, 'y': sim_y}

        return jsonify({'results': results})

    return render_template('similarity_coords.html')

# 4. Two axes: word + antiword
@app.route('/projection_coords', methods=['GET', 'POST'])
def projection_coords():
    if request.method == 'POST':
        word_x = request.form['word_x'].strip().lower()
        word_neg_x = request.form['word_neg_x'].strip().lower()
        word_y = request.form['word_y'].strip().lower()
        word_neg_y = request.form['word_neg_y'].strip().lower()
        word_list = request.form['content']
        word_list = word_list.lower()
        word_list = word_list.split()

        missing_coords = [ word for word in [word_x, word_neg_x, word_y, word_neg_y] if word not in model.key_to_index ]
        if len(missing_coords) > 0:
            return jsonify({ 'error': 'Not in the vocabulary: '+', '.join(missing_coords) }), 400

        # Compute the direction vectors
        x_axis = model[word_x] - model[word_neg_x]
        y_axis = model[word_y] - model[word_neg_y]

        # Project each word onto the axes
        results = []
        for word in word_list:
            if word in model.key_to_index:
                word_vec = model[word]
                proj_x = np.dot(word_vec, x_axis) / np.linalg.norm(x_axis)
                proj_y = np.dot(word_vec, y_axis) / np.linalg.norm(y_axis)
                results.append({'word': word, 'x': float(proj_x), 'y': float(proj_y)})

        return jsonify(results)

    return render_template('projection_coords.html')

# 5. Two words are two dimensions
@app.route('/find_projection', methods=['GET', 'POST'])
def find_projection():
    if request.method == 'GET':
        return render_template('projection')
    word1 = request.form['word1'].strip().lower()
    word2 = request.form['word2'].strip().lower()

    if word1 not in model.key_to_index or word2 not in model.key_to_index:
        return jsonify({'error': 'One or both words are not in the vocabulary.'}), 400

    # Retrieve the vectors for the two words
    vec1 = model[word1]
    vec2 = model[word2]

    # Create a 2D plane using vec1 and vec2
    plane_basis = np.array([vec1, vec2]).T
    plane_basis = np.linalg.qr(plane_basis)[0]  # Orthogonalize the basis

    # Find the top 10 most similar words to word1 and word2
    similar_words = model.most_similar(positive=[word1, word2], topn=10)

    # Project each word vector onto the plane
    projections = []
    for word, _ in similar_words:
        vec = model[word]
        # Calculate projection onto the plane
        coords = plane_basis.T @ vec
        projections.append({'word': word, 'x': coords[0], 'y': coords[1]})

    return jsonify({'projections': projections})

if __name__ == '__main__':
    app.run(debug=True)
