from flask import Flask, request, jsonify, render_template
from gensim.models import KeyedVectors

app = Flask(__name__)

# load the pre-trained word2vec model (use a smaller model for simplicity)
# download a model like 'GoogleNews-vectors-negative300.bin' and point to its location
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_word', methods=['POST'])
def find_word():
    word1 = request.form['word1'].strip().lower()
    word2 = request.form['word2'].strip().lower()

    if word1 not in model.key_to_index or word2 not in model.key_to_index:
        return jsonify({'error': 'One or both words are not in the vocabulary.'}), 400

    # add vectors and find the closest word
    vector_sum = model[word1] + model[word2]
    closest_word = model.most_similar(positive=[vector_sum], topn=1)[0][0]

    return jsonify({'result': closest_word})

if __name__ == '__main__':
    app.run(debug=True)
