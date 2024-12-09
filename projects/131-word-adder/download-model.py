from gensim.models import KeyedVectors
import gensim.downloader as api

# download the model (takes time)
model = api.load("word2vec-google-news-300")

# save it locally for reuse
model.save_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
