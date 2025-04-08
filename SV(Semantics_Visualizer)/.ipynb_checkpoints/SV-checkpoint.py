# Tool for dimensionality reduction
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# A class to load and work with pre-trained word embeddings
from gensim.models import KeyedVectors

# Load pre-trained word2vec model trained on Google News Data
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
# The file contains word embeddings where each word is represented as a 300-dimensional vector
model = KeyedVectors.load_word2vec_format('/home/gad/Documents/OFFLINE REPOS/FoNu_NLP_TG/SV(Semantics_Visualizer)/GoogleNews-vectors-negative300.bin', binary=True)

words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'boy', 'girl']
vectors = [model[word] for word in words]

# PCA object to reduce the data to 2 dimensions (so it can be plotted in a 2D space)
pca = PCA(n_components=2)
# Transform the vectors to 2 dimensions
reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.title('Semantic Relationships between Words')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()