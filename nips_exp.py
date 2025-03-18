
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from itertools import combinations

def load_data(filename):
    print("Loading and pre-processing data...")
    data = sio.loadmat(filename)
    authors = np.array(data['authors']).flatten()
    
    # Convert sparse matrix to a dense NumPy array
    documents = data['documents']
    if hasattr(documents, 'toarray'):  # Check if it's a sparse matrix
        documents = documents.toarray()
    
    return authors, documents

def prepare_coauthorship_matrix(authors, documents):
    no_authors = len(authors)
    P = np.zeros((no_authors, no_authors))
    
    for i in range(documents.shape[0]):
        doc_authors = np.where(documents[i, :] > 0)[0]
        if len(doc_authors) > 1:
            for a, b in combinations(doc_authors, 2):
                P[a, b] += 1
                P[b, a] += 1
    
    return P

def filter_authors(P, authors, documents):
    no_papers = np.sum(documents, axis=0)
    valid_authors = np.where(no_papers >= 2)[0]
    P = P[np.ix_(valid_authors, valid_authors)]
    authors = authors[valid_authors]
    
    authors_without_coop = np.where(np.sum(P != 0, axis=1) <= 1)[0]
    P = np.delete(P, authors_without_coop, axis=0)
    P = np.delete(P, authors_without_coop, axis=1)
    authors = np.delete(authors, authors_without_coop)
    
    return P, authors, no_papers[valid_authors]

def normalize_P(P):
    np.fill_diagonal(P, 0)
    P = P / np.sum(P, axis=1, keepdims=True)
    P = P + P.T
    P /= np.sum(P)
    return P

def run_tsne(P, no_maps=5, no_dims=2, max_iter=500):
    print(f"Running multiple maps t-SNE to construct {no_maps} maps of {P.shape[0]} authors...")
    print("This may take up to 15 minutes to compute!")
    from mult_maps_tsne import mult_maps_tsne  # Assuming the function is implemented in the previous file
    return mult_maps_tsne(P, no_maps, no_dims, max_iter)

def plot_maps(maps, weights, authors, font_sizes):
    print("Drawing maps... please note that 2,000 iterations were used to produce the results in the paper!")
    
    for i in range(maps.shape[2]):
        plt.figure(figsize=(8, 6))
        
        ind = np.where(weights[:, i] > 0.05)[0]
        plt.scatter(maps[ind, 0, i], maps[ind, 1, i], s=20 * weights[ind, i], alpha=0.6)
        
        width = np.max(maps[ind, 0, i]) - np.min(maps[ind, 0, i])
        for j, author in enumerate(authors[ind]):
            #author_name = author.split('_')[0]  # Extract first name part
            plt.text(maps[ind[j], 0, i] + 0.004 * width, maps[ind[j], 1, i], 
                     author, fontsize=font_sizes[ind[j]])
        
        plt.axis('off')
        plt.title(f"t-SNE Map {i+1}")
        plt.show()

if __name__ == "__main__":
    filename = 'nips_1-22.mat'  # Update with actual file path
    authors, documents = load_data(filename)
    
    P = prepare_coauthorship_matrix(authors, documents)
    P, authors, no_papers = filter_authors(P, authors, documents)
    P = normalize_P(P)
    
    font_sizes = 6 + np.round(np.log10(no_papers) * 6).astype(int)
    maps, weights = run_tsne(P)
    
    plot_maps(maps, weights, authors, font_sizes)
