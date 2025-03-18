from matplotlib import pyplot as plt
import numpy as np


def plot_tsne_mnistemb(X,y):
        plt.figure()
        
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        colormap = cm.get_cmap('viridis', num_classes) 
        
        color_dict = {cls: colormap(i / num_classes) for i, cls in enumerate(unique_classes)} 
        
        for cls in unique_classes:
            mask = y == cls 
            plt.scatter(X[mask, 0], X[mask, 1], c=[color_dict[cls]], s=3, label=f'Class {cls}')
        
        plt.legend()
        plt.show()

def plot_tsne_wordemb(embedding, words, filename="tsne_wordemb_plot.png"):
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of embedding
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)
    
    # Add text labels
    x_min, x_max = plt.xlim()
    width = x_max - x_min
    for i in range(len(words)):
        plt.text(embedding[i, 0] + 0.006 * width, embedding[i, 1], words[i], fontsize=8)
    
    plt.axis('off')
    plt.title("t-SNE Embedding")
    
    # Save plot
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

def plot_multiple_maps(maps, weights, words):
    print("Drawing maps...")
    output_dir = "output/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for m in range(maps.shape[2]):
        plt.figure(figsize=(8, 6))
        
        # Filter words with sufficient importance weight
        indices = np.where(weights[:, m] > 0.05)[0] if maps.shape[2] > 1 else np.arange(maps.shape[0])
        
        # Scatter plot
        plt.scatter(maps[indices, 0, m], maps[indices, 1, m], s=weights[indices, m] * 40, alpha=0.6)
        
        # Add text labels
        x_min, x_max = plt.xlim()
        width = x_max - x_min
        for i in indices:
            plt.text(maps[i, 0, m] + 0.006 * width, maps[i, 1, m], words[i], fontsize=8)
        
        plt.axis('off')
        plt.title(f"t-SNE Map {m+1}")
        plt.savefig(output_dir+"multiple_map_"+str(m)+'.png')
        plt.show()