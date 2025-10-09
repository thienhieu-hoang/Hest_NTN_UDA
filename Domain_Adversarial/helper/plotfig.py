import os
import numpy as np
import matplotlib.pyplot as plt

def figLoss(line_list=None, index_save=1, figure_save_path=None, fig_show=False, 
            fig_name=None, xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss'):
    """
    loss_list: List of tuples/lists [(loss_values1, 'Legend1'), (loss_values2, 'Legend2'), ...]
    """
    plt.figure(figsize=(10, 5))
    
    if line_list is not None:
        max_len = 0
        for loss_values, legend_name in line_list:
            x = range(0, len(loss_values) + 0)
            plt.plot(x, loss_values, label=legend_name)
            max_len = max(max_len, len(loss_values))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if max_len > 5:
            plt.xticks(range(0, max_len, int(max_len / 5)))
        else:
            plt.xticks(range(0, max_len, 1))
        if len(line_list) > 1:
            plt.legend()
    
    if figure_save_path is not None:
        os.makedirs(figure_save_path, exist_ok=True)
        save_path = os.path.join(figure_save_path, f"{index_save}{fig_name}")
        plt.savefig(save_path)
    
    if fig_show:
        plt.show()
    
    plt.clf()
    
def figChan(x, nmse =None, title=None, index_save=1, figure_save_path=None, name=None, fig_show=False):
    if x.ndim == 1:
        plt.figure(figsize=(2, 6))  # width=2 inches, height=6 inches
        plt.imshow(np.tile(x[:, np.newaxis], (1, 3)), aspect='auto', cmap='viridis')
        plt.colorbar()
        # plt.title(f"{input_condition}-Estimated channel at Symbol 2 Slot 1 as input condition")
        plt.xticks([])
        plt.ylabel('Subcarrier')
        plt.xlabel('Symbol 2, Slot 1')
        plt.title(title)
    else:
        plt.figure(figsize=(10, 5))
        plt.imshow(x,  aspect='auto', cmap='viridis', interpolation='none')
        plt.xlabel('OFDM symbol')
        plt.ylabel('Subcarrier')
    
        if nmse is not None:
            plt.title(f'{title}, NMSE: {nmse:.4f}')
        else:
            plt.title(title)
        plt.colorbar()
        if fig_show:
            plt.show()
            
    if figure_save_path is not None:
        os.makedirs(figure_save_path, exist_ok=True)
        plt.savefig(os.path.join(figure_save_path, 'epoch_' + str(index_save) + name), bbox_inches='tight')
    plt.clf()
    
def figTrueChan(x, title, index_save, figure_save_path, name):
    plt.figure(figsize=(10, 5))
    plt.imshow(x,  aspect='auto', cmap='viridis', interpolation='none')
    plt.xlabel('OFDM symbol')
    plt.ylabel('Subcarrier')
    plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join(figure_save_path,  str(index_save) + name) )
    plt.clf()
    
def figPredChan(x, title, y, index_save, figure_save_path, name):
    # x in cpu
    plt.figure(figsize=(10, 5))
    plt.imshow(x,  aspect='auto', cmap='viridis', interpolation='none')
    plt.xlabel('OFDM symbol')
    plt.ylabel('Subcarrier')
    plt.title(f'{title}, NMSE: {y:.4f}')
    plt.colorbar()
    plt.savefig(os.path.join(figure_save_path,  str(index_save) + name) )
    # plt.show()
    plt.clf()

def plotHist(X, fig_show=False, save_path=None, name=None):
    # to use:
        # plotHist(H_true_source, fig_show=True, save_path=f'{notebook_dir}/../generatedChan/', name='source')
    #
    if not isinstance(X, np.ndarray):
            X = np.array(X)

    if np.issubdtype(X.dtype, np.void):  # "|V16" means structured (real, imag)
        X = X.view(np.complex128).reshape(X.shape)
        # Flatten all elements to 1D for histogram
    X_flat = X.flatten()

    # Compute magnitude and phase
    magnitudes = np.abs(X_flat)
    phases = np.angle(X_flat)  # in radians, range [-π, π]

    # --- Plot magnitude histogram ---
    plt.figure(figsize=(10, 4))
    plt.hist(magnitudes, bins=400, color='blue', alpha=0.7)
    plt.title('Distribution of Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path + f'magnitude_{name}.svg')
    if fig_show:
        plt.show()
    plt.clf()

    # --- Plot phase histogram ---
    plt.figure(figsize=(10, 4))
    plt.hist(phases, bins=400, color='orange', alpha=0.7)
    plt.title('Distribution of Phases')
    plt.xlabel('Phase (radians)')
    plt.ylabel('Count')
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path + f'phase_{name}.svg')
    if fig_show:
        plt.show()
    plt.clf()
    
from scipy.stats import wasserstein_distance

def wasserstein_magnitude_only(X1, X2):
    """
    Compute Wasserstein-1 distance between the magnitude distributions
    of two complex-valued datasets.
    """
    # Convert to numpy arrays
    if not isinstance(X1, np.ndarray):
        X1 = np.array(X1)
    if np.issubdtype(X1.dtype, np.void):  # "|V16" means structured (real, imag)
        X1 = X1.view(np.complex128).reshape(X1.shape)
        
    if not isinstance(X2, np.ndarray):
        X2 = np.array(X2)
    if np.issubdtype(X2.dtype, np.void):  # "|V16" means structured (real, imag)
        X2 = X2.view(np.complex128).reshape(X2.shape)

    # Compute magnitudes
    mag1 = np.abs(X1).ravel()
    mag2 = np.abs(X2).ravel()

    # Compute 1D Wasserstein distance
    w_dist = wasserstein_distance(mag1, mag2)
    return w_dist


def wasserstein_approximate(X1, X2, reg=1e-2, n_samples=10000):
    """
    Approximate multi-dimensional Wasserstein (Sinkhorn) distance
    between two complex datasets, using the POT library.
    
    - reg: entropic regularization (higher = smoother, faster)
    - n_samples: number of random samples for computational efficiency
    """
    import ot  # POT = Python Optimal Transport

    # Convert to numpy arrays
    if not isinstance(X1, np.ndarray):
        X1 = np.array(X1)
    if np.issubdtype(X1.dtype, np.void):  # "|V16" means structured (real, imag)
        X1 = X1.view(np.complex128).reshape(X1.shape)
        
    if not isinstance(X2, np.ndarray):
        X2 = np.array(X2)
    if np.issubdtype(X2.dtype, np.void):  # "|V16" means structured (real, imag)
        X2 = X2.view(np.complex128).reshape(X2.shape)

    # Flatten into feature vectors (real + imag as 2 channels)
    X1_flat = np.stack([X1.real.ravel(), X1.imag.ravel()], axis=1)
    X2_flat = np.stack([X2.real.ravel(), X2.imag.ravel()], axis=1)

    # Randomly sample to reduce computational load
    n1, n2 = len(X1_flat), len(X2_flat)
    idx1 = np.random.choice(n1, min(n1, n_samples), replace=False)
    idx2 = np.random.choice(n2, min(n2, n_samples), replace=False)
    X1_sub = X1_flat[idx1]
    X2_sub = X2_flat[idx2]

    # Uniform weights
    a = np.ones((X1_sub.shape[0],)) / X1_sub.shape[0]
    b = np.ones((X2_sub.shape[0],)) / X2_sub.shape[0]

    # Compute Sinkhorn distance (regularized Wasserstein)
    M = ot.dist(X1_sub, X2_sub, metric='euclidean')
    sinkhorn_dist = ot.sinkhorn2(a, b, M, reg)
    return float(np.sqrt(sinkhorn_dist))