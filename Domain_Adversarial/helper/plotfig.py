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