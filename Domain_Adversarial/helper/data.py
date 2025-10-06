import h5py

def temp_save_features():
    # temporarily save the features for calculateing PAD
    with h5py.File('all_features.h5', 'w') as f:
        dset_feats = f.create_dataset('features', shape=(0, feature_dim), maxshape=(None, feature_dim), dtype='float32', chunks=True)
        dset_labels = f.create_dataset('labels', shape=(0,), maxshape=(None,), dtype='int32', chunks=True)
        
        for batch_features, batch_labels in data_loader:
            feats = batch_features.cpu().numpy()
            labels = batch_labels.cpu().numpy()
            n = feats.shape[0]
            
            # Resize the dataset
            dset_feats.resize((dset_feats.shape[0] + n, feature_dim))
            dset_labels.resize((dset_labels.shape[0] + n,))
            
            # Append new data
            dset_feats[-n:] = feats
            dset_labels[-n:] = labels