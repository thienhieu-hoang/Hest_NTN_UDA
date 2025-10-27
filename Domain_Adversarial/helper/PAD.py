import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import h5py
from sklearn.kernel_approximation import RBFSampler
import tensorflow as tf
from tensorflow.keras import layers, models
import time
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def original_PAD(loader_source, loader_target):
    # Calculate initial PAD of 2 original training datasets
    loader_source.reset()
    loader_target.reset()
    all_features = []
    all_labels = []
    #
    for batch_idx in range(loader_target.total_batches):
        # batch_target: shape (batch_size, 792, 2)
        batch_target = loader_target.next_batch()  # (batch_size, 792,2)
        batch_target = batch_target[:,:, 0:2]
        real_target = batch_target['real']  # (batch_size, 792,2)
        imag_target = batch_target['imag']  # (batch_size, 792,2)
        real_flat = real_target.reshape(real_target.shape[0], -1)  # (batch_size, 1584)
        imag_flat = imag_target.reshape(imag_target.shape[0], -1)  # (batch_size, 1584)
        combined_target = np.concatenate([real_flat, imag_flat], axis=1)  # (batch_size, 3168)
        target_labels = (np.ones(combined_target.shape[0], dtype=int))
        # 
        batch_source = loader_source.next_batch()  # (batch_size, 792, 3)
        batch_source = batch_source[:,:,0:2]
        real_source = batch_source['real']  # (batch_size, 792,2)
        imag_source = batch_source['imag']  # (batch_size, 792,2)
        real_flat_source = real_source.reshape(real_source.shape[0], -1)  # (batch_size, 1584)
        imag_flat_source = imag_source.reshape(imag_source.shape[0], -1)  # (batch_size, 1584)
        combined_source = np.concatenate([real_flat_source, imag_flat_source], axis=1)  # (batch_size, 3168)
        source_labels = (np.zeros(combined_source.shape[0], dtype=int))

        # --- Combine and append ---
        all_features.append(combined_source)
        all_features.append(combined_target)
        all_labels.append(source_labels)
        all_labels.append(target_labels)

    # Stack all batches into a single dataset
    X = np.vstack(all_features)  # shape: (n_samples, 3168)
    y = np.concatenate(all_labels)  # shape: (n_samples,)
    print('X shape = ', X.shape)

    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)
    
    print('X1 shape = ', X1.shape, 'y1 shape = ', y1.shape)  
    print(X2.shape, y2.shape) 

    C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    best_epsilon = 1.0
    best_C = None
    for C in C_values:
        svm = SVC(C=C, probability=True)
        svm.fit(X1, y1)
        accuracy = svm.score(X2, y2)
        error_rate = 1 - accuracy
        print(f"C: {C}, Error rate: {error_rate:.4f}")
        if error_rate < best_epsilon:
            best_epsilon = error_rate
            best_C = C
    print(f"Best C: {best_C}, Best error rate: {best_epsilon:.4f}")
    pad = 2 * (1 - 2 * best_epsilon)
    print(f"PAD = {pad:.4f}")
    
    return pad 

def cal_PAD(features_source, features_target, num_samples=2048, pca_components=None):
    
    start = time.perf_counter()
    def load_h5_first(filename, num_samples):
        with h5py.File(filename, 'r') as f:
            dset = f['features']
            if num_samples is not None and num_samples < dset.shape[0]:
                features = dset[:num_samples]
            else:
                features = dset[:]
        return features
    
    if isinstance(features_source, str):
        print(f"Loading features_source from {features_source}")
        features_source = load_h5_first(features_source, num_samples)
    else:
        if num_samples is not None and num_samples < features_source.shape[0]:
            features_source = features_source[:num_samples]
    if isinstance(features_target, str):
        print(f"Loading features_target from {features_target}")
        features_target = load_h5_first(features_target, num_samples)
    else:
        if num_samples is not None and num_samples < features_target.shape[0]:
            features_target = features_target[:num_samples]

    X = np.vstack((features_source, features_target))
    X = X.reshape(X.shape[0], -1)  # Flatten the features if needed
    y = np.concatenate((np.zeros(features_source.shape[0]), np.ones(features_target.shape[0])))

    # PCA for dimensionality reduction
    if pca_components is not None and pca_components < X.shape[1]:
        print(f"Applying PCA to reduce to {pca_components} components")
        pca = PCA(n_components=pca_components, random_state=42)
        X = pca.fit_transform(X)
        import gc
        gc.collect()  # Optional, to free memory immediately

    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)

    C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    best_epsilon = 1.0
    best_C = None
    
    for C in C_values:
        svm = SVC(C=C, probability=True)
        svm.fit(X1, y1)
        accuracy = svm.score(X2, y2)
        error_rate = 1 - accuracy
        print(f"== C: {C}, Error rate: {error_rate:.4f}")
        
        print("Time", time.perf_counter() - start, "seconds")
        
        if error_rate < best_epsilon:
            best_epsilon = error_rate
            best_C = C
            
    print(f"Best C: {best_C}, Best error rate: {best_epsilon:.4f}")
    
    pad = 2 * (1 - 2 * best_epsilon)
    print(f"============ PAD (SVM) = {pad:.4f}")
    
    return pad

def cal_PAD_SGD(source_file, target_file):
  with h5py.File(source_file, 'r') as f_source, h5py.File(target_file, 'r') as f_target:
    n_source = f_source['features'].shape[0]
    n_target = f_target['features'].shape[0]
    total = n_source + n_target

    # Create index and label arrays
    indices = np.arange(total)
    labels = np.zeros(total, dtype=int)
    labels[n_source:] = 1  # target samples get label 1

    # # Shuffle indices
    np.random.seed(42)
    np.random.shuffle(indices)

    # Select half for your split
    split = total // 2
    train_indices = indices[:split]
    train_labels = labels[train_indices]
    test_indices = indices[split:]
    test_labels = labels[test_indices]

    batch_size = 16
    alpha_values = np.logspace(-6, 0, 10)
    best_epsilon_sgd = 1.0
    best_alpha = None
    
    # --- Train SGDClassifier batch by batch ---
    for alpha in alpha_values:
        clf = SGDClassifier(loss='hinge', alpha=alpha, max_iter=5000, tol=1e-5, random_state=42)
        classes = np.array([0, 1])
        first_batch = True
        

        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            batch_features = []
            for idx in batch_indices:
                if idx < n_source:
                    feat = f_source['features'][idx]
                else:
                    feat = f_target['features'][idx - n_source]
                batch_features.append(feat)
            batch_features = np.stack(batch_features, axis=0)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            if first_batch:
                clf.partial_fit(batch_features, batch_labels, classes=classes)
                first_batch = False
            else:
                clf.partial_fit(batch_features, batch_labels)

        # --- Test SGDClassifier batch by batch ---
        y_true = []
        y_pred = []
        for i in range(0, len(test_indices), batch_size):
            batch_indices = test_indices[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]
            batch_features = []
            for idx in batch_indices:
                if idx < n_source:
                    feat = f_source['features'][idx]
                else:
                    feat = f_target['features'][idx - n_source]
                batch_features.append(feat)
            batch_features = np.stack(batch_features, axis=0)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            preds = clf.predict(batch_features)
            y_true.extend(batch_labels)
            y_pred.extend(preds)

        error_rate = 1 - accuracy_score(y_true, y_pred)
        # print(f"alpha: {alpha}, Error rate: {error_rate:.4f}")
        if error_rate < best_epsilon_sgd:
            best_epsilon_sgd = error_rate
            best_alpha = alpha

        
    pad = 2 * (1 - 2 * best_epsilon_sgd)
    print(f"========= PAD (SGDClassifier) = {pad:.4f}")
  return pad

def original_PAD_SGD(loader_H_1_6_11_train_source, loader_H_1_6_11_train_target):
    """
    Calculate PAD of 2 original training datasets using SGDClassifier (for fair comparison with extracted features PAD).
    Similar to original_PAD, but uses SGDClassifier and sweeps over alpha values.
    """
    loader_H_1_6_11_train_source.reset()
    loader_H_1_6_11_train_target.reset()
    all_features = []
    all_labels = []
    #
    for batch_idx in range(loader_H_1_6_11_train_target.total_batches):
        # batch_target: shape (batch_size, 792, 2)
        batch_target = loader_H_1_6_11_train_target.next_batch()  # (batch_size, 792,2)
        batch_target = batch_target[:,:, 0:2]
        real_target = batch_target['real']  # (batch_size, 792,2)
        imag_target = batch_target['imag']  # (batch_size, 792,2)
        real_flat = real_target.reshape(real_target.shape[0], -1)  # (batch_size, 1584)
        imag_flat = imag_target.reshape(imag_target.shape[0], -1)  # (batch_size, 1584)
        combined_target = np.concatenate([real_flat, imag_flat], axis=1)  # (batch_size, 3168)
        target_labels = (np.ones(combined_target.shape[0], dtype=int))
        # 
        batch_source = loader_H_1_6_11_train_source.next_batch()  # (batch_size, 792, 3)
        batch_source = batch_source[:,:,0:2]
        real_source = batch_source['real']  # (batch_size, 792,2)
        imag_source = batch_source['imag']  # (batch_size, 792,2)
        real_flat_source = real_source.reshape(real_source.shape[0], -1)  # (batch_size, 1584)
        imag_flat_source = imag_source.reshape(imag_source.shape[0], -1)  # (batch_size, 1584)
        combined_source = np.concatenate([real_flat_source, imag_flat_source], axis=1)  # (batch_size, 3168)
        source_labels = (np.zeros(combined_source.shape[0], dtype=int))

        # --- Combine and append ---
        all_features.append(combined_source)
        all_features.append(combined_target)
        all_labels.append(source_labels)
        all_labels.append(target_labels)

    # Stack all batches into a single dataset
    X = np.vstack(all_features)  # shape: (n_samples, 3168)
    y = np.concatenate(all_labels)  # shape: (n_samples,)
    print('X shape = ', X.shape)

    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)

    print('X1 shape = ', X1.shape, 'y1 shape = ', y1.shape)  
    print(X2.shape, y2.shape) 

    alpha_values = np.logspace(-6, 0, 10)
    best_epsilon = 1.0
    best_alpha = None
    batch_size = 16
    for alpha in alpha_values:
        clf = SGDClassifier(loss='hinge', alpha=alpha, max_iter=5000, tol=1e-5, random_state=42)
        classes = np.array([0, 1])
        first_batch = True
        # --- Train SGDClassifier batch by batch ---
        for i in range(0, X1.shape[0], batch_size):
            batch_features = X1[i:i+batch_size].astype(np.float64)
            batch_labels = y1[i:i+batch_size]
            if first_batch:
                clf.partial_fit(batch_features, batch_labels, classes=classes)
                first_batch = False
            else:
                clf.partial_fit(batch_features, batch_labels)
        # --- Test SGDClassifier batch by batch ---
        y_true = []
        y_pred = []
        for i in range(0, X2.shape[0], batch_size):
            batch_features = X2[i:i+batch_size].astype(np.float64)
            batch_labels = y2[i:i+batch_size]
            preds = clf.predict(batch_features)
            y_true.extend(batch_labels)
            y_pred.extend(preds)
        error_rate = 1 - accuracy_score(y_true, y_pred)
        print(f"alpha: {alpha}, Error rate: {error_rate:.4f}")
        if error_rate < best_epsilon:
            best_epsilon = error_rate
            best_alpha = alpha
    print(f"Best alpha: {best_alpha}, Best error rate: {best_epsilon:.4f}")
    pad = 2 * (1 - 2 * best_epsilon)
    print(f"PAD (SGDClassifier) = {pad:.4f}")
    return pad

def original_PAD_KernelApprox_SGD(loader_H_1_6_11_train_source, loader_H_1_6_11_train_target, gamma=1.0, n_components=500):
    """
    Calculate PAD using kernel approximation (RBFSampler) + SGDClassifier, for nonlinear online classification.
    Similar to original_PAD_SGD, but uses RBFSampler to approximate RBF kernel.
    """
    loader_H_1_6_11_train_source.reset()
    loader_H_1_6_11_train_target.reset()
    all_features = []
    all_labels = []
    #
    for batch_idx in range(loader_H_1_6_11_train_target.total_batches):
        # batch_target: shape (batch_size, 792, 2)
        batch_target = loader_H_1_6_11_train_target.next_batch()  # (batch_size, 792,2)
        batch_target = batch_target[:,:, 0:2]
        real_target = batch_target['real']  # (batch_size, 792,2)
        imag_target = batch_target['imag']  # (batch_size, 792,2)
        real_flat = real_target.reshape(real_target.shape[0], -1)  # (batch_size, 1584)
        imag_flat = imag_target.reshape(imag_target.shape[0], -1)  # (batch_size, 1584)
        combined_target = np.concatenate([real_flat, imag_flat], axis=1)  # (batch_size, 3168)
        target_labels = (np.ones(combined_target.shape[0], dtype=int))
        # 
        batch_source = loader_H_1_6_11_train_source.next_batch()  # (batch_size, 792, 3)
        batch_source = batch_source[:,:,0:2]
        real_source = batch_source['real']  # (batch_size, 792,2)
        imag_source = batch_source['imag']  # (batch_size, 792,2)
        real_flat_source = real_source.reshape(real_source.shape[0], -1)  # (batch_size, 1584)
        imag_flat_source = imag_source.reshape(imag_source.shape[0], -1)  # (batch_size, 1584)
        combined_source = np.concatenate([real_flat_source, imag_flat_source], axis=1)  # (batch_size, 3168)
        source_labels = (np.zeros(combined_source.shape[0], dtype=int))

        # --- Combine and append ---
        all_features.append(combined_source)
        all_features.append(combined_target)
        all_labels.append(source_labels)
        all_labels.append(target_labels)

    # Stack all batches into a single dataset
    X = np.vstack(all_features)  # shape: (n_samples, 3168)
    y = np.concatenate(all_labels)  # shape: (n_samples,)
    print('X shape = ', X.shape)

    from sklearn.model_selection import train_test_split
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)
    
    print('X1 shape = ', X1.shape, 'y1 shape = ', y1.shape)  
    print(X2.shape, y2.shape) 

    # Fit RBFSampler on training set
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    rbf_feature.fit(X1)
    X1_rbf = rbf_feature.transform(X1)
    X2_rbf = rbf_feature.transform(X2)

    alpha_values = np.logspace(-6, 0, 10)
    best_epsilon = 1.0
    best_alpha = None
    batch_size = 16
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    for alpha in alpha_values:
        clf = SGDClassifier(loss='hinge', alpha=alpha, max_iter=5000, tol=1e-5, random_state=42)
        classes = np.array([0, 1])
        first_batch = True
        # --- Train SGDClassifier batch by batch ---
        for i in range(0, X1_rbf.shape[0], batch_size):
            batch_features = X1_rbf[i:i+batch_size].astype(np.float64)
            batch_labels = y1[i:i+batch_size]
            if first_batch:
                clf.partial_fit(batch_features, batch_labels, classes=classes)
                first_batch = False
            else:
                clf.partial_fit(batch_features, batch_labels)
        # --- Test SGDClassifier batch by batch ---
        y_true = []
        y_pred = []
        for i in range(0, X2_rbf.shape[0], batch_size):
            batch_features = X2_rbf[i:i+batch_size].astype(np.float64)
            batch_labels = y2[i:i+batch_size]
            preds = clf.predict(batch_features)
            y_true.extend(batch_labels)
            y_pred.extend(preds)
        error_rate = 1 - accuracy_score(y_true, y_pred)
        print(f"alpha: {alpha}, Error rate: {error_rate:.4f}")
        if error_rate < best_epsilon:
            best_epsilon = error_rate
            best_alpha = alpha
    print(f"Best alpha: {best_alpha}, Best error rate: {best_epsilon:.4f}")
    pad = 2 * (1 - 2 * best_epsilon)
    print(f"PAD (KernelApprox+SGDClassifier) = {pad:.4f}")
    return pad

def original_PAD_CNN(loader_H_1_6_11_train_source, loader_H_1_6_11_train_target, epochs=10, batch_size=32, verbose=1):
    """
    Calculate PAD using a simple CNN classifier (1D CNN for flat features).
    Inputs: loader_H_1_6_11_train_source, loader_H_1_6_11_train_target
    Returns: PAD value (float)
    """
    loader_H_1_6_11_train_source.reset()
    loader_H_1_6_11_train_target.reset()
    all_features = []
    all_labels = []
    #
    for batch_idx in range(loader_H_1_6_11_train_target.total_batches):
        # batch_target: shape (batch_size, 792, 2)
        batch_target = loader_H_1_6_11_train_target.next_batch()  # (batch_size, 792,2)
        batch_target = batch_target[:,:, 0:2]
        real_target = batch_target['real']  # (batch_size, 792,2)
        imag_target = batch_target['imag']  # (batch_size, 792,2)
        real_flat = real_target.reshape(real_target.shape[0], -1)  # (batch_size, 1584)
        imag_flat = imag_target.reshape(imag_target.shape[0], -1)  # (batch_size, 1584)
        combined_target = np.concatenate([real_flat, imag_flat], axis=1)  # (batch_size, 3168)
        target_labels = (np.ones(combined_target.shape[0], dtype=int))
        # 
        batch_source = loader_H_1_6_11_train_source.next_batch()  # (batch_size, 792, 3)
        batch_source = batch_source[:,:,0:2]
        real_source = batch_source['real']  # (batch_size, 792,2)
        imag_source = batch_source['imag']  # (batch_size, 792,2)
        real_flat_source = real_source.reshape(real_source.shape[0], -1)  # (batch_size, 1584)
        imag_flat_source = imag_source.reshape(imag_source.shape[0], -1)  # (batch_size, 1584)
        combined_source = np.concatenate([real_flat_source, imag_flat_source], axis=1)  # (batch_size, 3168)
        source_labels = (np.zeros(combined_source.shape[0], dtype=int))

        # --- Combine and append ---
        all_features.append(combined_source)
        all_features.append(combined_target)
        all_labels.append(source_labels)
        all_labels.append(target_labels)

    # Stack all batches into a single dataset
    X = np.vstack(all_features)  # shape: (n_samples, 3168)
    y = np.concatenate(all_labels)  # shape: (n_samples,)
    print('X shape = ', X.shape)

    from sklearn.model_selection import train_test_split
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)

    print('X1 shape = ', X1.shape, 'y1 shape = ', y1.shape)  
    print(X2.shape, y2.shape) 

    # Reshape for 1D CNN: (samples, features, 1)
    X1_cnn = X1[..., np.newaxis]
    X2_cnn = X2[..., np.newaxis]

    # Build simple 1D CNN model
    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=(X1_cnn.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train
    model.fit(X1_cnn, y1, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(X2_cnn, y2))

    # Test
    y_pred = np.argmax(model.predict(X2_cnn), axis=1)
    acc = accuracy_score(y2, y_pred)
    error = 1 - acc
    pad = 2 * (1 - 2 * error)
    print(f"Test accuracy: {acc:.4f}, Error: {error:.4f}, PAD: {pad:.4f}")
    return pad

def cal_PAD2(features_source_h5, features_target_h5, num_samples=2048, pca_components=100, batch_size=128):
    """
    loading batch by batch to avoid exploading RAM
    loading .h5 files batch by batch, fitting IncrementalPCA, transforming, then SVM.
    """
    print(f"Fitting IncrementalPCA on batches from {features_source_h5} and {features_target_h5}")
    # Open both files
    with h5py.File(features_source_h5, 'r') as f_source, h5py.File(features_target_h5, 'r') as f_target:
        dset_source = f_source['features']
        dset_target = f_target['features']
        n_source = dset_source.shape[0]
        n_target = dset_target.shape[0]
        n_source = min(n_source, num_samples) if num_samples is not None else n_source
        n_target = min(n_target, num_samples) if num_samples is not None else n_target
        total = n_source + n_target
        # Fit IncrementalPCA
        ipca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)
        # First pass: partial_fit
        idx_source = 0
        idx_target = 0
        while idx_source < n_source or idx_target < n_target:
            batch_source = dset_source[idx_source:min(idx_source+batch_size, n_source)]
            batch_target = dset_target[idx_target:min(idx_target+batch_size, n_target)]
            batch = np.concatenate([batch_source, batch_target], axis=0)
            batch = batch.reshape(batch.shape[0], -1)
            ipca.partial_fit(batch)
            idx_source += batch_size
            idx_target += batch_size
            print(f"Fitted PCA on batch: source {idx_source}/{n_source}, target {idx_target}/{n_target}")
        # Second pass: transform and collect reduced features
        reduced_source = []
        reduced_target = []
        for idx in range(0, n_source, batch_size):
            batch = dset_source[idx:min(idx+batch_size, n_source)]
            batch = batch.reshape(batch.shape[0], -1)
            reduced_source.append(ipca.transform(batch))
        for idx in range(0, n_target, batch_size):
            batch = dset_target[idx:min(idx+batch_size, n_target)]
            batch = batch.reshape(batch.shape[0], -1)
            reduced_target.append(ipca.transform(batch))
        X_source = np.vstack(reduced_source)
        X_target = np.vstack(reduced_target)
        print(f"Reduced source shape: {X_source.shape}, target shape: {X_target.shape}")
    # Stack and label
    X = np.vstack((X_source, X_target))
    y = np.concatenate((np.zeros(X_source.shape[0]), np.ones(X_target.shape[0])))
    # Train/test split
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    # SVM
    C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    best_epsilon = 1.0
    best_C = None
    for C in C_values:
        svm = SVC(C=C, probability=True)
        svm.fit(X1, y1)
        accuracy = svm.score(X2, y2)
        error_rate = 1 - accuracy
        print(f"== C: {C}, Error rate: {error_rate:.4f}")
        if error_rate < best_epsilon:
            best_epsilon = error_rate
            best_C = C
    print(f"Best C: {best_C}, Best error rate: {best_epsilon:.4f}")
    pad = 2 * (1 - 2 * best_epsilon)
    print(f"============ PAD (SVM, batch PCA) = {pad:.4f}")
    return pad

def extract_features_with_pca(features_source_h5, features_target_h5, 
                            num_samples=2048, pca_components=100, batch_size=128):
    """
    if features_source_h5 is a string, it is a path to .h5 file
        Load features from .h5 files batch by batch, fit IncrementalPCA, 
        return reduced features X and domain labels y.
    """
    if isinstance(features_source_h5, str) and isinstance(features_target_h5, str):
        print(f"Fitting IncrementalPCA on batches from {features_source_h5} and {features_target_h5}")
        
        # Open both files
        with h5py.File(features_source_h5, 'r') as f_source, h5py.File(features_target_h5, 'r') as f_target:
            dset_source = f_source['features']
            dset_target = f_target['features']

            n_source = min(dset_source.shape[0], num_samples) if num_samples else dset_source.shape[0]
            n_target = min(dset_target.shape[0], num_samples) if num_samples else dset_target.shape[0]

            # Initialize IncrementalPCA
            ipca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)

            # First pass: partial_fit
            idx_source = idx_target = 0
            while idx_source < n_source or idx_target < n_target:
                batch_source = dset_source[idx_source:min(idx_source+batch_size, n_source)]
                batch_target = dset_target[idx_target:min(idx_target+batch_size, n_target)]
                batch = np.concatenate([batch_source, batch_target], axis=0).reshape(-1, np.prod(dset_source.shape[1:]))
                ipca.partial_fit(batch)
                idx_source += batch_size
                idx_target += batch_size
                print(f"Fitted PCA on batch: source {min(idx_source, n_source)}/{n_source}, target {min(idx_target, n_target)}/{n_target}")

            # Second pass: transform and collect reduced features
            reduced_source, reduced_target = [], []

            for idx in range(0, n_source, batch_size):
                batch = dset_source[idx:min(idx+batch_size, n_source)].reshape(-1, np.prod(dset_source.shape[1:]))
                reduced_source.append(ipca.transform(batch))

            for idx in range(0, n_target, batch_size):
                batch = dset_target[idx:min(idx+batch_size, n_target)].reshape(-1, np.prod(dset_target.shape[1:]))
                reduced_target.append(ipca.transform(batch))

            X_source = np.vstack(reduced_source)
            X_target = np.vstack(reduced_target)
            print(f"Reduced source shape: {X_source.shape}, target shape: {X_target.shape}")
    else:
        H_input_train_source_numpy = h5loader_to_numpy_v2(features_source_h5)
        H_input_train_target_numpy = h5loader_to_numpy_v2(features_target_h5)

        H_input_train_source_real = complex_to_real_stack(H_input_train_source_numpy)
        H_input_train_target_real = complex_to_real_stack(H_input_train_target_numpy)
        dset_source = H_input_train_source_real
        dset_target = H_input_train_target_real
        n_source = min(dset_source.shape[0], num_samples) if num_samples else dset_source.shape[0]
        n_target = min(dset_target.shape[0], num_samples) if num_samples else dset_target.shape[0]

        # Initialize IncrementalPCA
        ipca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)

        # First pass: partial_fit
        idx_source = idx_target = 0
        while idx_source < n_source or idx_target < n_target:
            batch_source = dset_source[idx_source:min(idx_source+batch_size, n_source)]
            batch_target = dset_target[idx_target:min(idx_target+batch_size, n_target)]
            batch = np.concatenate([batch_source, batch_target], axis=0).reshape(-1, np.prod(dset_source.shape[1:]))
            ipca.partial_fit(batch)
            idx_source += batch_size
            idx_target += batch_size
            print(f"Fitted PCA on batch: source {min(idx_source, n_source)}/{n_source}, target {min(idx_target, n_target)}/{n_target}")

        # Second pass: transform and collect reduced features
        reduced_source, reduced_target = [], []

        for idx in range(0, n_source, batch_size):
            batch = dset_source[idx:min(idx+batch_size, n_source)].reshape(-1, np.prod(dset_source.shape[1:]))
            reduced_source.append(ipca.transform(batch))

        for idx in range(0, n_target, batch_size):
            batch = dset_target[idx:min(idx+batch_size, n_target)].reshape(-1, np.prod(dset_target.shape[1:]))
            reduced_target.append(ipca.transform(batch))

        X_source = np.vstack(reduced_source)
        X_target = np.vstack(reduced_target)
        print(f"Reduced source shape: {X_source.shape}, target shape: {X_target.shape}")
    # Stack and label
    X = np.vstack((X_source, X_target))
    y = np.concatenate((np.zeros(X_source.shape[0]), np.ones(X_target.shape[0])))

    return X, y

def calc_pad_svm(X, y):
    """
    Calculate Proxy A-distance (PAD) using SVM.
    Input: 
        X - features (after PCA)
        y - domain labels (0=source, 1=target)
    """
    # Split into train/test
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)

    # Standardize features
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)
    X2_scaled = scaler.transform(X2)

    # Try multiple C values
    C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    best_epsilon = 1.0
    best_C = None

    for C in C_values:
        svm = SVC(C=C, probability=True)
        svm.fit(X1_scaled, y1)
        accuracy = svm.score(X2_scaled, y2)
        error_rate = 1 - accuracy
        print(f"== C: {C}, Error rate: {error_rate:.4f}")
        if error_rate < best_epsilon:
            best_epsilon = error_rate
            best_C = C

    print(f"Best C: {best_C}, Best error rate: {best_epsilon:.4f}")
    if best_epsilon>0.5: 
        print(f"Flip the predictions")
        best_epsilon = 1- best_epsilon

    # Compute PAD
    pad = 2 * (1 - 2 * best_epsilon)
    print(f"============ PAD (SVM) = {pad:.4f}")

    return pad

def calc_pad_lda(X, y):
    """
    Calculate Proxy A-distance (PAD) using Linear Discriminant Analysis (LDA).
    Input:
        X - features (after PCA)
        y - domain labels (0=source, 1=target)
    """
    # Split train/test
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X1, y1)
    accuracy = lda.score(X2, y2)
    error_rate = 1 - accuracy

    print(f"LDA Error rate: {error_rate:.4f}")
    if error_rate>0.5: 
        print(f"Flip the predictions")
        error_rate = 1- error_rate

    # Compute PAD
    pad = 2 * (1 - 2 * error_rate)
    print(f"============ PAD (LDA) = {pad:.4f}")

    return pad

from sklearn.linear_model import LogisticRegression

def calc_pad_logreg(X, y):
    """
    Calculate Proxy A-distance (PAD) using Logistic Regression.
    Input:
        X - features (after PCA)
        y - domain labels (0=source, 1=target)
    """
    # Split train/test
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)

    logreg = LogisticRegression(max_iter=500, solver='lbfgs')
    logreg.fit(X1, y1)
    accuracy = logreg.score(X2, y2)
    error_rate = 1 - accuracy

    print(f"Logistic Regression Error rate: {error_rate:.4f}")
    if error_rate>0.5: 
        print(f"Flip the predictions")
        error_rate = 1- error_rate

    # Compute PAD
    pad = 2 * (1 - 2 * error_rate)
    print(f"============ PAD (LogReg) = {pad:.4f}")

    return pad

def h5loader_to_numpy_v2(loader):
    loader.reset()  # Reset to start from beginning
    all_data = []
    
    # Try different possible method names
    if hasattr(loader, 'get_batch'):
        # Original approach
        for batch_idx in range(loader.total_batches):
            batch_data = loader.get_batch()
            all_data.append(batch_data)
    elif hasattr(loader, 'next_batch'):
        # Alternative method name
        for batch_idx in range(loader.total_batches):
            batch_data = loader.next_batch()
            all_data.append(batch_data)
    elif hasattr(loader, '__next__'):
        # Iterator approach
        for batch_idx in range(loader.total_batches):
            batch_data = next(loader)
            all_data.append(batch_data)
    elif hasattr(loader, 'get_data'):
        # Another possible method name
        for batch_idx in range(loader.total_batches):
            batch_data = loader.get_data()
            all_data.append(batch_data)
    else:
        # Direct data access if available
        if hasattr(loader, 'dataset') and hasattr(loader, 'shuffled_indices'):
            return loader.dataset[loader.shuffled_indices]
    
    # Concatenate all batches
    return np.concatenate(all_data, axis=0)

def complex_to_real_stack(complex_array):
    """Convert complex array to real array by stacking real and imaginary parts"""
    if complex_array.dtype.names:  # Structured array
        real_part = complex_array['real']
        imag_part = complex_array['imag']
    else:  # Regular complex array
        real_part = np.real(complex_array)
        imag_part = np.imag(complex_array)
    
    # Stack along the last dimension: (..., 2) where [..., 0] = real, [..., 1] = imag
    # Ensure the result is float64
    return np.stack([real_part, imag_part], axis=-1).astype(np.float64)