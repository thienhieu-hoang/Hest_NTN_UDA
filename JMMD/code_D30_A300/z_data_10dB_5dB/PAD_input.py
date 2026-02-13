"""
Script to analyze PAD (Proxy A-Distance) before and after FDA translation
Refactored with reusable functions for unscaled, scaled, and FDA versions
"""

import tensorflow as tf
import os
import sys
import numpy as np
from scipy.io import savemat
import h5py
import matplotlib.pyplot as plt

# Add the root project directory
try:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(code_dir, '..', '..', '..'))
except NameError:
    code_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(code_dir, '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Code dir:", code_dir)
print("Project root:", project_root)

from sklearn.decomposition import IncrementalPCA, PCA
from Domain_Adversarial.helper import loader, plotfig, PAD
from Domain_Adversarial.helper.utils import H5BatchLoader, minmaxScaler, complx2real, deMinMax
from Domain_Adversarial.helper.utils_GAN import visualize_H
from JMMD.helper.utils_GAN import save_checkpoint_jmmd as save_checkpoint

# ============================================================================
# CONFIGURATION
# ============================================================================

SNR_SOURCE = 10
SNR_TARGET = 5

source_data_file_path = os.path.abspath(os.path.join(
    code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_D_30_sim', 
    f'SNR_{SNR_SOURCE}dB', 'matlabNTN.mat'
))

target_data_file_path = os.path.abspath(os.path.join(
    code_dir, '..', '..', '..', 'generatedChan', 'MATLAB', 'TDL_A_300_sim', 
    f'SNR_{SNR_TARGET}dB', 'matlabNTN.mat'
))

norm_approach = 'minmax'
lower_range = -1

batch_size = 8
pca_components_first = 256 # 512
pca_components_second = 100
incremental_batch_size = 64
max_fitting_batches = 5 #3

fda_win_h = 13
fda_win_w = 3

# Paths to save
path_temp = code_dir + f'/results/'
os.makedirs(path_temp, exist_ok=True)
idx_save_path = loader.find_incremental_filename(path_temp, 'ver', '_', '')
save_path = path_temp + f'ver{idx_save_path}_'
os.makedirs(save_path, exist_ok=True)

print(f"\nSaving results to: {save_path}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_fda_translation(x_src_real, x_tgt_real, win_h=13, win_w=3):
    """Apply FDA translation: mix source input with target style
    input shapes: (N, H, W, 2) where last dim is [real, imag]
    output shape: (N, H, W, 2) also [real, imag]
    """
    from JMMD.helper.utils_GAN import F_extract_DD, fda_mix_pixels, F_inverse_DD, apply_phase_to_amplitude
    
    # Convert to complex
    x_src_complex = tf.complex(x_src_real[:,:,:,0], x_src_real[:,:,:,1])
    x_tgt_complex = tf.complex(x_tgt_real[:,:,:,0], x_tgt_real[:,:,:,1])
    
    # Extract amplitude and phase
    src_amplitude, src_phase = F_extract_DD(x_src_complex)
    tgt_amplitude, tgt_phase = F_extract_DD(x_tgt_complex)
    
    # Mix amplitudes (low frequency from target, rest from source)
    mixed_amplitude = fda_mix_pixels(src_amplitude, tgt_amplitude, win_h, win_w)
    
    # Reconstruct with source phase
    mixed_complex_dd = apply_phase_to_amplitude(mixed_amplitude, src_phase)
    x_fda_mixed_complex = F_inverse_DD(mixed_complex_dd)
    
    # Convert back to real
    x_fda_mixed = tf.stack(
        [tf.math.real(x_fda_mixed_complex), tf.math.imag(x_fda_mixed_complex)], 
        axis=-1
    )
    
    return x_fda_mixed.numpy()

def plot_magnitude_phase_histograms(
    data_real, save_prefix="", stage_name="", label="",
    mag_figsize=(10, 5), phase_figsize=(10, 5), label_font=20, numbins=500  
):
    """
    Plot histograms of magnitude and phase for complex data
    
    Args:
        data_real: Real data with shape (N, H, W, 2) where last dim is [real, imag]
        save_prefix: Filename prefix for saving figures
        stage_name: Name for plot title (unused)
        label: Source or Target label
        mag_figsize: (width, height) for magnitude figure
        phase_figsize: (width, height) for phase figure
    """
    # Convert to complex (handle both NumPy and TensorFlow)
    if tf.is_tensor(data_real):
        data_complex = tf.complex(data_real[..., 0], data_real[..., 1])
        magnitude = tf.abs(data_complex).numpy().flatten()
        phase = tf.math.angle(data_complex).numpy().flatten()
    else:
        data_complex = data_real[..., 0] + 1j * data_real[..., 1]
        magnitude = np.abs(data_complex).flatten()
        phase = np.angle(data_complex).flatten()
    
    # Magnitude figure
    fig_mag, ax_mag = plt.subplots(1, 1, figsize=mag_figsize)
    ax_mag.hist(magnitude, bins=numbins, color='blue', alpha=0.7, edgecolor='none')
    ax_mag.set_xlabel('Magnitude', fontsize=label_font)
    ax_mag.set_ylabel('Count', fontsize=label_font)
    ax_mag.grid(True, alpha=0.3)
    if magnitude.mean()<1e-4:
        ax_mag.text(0.98, 0.98, f'Mean: {magnitude.mean()*1e8:.3f}e8\nStd: {magnitude.std()*1e8:.3f}e8',
                    transform=ax_mag.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_mag.text(0.98, 0.98, f'Mean: {magnitude.mean():.4f}\nStd: {magnitude.std():.4f}',
                    transform=ax_mag.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig_mag.tight_layout()
    save_path_mag = f'{save_path}/{save_prefix}_{label.lower()}_magnitude_hist.pdf'
    fig_mag.savefig(save_path_mag, dpi=150, bbox_inches='tight')
    print(f"✓ Saved magnitude histogram to: {save_path_mag}")
    plt.close(fig_mag)
    
    # Phase figure
    fig_phase, ax_phase = plt.subplots(1, 1, figsize=phase_figsize)
    ax_phase.hist(phase, bins=numbins, color='red', alpha=0.7, edgecolor='none')
    ax_phase.set_xlabel('Phase (radians)', fontsize=label_font)
    ax_phase.set_ylabel('Count', fontsize=label_font)
    ax_phase.grid(True, alpha=0.3)
    ax_phase.text(0.98, 0.98, f'Mean: {phase.mean():.4f}\nStd: {phase.std():.4f}',
                transform=ax_phase.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig_phase.tight_layout()
    save_path_phase = f'{save_path}/{save_prefix}_{label.lower()}_phase_hist.pdf'
    fig_phase.savefig(save_path_phase, dpi=150, bbox_inches='tight')
    print(f"✓ Saved phase histogram to: {save_path_phase}")
    plt.close(fig_phase)
    
from Domain_Adversarial.helper.PAD import apply_incremental_pca_and_save, load_and_calculate_pad



# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

source_file = h5py.File(source_data_file_path, 'r')
H_true_source = source_file['H_perfect']
H_input_source = source_file['H_li']  # H_li H_prac
N_samp_source = H_true_source.shape[0]

target_file = h5py.File(target_data_file_path, 'r')
H_true_target = target_file['H_perfect']
H_input_target = target_file['H_li']  # H_li H_prac
N_samp_target = H_true_target.shape[0]

print(f'N_samp_source = {N_samp_source}')
print(f'N_samp_target = {N_samp_target}')

# For testing
# N_use_source = 96
# N_use_target = 96

# For full analysis (uncomment)
N_use_source = int(N_samp_source * 0.9)
N_use_target = int(N_samp_target * 0.9)

print(f"Using {N_use_source} source and {N_use_target} target samples")

# Extract and convert data
H_true_source_use = H_true_source[:N_use_source]
H_input_source_use = H_input_source[:N_use_source]
H_true_target_use = H_true_target[:N_use_target]
H_input_target_use = H_input_target[:N_use_target]

H_input_source_real = complx2real(H_input_source_use)
H_input_source_real = np.transpose(H_input_source_real, (0, 2, 3, 1))

H_input_target_real = complx2real(H_input_target_use)
H_input_target_real = np.transpose(H_input_target_real, (0, 2, 3, 1))

H_true_source_real = complx2real(H_true_source_use)
H_true_source_real = np.transpose(H_true_source_real, (0, 2, 3, 1))

H_true_target_real = complx2real(H_true_target_use)
H_true_target_real = np.transpose(H_true_target_real, (0, 2, 3, 1))

print(f"H_input_source: {H_input_source_real.shape}")
print(f"H_input_target: {H_input_target_real.shape}")

# ============================================================================
# SCALE INPUTS
# ============================================================================

print("\n" + "="*80)
print("SCALING INPUTS TO [-1, 1]")
print("="*80)

H_input_source_scaled, x_min_src, x_max_src = minmaxScaler(
    H_input_source_real, lower_range=lower_range, linear_interp=False
)

H_input_target_scaled, x_min_tgt, x_max_tgt = minmaxScaler(
    H_input_target_real, lower_range=lower_range, linear_interp=False
)

print("✓ Scaling complete")

# ============================================================================
# STEP 1A: PAD UNSCALED RAW INPUTS
# ============================================================================

h5_path_source_unscaled = f'{save_path}/features_source_unscaled.h5'
h5_path_target_unscaled = f'{save_path}/features_target_unscaled.h5'

# Plot histograms before calculating PAD
print("\nPlotting magnitude/phase histograms for UNSCALED inputs...")
plot_magnitude_phase_histograms(H_input_source_real, '1a_unscaled_raw', 'UNSCALED', 'Source')
plot_magnitude_phase_histograms(H_input_target_real, '1a_unscaled_raw', 'UNSCALED', 'Target')


pca_src_unscaled, pca_tgt_unscaled, var_src_unscaled, var_tgt_unscaled = \
    apply_incremental_pca_and_save(
        H_input_source_real, H_input_target_real,
        h5_path_source_unscaled, h5_path_target_unscaled,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="UNSCALED"
    )

pad_svm_unscaled, X_unscaled, y_unscaled = load_and_calculate_pad(
    h5_path_source_unscaled, h5_path_target_unscaled,
    pca_components_second=pca_components_second,
    stage_name="UNSCALED"
)
# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_unscaled):
    os.remove(h5_path_source_unscaled)
    print(f"✓ Deleted: {h5_path_source_unscaled}")
if os.path.exists(h5_path_target_unscaled):
    os.remove(h5_path_target_unscaled)
    print(f"✓ Deleted: {h5_path_target_unscaled}")
    
# ============================================================================
# STEP 1B: PAD SCALED INPUTS
# ============================================================================

h5_path_source_scaled = f'{save_path}/features_source_scaled.h5'
h5_path_target_scaled = f'{save_path}/features_target_scaled.h5'

# Plot histograms before calculating PAD
print("\nPlotting magnitude/phase histograms for SCALED inputs...")
plot_magnitude_phase_histograms(H_input_source_scaled, '1b_scaled_raw', 'SCALED', 'Source')
plot_magnitude_phase_histograms(H_input_target_scaled, '1b_scaled_raw', 'SCALED', 'Target')


pca_src_scaled, pca_tgt_scaled, var_src_scaled, var_tgt_scaled = \
    apply_incremental_pca_and_save(
        H_input_source_scaled, H_input_target_scaled,
        h5_path_source_scaled, h5_path_target_scaled,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="SCALED"
    )

pad_svm_scaled, X_scaled, y_scaled = load_and_calculate_pad(
    h5_path_source_scaled, h5_path_target_scaled,
    pca_components_second=pca_components_second,
    stage_name="SCALED"
)
    
# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_scaled):
    os.remove(h5_path_source_scaled)
    print(f"✓ Deleted: {h5_path_source_scaled}")
if os.path.exists(h5_path_target_scaled):
    os.remove(h5_path_target_scaled)
    print(f"✓ Deleted: {h5_path_target_scaled}")

# ============================================================================
# STEP 2: FDA TRANSLATION (on original unscaled inputs)
# ============================================================================

print("\n" + "="*80)
print("APPLYING FDA TRANSLATION")
print("="*80)

print(f"\nApplying FDA with window ({fda_win_h}, {fda_win_w})...")
H_input_source_fda = []
H_truePseudo = []

for i in range(N_use_source):
    if (i + 1) % 20 == 0:
        print(f"  Processing sample {i+1}/{N_use_source}")
    
    tgt_idx = i % N_use_target
    x_fda = apply_fda_translation(   # source input translated to target input
        H_input_source_real[i:i+1], 
        H_input_target_real[tgt_idx:tgt_idx+1],
        win_h=fda_win_h,
        win_w=fda_win_w
    )
    H_input_source_fda.append(x_fda[0])
    
    y_fda_ = apply_fda_translation(   # source label translated to target input
        H_true_source_real[tgt_idx:tgt_idx+1], 
        H_input_target_real[i:i+1],
        win_h=fda_win_h,
        win_w=fda_win_w
    )
    H_truePseudo.append(y_fda_[0])

H_input_source_fda = np.array(H_input_source_fda) 
H_truePseudo = np.array(H_truePseudo)
print(f"✓ FDA shape: {H_input_source_fda.shape}")

# ============================================================================
# STEP 2A: FDA FOR FDA unscaled inputs
# ============================================================================

print("\nCalculating PAD for FDA-translated unscaled inputs...")

h5_path_source_fda = f'{save_path}/features_source_fda.h5'
h5_path_target_fda = f'{save_path}/features_target_fda.h5'

# Plot histograms before calculating PAD
print("\nPlotting magnitude/phase histograms for UNSCALED inputs...")
plot_magnitude_phase_histograms(H_input_source_fda, '2a_unscaled_FDA', 'UNSCALED', 'Source')
plot_magnitude_phase_histograms(H_input_target_real, '2a_unscaled_FDA', 'UNSCALED', 'Target')


pca_src_fda, pca_tgt_fda, var_src_fda, var_tgt_fda = \
    apply_incremental_pca_and_save(
        H_input_source_fda, H_input_target_real,
        h5_path_source_fda, h5_path_target_fda,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="FDA_UNSCALED"
    )

pad_svm_fda, X_fda, y_fda = load_and_calculate_pad(
    h5_path_source_fda, h5_path_target_fda,
    pca_components_second=pca_components_second,
    stage_name="FDA_UNSCALED"
)

# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_fda):
    os.remove(h5_path_source_fda)
    print(f"✓ Deleted: {h5_path_source_fda}")
if os.path.exists(h5_path_target_fda):
    os.remove(h5_path_target_fda)
    print(f"✓ Deleted: {h5_path_target_fda}")

# ============================================================================
# STEP 2B: FDA FOR FDA scaled inputs
# ============================================================================
print("\nCalculating PAD for FDA-translated scaled inputs...")

h5_path_source_fda = f'{save_path}/features_source_fda.h5'
h5_path_target_fda = f'{save_path}/features_target_fda.h5'

H_input_source_scaled, x_min_src, x_max_src = minmaxScaler(
    H_input_source_fda, lower_range=lower_range, linear_interp=False
)

H_input_target_scaled, x_min_tgt, x_max_tgt = minmaxScaler(
    H_input_target_real, lower_range=lower_range, linear_interp=False
)

# Plot histograms before calculating PAD
print("\nPlotting magnitude/phase histograms for SCALED inputs...")
plot_magnitude_phase_histograms(H_input_source_scaled, '2b_scaled_FDA', 'SCALED', 'Source')
plot_magnitude_phase_histograms(H_input_target_scaled, '2b_scaled_FDA', 'SCALED', 'Target')

pca_src_fda, pca_tgt_fda, var_src_fda, var_tgt_fda = \
    apply_incremental_pca_and_save(
        H_input_source_scaled, H_input_target_scaled,
        h5_path_source_fda, h5_path_target_fda,
        batch_size=batch_size,
        pca_components_first=pca_components_first,
        incremental_batch_size=incremental_batch_size,
        max_fitting_batches=max_fitting_batches,
        data_name="FDA_SCALED"
    )

pad_svm_fda, X_fda, y_fda = load_and_calculate_pad(
    h5_path_source_fda, h5_path_target_fda,
    pca_components_second=pca_components_second,
    stage_name="FDA_SCALED"
)

# Delete H5 files after calculating PAD
if os.path.exists(h5_path_source_fda):
    os.remove(h5_path_source_fda)
    print(f"✓ Deleted: {h5_path_source_fda}")
if os.path.exists(h5_path_target_fda):
    os.remove(h5_path_target_fda)
    print(f"✓ Deleted: {h5_path_target_fda}")
    
    
# Plot label histograms (original, unscaled only)
print("\nPlotting magnitude/phase histograms for LABELS (original)...")
plot_magnitude_phase_histograms(H_true_source_real, '3a_labels_original', 'LABELS', 'Source')
plot_magnitude_phase_histograms(H_true_target_real, '3b_labels_original', 'LABELS', 'Target')

# Plot label histograms (translated source labels)
print("\nPlotting magnitude/phase histograms for TRANSLATED LABELS...")
plot_magnitude_phase_histograms(H_truePseudo, '3c_labels_translated', 'LABELS', 'SourceFDA')
