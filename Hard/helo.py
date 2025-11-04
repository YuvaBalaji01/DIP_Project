import numpy as np
import cv2
import matplotlib.pyplot as plt

def adaptive_enhancement_pipeline(image_path, D0_smooth, D0_sharp, A_gain, B_gain, noise_std):
    """
    Implements a full frequency domain image enhancement pipeline:
    1. Adds Gaussian Noise.
    2. Smoothes the noise using a Gaussian Lowpass Filter (GLPF).
    3. Restores detail using a Butterworth High-Frequency Emphasis (HFE) filter.

    Args:
        image_path (str): Path to the input image file.
        D0_smooth (float): Cutoff freq. for GLPF (Noise Reduction).
        D0_sharp (float): Cutoff freq. for BHPF (Detail Restoration).
        A_gain (float): Low-frequency gain for HFE.
        B_gain (float): High-frequency gain for HFE.
        noise_std (float): Standard deviation of the synthetic Gaussian noise.
    """
    
    # --- 1. Load Image and Generate Noise ---
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    M, N = img.shape
    f_orig = img.astype(np.float32)

    # Add Synthetic Gaussian Noise
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    f_noisy = np.clip(f_orig + noise, 0, 255)
    f = f_noisy # Use the noisy image for processing

    # Centering (Shifting) factor and meshgrids
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    shift_factor = ((-1)**(X + Y))
    
    # Calculate Distance Matrix D(u, v) (used by both filters)
    u = np.arange(M)
    v = np.arange(N)
    u_center, v_center = M / 2, N / 2
    U, V = np.meshgrid(u - u_center, v - v_center, indexing='ij')
    D = np.sqrt(U**2 + V**2)

    # --- 2. FFT: Shift and Transform Noisy Image ---
    
    f_shifted = f * shift_factor
    F = np.fft.fft2(f_shifted)
    
    # --- 3. Stage 1: Noise Reduction (GLPF) ---
    
    # Gaussian Lowpass Filter H_LP(u, v)
    H_GLPF = np.exp(-(D**2) / (2 * (D0_smooth**2)))

    # Apply GLPF to the spectrum F
    G_smooth = F * H_GLPF
    
    # --- 4. Stage 2: Detail Restoration (HFE) ---

    # Calculate the Butterworth Lowpass Filter (BLPF) for inversion
    n = 2 # Fixed Butterworth order
    H_BLPF = 1 / (1 + (D / D0_sharp)**(2 * n))

    # Invert to get the Highpass Filter (BHPF)
    H_BHPF = 1 - H_BLPF
    
    # Implement High-Frequency Emphasis (HFE)
    H_HFE = A_gain + B_gain * H_BHPF

    # Apply HFE to the smoothed spectrum G_smooth
    G_enhanced = G_smooth * H_HFE

    # --- 5. Inverse FFT and Post-Processing ---
    
    # Inverse Transform G_enhanced
    g_shifted = np.fft.ifft2(G_enhanced)
    g_final = np.real(g_shifted) * shift_factor
    
    g_output = np.clip(g_final, 0, 255).astype(np.uint8)

    # Get the intermediate smoothed image for visualization
    g_smooth_spatial = np.fft.ifft2(G_smooth)
    g_smooth_spatial = np.real(g_smooth_spatial) * shift_factor
    g_smooth_output = np.clip(g_smooth_spatial, 0, 255).astype(np.uint8)


    # --- 6. Visualization (4-Image Comparison) ---
    
    plt.figure(figsize=(12, 12))
    
    # 1. Original Clean Image
    plt.subplot(2, 2, 1)
    plt.imshow(f_orig, cmap='gray')
    plt.title('1. Original Clean Image')
    plt.axis('off')
    
    # 2. Noisy Image (Input to Pipeline)
    plt.subplot(2, 2, 2)
    plt.imshow(f_noisy.astype(np.uint8), cmap='gray')
    plt.title(f'2. Input Noisy Image (Std Dev={noise_std})')
    plt.axis('off')

    # 3. Smoothed Image (Noise Reduced)
    plt.subplot(2, 2, 3)
    plt.imshow(g_smooth_output, cmap='gray')
    plt.title(f'3. Smoothed (GLPF D0={D0_smooth})')
    plt.axis('off')
    
    # 4. Final Enhanced Image
    plt.subplot(2, 2, 4)
    plt.imshow(g_output, cmap='gray')
    plt.title(f'4. Final Enhanced (HFE A={A_gain}, B={B_gain})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Execution Example ---
image_file = '../Lenna_(test_image).png' # Adjust path as needed

# --- Enhancement Parameters ---
# D0_smooth: Needs to be small (e.g., 10-30) to aggressively remove high-freq noise.
D0_smooth_value = 25 
# D0_sharp: Can be larger (e.g., 40-60) to target a broader range of mid/high frequencies for sharpening.
D0_sharp_value = 50 

# HFE Parameters:
A_value = 0.9  # Keep low frequencies/brightness intact
B_value = 1.8  # Aggressively boost edges

# Noise Generation Parameter:
noise_standard_deviation = 20 # Moderate noise level

# Run the Hard Project Pipeline
adaptive_enhancement_pipeline(
    image_file, 
    D0_smooth_value, 
    D0_sharp_value, 
    A_value, 
    B_value, 
    noise_standard_deviation
)