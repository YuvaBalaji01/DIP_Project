import numpy as np
import cv2
import matplotlib.pyplot as plt

def high_frequency_emphasis_filter(image_path, D0, A, B, n=2):
    """
    Performs image sharpening using a High-Frequency Emphasis (HFE) filter
    based on the Butterworth Highpass Filter (BHPF).

    Args:
        image_path (str): Path to the input image file (e.g., 'lena.png').
        D0 (float): The cutoff frequency for the filter.
        A (float): Low-frequency gain (preserves overall brightness/structure).
        B (float): High-frequency gain (controls sharpening strength).
        n (int): Butterworth filter order (default is 2).
    """
    
    # --- 1. Load and Preprocess Image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    M, N = img.shape
    f = img.astype(np.float32)

    # Centering (Shifting): Multiply by (-1)^(x+y)
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f_shifted = f * ((-1)**(X + Y))

    # --- 2. Compute the 2-D FFT ---
    F = np.fft.fft2(f_shifted)

    # --- 3. Create the BHPF and HFE Filter Mask ---
    
    # Create the distance matrix D(u, v)
    u = np.arange(M)
    v = np.arange(N)
    u_center, v_center = M / 2, N / 2
    
    U, V = np.meshgrid(u - u_center, v - v_center, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    
    # Calculate the Butterworth Lowpass Filter (BLPF)
    # H_LP = 1 / (1 + (D / D0)**(2 * n))
    H_LP = 1 / (1 + (D / D0)**(2 * n))

    # Invert to get the Highpass Filter (BHPF)
    H_HP = 1 - H_LP
    
    # Implement High-Frequency Emphasis (HFE)
    # H_HFE = A + B * H_HP
    H_HFE = A + B * H_HP

    # --- 4. Filtering ---
    
    # Apply the HFE filter: G(u, v) = F(u, v) * H_HFE(u, v)
    G = F * H_HFE

    # --- 5. Inverse FFT and Post-Processing ---
    
    # Compute the 2D Inverse FFT
    g_shifted = np.fft.ifft2(G)
    
    # Inverse Centering and taking the real part
    g = np.real(g_shifted) * ((-1)**(X + Y))
    
    # Final scaling and type conversion for display (0 to 255)
    g_output = np.clip(g, 0, 255).astype(np.uint8)

    # --- 6. Visualization ---
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Filter Mask
    plt.subplot(1, 3, 1)
    plt.imshow(H_HFE, cmap='gray')
    plt.title(f'HFE Filter Mask (A={A}, B={B})')
    plt.axis('off')
    
    # Plot 2: Original Image
    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot 3: Filtered Image
    plt.subplot(1, 3, 3)
    plt.imshow(g_output, cmap='gray')
    plt.title(f'HFE Filtered (D0={D0})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Execution Example ---
# IMPORTANT: Replace 'lena_test_image.png' with your actual image file path.
image_file = '../Lenna_(test_image).png' 

# Tuning Parameters:
# D0: Controls the width of the boost region (small D0 means a wide boost)
cutoff_frequency = 40 

# A: Low-frequency gain (Keep close to 1.0 to preserve brightness)
A_gain = 0.1 

# B: High-frequency gain (Higher B means stronger sharpening)
B_gain = 2.5 

# Run the function
high_frequency_emphasis_filter(image_file, cutoff_frequency, A_gain, B_gain)