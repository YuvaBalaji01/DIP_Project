import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_lowpass_filter(image_path, D0):
    """
    Performs image smoothing using a Gaussian Lowpass Filter (GLPF) in the frequency domain.

    Args:
        image_path (str): Path to the input image file.
        D0 (float): The cutoff frequency (standard deviation) of the Gaussian filter.
    """
    
    # --- 1. Load and Preprocess Image ---
    
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    M, N = img.shape
    
    # Convert image to float32 for FFT calculation
    f = img.astype(np.float32)

    # Centering (Shifting): Multiply by (-1)^(x+y)
    # This moves the DC component to the center for proper filtering
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f_shifted = f * ((-1)**(X + Y))

    # --- 2. Compute the 2-D FFT ---
    
    # Compute the 2D FFT
    F = np.fft.fft2(f_shifted)

    # --- 3. Create the Gaussian Lowpass Filter (GLPF) ---
    
    # Create the distance matrix D(u, v)
    u = np.arange(M)
    v = np.arange(N)
    
    # Center points (M/2 and N/2)
    u_center = M / 2
    v_center = N / 2
    
    # Calculate distance D(u, v) from the center
    U, V = np.meshgrid(u - u_center, v - v_center, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    
    # Calculate the Gaussian Lowpass Filter H(u, v)
    # H(u, v) = exp(-D^2 / (2 * D0^2))
    H = np.exp(-(D**2) / (2 * (D0**2)))

    # --- 4. Filtering ---
    
    # Apply the filter: G(u, v) = F(u, v) * H(u, v)
    G = F * H

    # --- 5. Inverse FFT and Post-Processing ---
    
    # Compute the 2D Inverse FFT
    g_shifted = np.fft.ifft2(G)
    
    # Inverse Centering (Shifting back)
    # The real part is the resulting image, imaginary part is numerical error
    g = np.real(g_shifted) * ((-1)**(X + Y))
    
    # Final scaling and type conversion for display (0 to 255)
    g_output = np.clip(g, 0, 255).astype(np.uint8)

    # --- 6. Visualization ---
    
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Filtered Image
    plt.subplot(1, 2, 2)
    plt.imshow(g_output, cmap='gray')
    plt.title(f'GLPF Filtered (D0={D0})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Execution Example ---
# IMPORTANT: Replace 'path/to/your/image.jpg' with a real image file path.
# Test with different D0 values (e.g., 10, 30, 80) to see varying levels of blur.
image_file = 'Lenna_(test_image).png' # Use a standard test image like 'lena.jpg' or a photo of your own
cutoff_frequency = 30 # A small D0 value for significant smoothing

# Run the function
gaussian_lowpass_filter(image_file, cutoff_frequency)