
import numpy as np
from PIL import Image, ImageOps
import os
import random
import matplotlib.pyplot as plt

# --- Configuration ---
OUTPUT_DIR_VC = "vc_image_shares"
# Define the subpixel patterns (0=Black, 255=White)
# We need pairs of 2x2 blocks.
# For a WHITE pixel: both shares get the SAME block (chosen randomly from a pair)
# For a BLACK pixel: shares get COMPLEMENTARY blocks (one from each pair)
# Pattern representation: 0 = Black, 1 = White (easier for logic)
# Will convert to 0/255 later for image saving.

# Example Patterns (Each block has 2 black, 2 white subpixels)
PATTERNS = [
    (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]])), # Pair 1
    (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [1, 1]])), # Pair 2 (Less standard, but works)
    # Add more pairs for more randomness if desired, choose one pair randomly per pixel
]
# Using the most standard pair:
STANDARD_PATTERNS = (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]))


# --- Visual Cryptography Functions ---

def preprocess_image_for_vc(image_path, threshold=128):
    """Loads an image, converts to grayscale, then to binary (B&W), returns numpy array."""
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        # Convert to binary: pixels >= threshold are white (255), others black (0)
        img_binary = img.point(lambda p: 255 if p >= threshold else 0, mode='1') # mode '1' is binary
        # Convert PIL binary image (which might be boolean) to numpy array uint8 (0 and 255)
        img_array = np.array(img_binary.convert('L'), dtype=np.uint8) # Convert back to L to get 0/255
        # Optional: Invert if your logic assumes 0=White, 1=Black (our PATTERNS use 0=Black)
        # If needed: img_array = 255 - img_array
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image for VC: {e}")
        return None

def create_vc_shares(binary_image_data):
    """Creates two VC shares from a binary image numpy array (0=Black, 255=White)."""
    if binary_image_data is None:
        return None, None
    height, width = binary_image_data.shape
    # Shares will be twice the size in each dimension
    share1 = np.zeros((height * 2, width * 2), dtype=np.uint8)
    share2 = np.zeros((height * 2, width * 2), dtype=np.uint8)

    print(f"Creating 2 VC shares for binary image of size {width}x{height}...")

    # Get the standard patterns (0=Black, 1=White internally for now)
    pattern1, pattern2 = STANDARD_PATTERNS

    for r in range(height):
        for c in range(width):
            pixel_value = binary_image_data[r, c]
            # Randomly choose which pattern pair orientation to use
            # (e.g., [[0,1],[1,0]] or [[1,0],[0,1]])
            use_pattern1, use_pattern2 = random.choice([(pattern1, pattern2), (pattern2, pattern1)])

            r_start, c_start = r * 2, c * 2

            if pixel_value == 255: # Original pixel is WHITE
                # Assign the *same* pattern to both shares
                chosen_pattern = random.choice([use_pattern1, use_pattern2])
                # Convert 0/1 pattern to 0/255 for image
                block = (chosen_pattern * 255).astype(np.uint8)
                share1[r_start:r_start+2, c_start:c_start+2] = block
                share2[r_start:r_start+2, c_start:c_start+2] = block
            else: # Original pixel is BLACK (0)
                # Assign *complementary* patterns to the shares
                block1 = (use_pattern1 * 255).astype(np.uint8)
                block2 = (use_pattern2 * 255).astype(np.uint8)
                share1[r_start:r_start+2, c_start:c_start+2] = block1
                share2[r_start:r_start+2, c_start:c_start+2] = block2

        # Optional: Print progress
        if (r + 1) % (height // 10) == 0 or r == height - 1:
             print(f"  Processed row {r+1}/{height}")

    print("VC Share creation complete.")
    return share1, share2

def reconstruct_vc_image(share1_data, share2_data):
    """Reconstructs the image by overlaying (stacking) the two shares."""
    if share1_data is None or share2_data is None:
        return None
    if share1_data.shape != share2_data.shape:
        raise ValueError("Shares must have the same dimensions for reconstruction.")

    print("Reconstructing by overlaying shares...")
    # Overlaying: A pixel is white (255) ONLY if BOTH shares are white at that position.
    # Otherwise, it's black (0). This is equivalent to a MIN operation.
    # Or logical AND if shares were 0/1 boolean.
    reconstructed_data = np.minimum(share1_data, share2_data)
    print("Reconstruction complete.")
    return reconstructed_data

# --- Saving ---

def save_vc_shares(share1, share2, base_filename):
    """Saves the two VC shares as PNG images."""
    if not os.path.exists(OUTPUT_DIR_VC):
        os.makedirs(OUTPUT_DIR_VC)
        print(f"Created directory: {OUTPUT_DIR_VC}")

    try:
        path1 = os.path.join(OUTPUT_DIR_VC, f"{base_filename}_vc_share1.png")
        path2 = os.path.join(OUTPUT_DIR_VC, f"{base_filename}_vc_share2.png")
        Image.fromarray(share1).save(path1)
        Image.fromarray(share2).save(path2)
        print(f"Saved VC Share 1 to {path1}")
        print(f"Saved VC Share 2 to {path2}")
        return path1, path2
    except Exception as e:
        print(f"Error saving VC shares: {e}")
        return None, None

# --- Display ---

def display_vc_results(original_binary, share1, share2, reconstructed):
    """Displays the original binary, the two shares, and the reconstructed image."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    titles = ["Original (Binary)", "Share 1", "Share 2", "Reconstructed (Overlay)"]
    images = [original_binary, share1, share2, reconstructed]

    for i, ax in enumerate(axes):
        if images[i] is not None:
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=255)
            ax.set_title(titles[i])
        else:
            ax.set_title(f"{titles[i]} (Not Available)")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # --- Parameters ---
    image_path = input("Enter the path to the image file: ") # e.g., 'lena_gray.png' or 'test_pattern.png'
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist. Please provide a valid path.")
        exit()

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Image Preprocessing for VC (Load -> Grayscale -> Binary)
    print("-" * 30)
    print("1. Preprocessing Image for Visual Cryptography...")
    # You can adjust the threshold (0-255) for binarization
    binary_image_data = preprocess_image_for_vc(image_path, threshold=128)

    if binary_image_data is not None:
        print(f"Image loaded and binarized ({binary_image_data.shape[1]}x{binary_image_data.shape[0]}).")

        # 2. Create VC Shares
        print("-" * 30)
        print("2. Creating Visual Cryptography Shares...")
        share1_data, share2_data = create_vc_shares(binary_image_data)

        if share1_data is not None and share2_data is not None:
            # 3. Save Shares
            print("-" * 30)
            print("3. Saving VC Shares...")
            save_vc_shares(share1_data, share2_data, base_filename)
            print(f"VC Shares saved in directory: {OUTPUT_DIR_VC}")

            # 4. Reconstruction (Simulated Overlay)
            print("-" * 30)
            print("4. Simulating Reconstruction by Overlay...")
            # In this simple 2-out-of-2 scheme, we just use the two shares we created.
            reconstructed_image = reconstruct_vc_image(share1_data, share2_data)

            # 5. Display Results
            print("-" * 30)
            print("5. Displaying Results...")
            display_vc_results(binary_image_data, share1_data, share2_data, reconstructed_image)
        else:
            print("Failed to create VC shares.")

    else:
        print("Could not load and preprocess the image. Exiting.")
