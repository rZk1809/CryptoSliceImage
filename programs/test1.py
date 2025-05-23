
import numpy as np
from PIL import Image
import os
import random
import hashlib
import math
import matplotlib.pyplot as plt

# --- Configuration ---
PRIME = 257  # Use a prime > 255 for 8-bit grayscale
OUTPUT_DIR = "image_shares"
ORIGINAL_HASH_FILE = os.path.join(OUTPUT_DIR, "original_hash.txt")
SHARE_HASH_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, "share_{}_hash.txt")

# --- Shamir's Secret Sharing Core Functions (Modulo PRIME) ---

def extended_gcd(a, b):
    """Computes extended Euclidean algorithm, returns (gcd, x, y) such that ax + by = gcd."""
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (gcd, x, y)

def modInverse(a, m):
    """Computes modular multiplicative inverse of a modulo m."""
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return (x % m + m) % m

def generate_polynomial(degree, secret):
    """Generates a random polynomial P(x) of degree 'degree' such that P(0) = secret."""
    if secret >= PRIME:
         # Handle potential case where input pixel value might somehow exceed prime
         # This shouldn't happen with PRIME=257 and 8-bit grayscale, but good practice.
         print(f"Warning: Secret {secret} >= PRIME {PRIME}. Using modulo.")
         secret = secret % PRIME
    coefficients = [secret] + [random.randint(0, PRIME - 1) for _ in range(degree)]
    # Ensure coefficients are within the field
    return [c % PRIME for c in coefficients]

def evaluate_polynomial(coeffs, x):
    """Evaluates the polynomial P(x) = coeffs[0] + coeffs[1]*x + ... modulo PRIME."""
    result = 0
    power_of_x = 1
    for coeff in coeffs:
        term = (coeff * power_of_x) % PRIME
        result = (result + term) % PRIME
        power_of_x = (power_of_x * x) % PRIME
    return result

def lagrange_interpolate(points, x_target=0):
    """
    Finds the y-value corresponding to x_target (usually 0 for the secret)
    using Lagrange interpolation.
    points is a list of (x, y) pairs.
    k is the number of points (threshold).
    """
    if not points:
        raise ValueError("Need at least one point for interpolation")

    k = len(points)
    xs, ys = zip(*points)

    if len(set(xs)) != k:
        raise ValueError("X values must be distinct for Lagrange interpolation")

    secret = 0
    for i in range(k):
        numerator, denominator = 1, 1
        for j in range(k):
            if i == j:
                continue
            # Calculate L_i(x_target)
            numerator = (numerator * (x_target - xs[j])) % PRIME
            denominator = (denominator * (xs[i] - xs[j])) % PRIME

        # Ensure numerator and denominator are positive before inverse
        numerator = (numerator + PRIME) % PRIME
        denominator = (denominator + PRIME) % PRIME

        term = (ys[i] * numerator * modInverse(denominator, PRIME)) % PRIME
        secret = (secret + term) % PRIME

    return secret

# --- Image Processing and Sharing Functions ---

def preprocess_image(image_path):
    """Loads an image, converts to grayscale, and returns as numpy array."""
    try:
        img = Image.open(image_path).convert('L') # 'L' mode is 8-bit grayscale
        return np.array(img, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def create_shares(image_data, k, n):
    """
    Splits the image_data into n shares using SSS (k-out-of-n).
    Returns a list of n numpy arrays representing the shares.
    """
    height, width = image_data.shape
    # Initialize shares as numpy arrays of appropriate type to hold values up to PRIME-1
    # Using float64 temporarily during calculation before clipping if needed,
    # or int type large enough (like uint16 or int32 if PRIME were much larger)
    shares_data = [np.zeros_like(image_data, dtype=np.int32) for _ in range(n)]

    print(f"Creating {n} shares (k={k}) for image of size {width}x{height}...")
    # Iterate through each pixel
    for r in range(height):
        for c in range(width):
            secret_pixel_value = int(image_data[r, c]) # Ensure it's a Python int
            if secret_pixel_value >= PRIME:
                # Should not happen with PRIME=257 and uint8 images
                print(f"Warning: Pixel value {secret_pixel_value} at ({r},{c}) >= PRIME {PRIME}. Clamping.")
                secret_pixel_value = PRIME - 1

            # Generate a polynomial of degree k-1 for this pixel
            coeffs = generate_polynomial(k - 1, secret_pixel_value)

            # Generate n share values for this pixel by evaluating P(x) at x=1, 2, ..., n
            for i in range(n):
                share_x = i + 1 # Use x = 1, 2, ..., n
                share_value = evaluate_polynomial(coeffs, share_x)
                shares_data[i][r, c] = share_value
        # Optional: Print progress
        if (r + 1) % (height // 10) == 0 or r == height - 1:
             print(f"  Processed row {r+1}/{height}")

    print("Share creation complete.")
    # Convert share data back to uint8 for saving/viewing.
    # Values are 0 to PRIME-1 (0-256). We can map these to 0-255 for display.
    # A simple modulo or scaling might distort visual randomness slightly,
    # but direct clipping is common for visualization.
    shares_uint8 = [(s % 256).astype(np.uint8) for s in shares_data]
    # Store the original integer shares too, as they are needed for reconstruction
    return shares_uint8, shares_data # Return visual shares and numerical shares


def reconstruct_image(shares_to_reconstruct, share_indices, k):
    """
    Reconstructs the image from k shares using Lagrange interpolation.
    shares_to_reconstruct: List of k numpy arrays (numerical share data).
    share_indices: List of the original indices (1 to n) corresponding to the shares.
    k: The threshold.
    """
    if len(shares_to_reconstruct) < k or len(share_indices) < k:
        raise ValueError(f"Need at least k={k} shares and indices to reconstruct.")
    if len(shares_to_reconstruct) != len(share_indices):
        raise ValueError("Number of shares and indices must match.")

    # Use the first share provided to get dimensions
    height, width = shares_to_reconstruct[0].shape
    reconstructed_data = np.zeros((height, width), dtype=np.int32) # Use int32 to handle intermediate sums

    print(f"Reconstructing image from {len(shares_to_reconstruct)} shares (k={k})...")
    # Iterate through each pixel position
    for r in range(height):
        for c in range(width):
            # Gather the points (x, y) for this pixel from the k shares
            points = []
            for i in range(len(shares_to_reconstruct)):
                share_index = share_indices[i] # This is the 'x' value (1, 2, ..., n)
                share_value = int(shares_to_reconstruct[i][r, c]) # The 'y' value (numerical share)
                points.append((share_index, share_value))

            # Use Lagrange interpolation to find the secret (original pixel value P(0))
            try:
                reconstructed_pixel_value = lagrange_interpolate(points, x_target=0)
                reconstructed_data[r, c] = reconstructed_pixel_value
            except Exception as e:
                print(f"Error during interpolation at ({r},{c}): {e}")
                # Handle error, e.g., set pixel to black or raise exception
                reconstructed_data[r, c] = 0 # Example: Set to black on error
                # Optionally re-raise if you want the whole process to stop
                # raise e

        # Optional: Print progress
        if (r + 1) % (height // 10) == 0 or r == height - 1:
             print(f"  Reconstructed row {r+1}/{height}")

    print("Image reconstruction complete.")
    # Clip values to 0-255 and convert to uint8 for the final image
    reconstructed_image = np.clip(reconstructed_data, 0, 255).astype(np.uint8)
    return reconstructed_image

# --- Hashing for Integrity ---

def calculate_hash(data):
    """Calculates SHA-256 hash of numpy array data."""
    return hashlib.sha256(data.tobytes()).hexdigest()

# --- Saving and Loading ---

def save_shares_and_hashes(shares_visual, shares_numerical, base_filename):
    """Saves visual shares as images and numerical shares (e.g., as .npy)."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    share_hashes = []
    for i, (visual_share, numerical_share) in enumerate(zip(shares_visual, shares_numerical)):
        # Save visual share (noise-like image)
        share_filename_png = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{i+1}.png")
        try:
            Image.fromarray(visual_share).save(share_filename_png)
            print(f"Saved visual share {i+1} to {share_filename_png}")
        except Exception as e:
            print(f"Error saving visual share {i+1}: {e}")

        # Save numerical share (needed for reconstruction)
        share_filename_npy = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{i+1}_numerical.npy")
        try:
            np.save(share_filename_npy, numerical_share)
            print(f"Saved numerical share {i+1} data to {share_filename_npy}")
            # Calculate and store hash of the numerical share
            share_hash = calculate_hash(numerical_share)
            share_hashes.append(share_hash)
            hash_file = SHARE_HASH_FILE_TEMPLATE.format(i+1)
            with open(hash_file, 'w') as f:
                f.write(share_hash)
            print(f"Saved hash for share {i+1} to {hash_file}")

        except Exception as e:
            print(f"Error saving numerical share {i+1} data or hash: {e}")
    return share_hashes


def load_numerical_shares(base_filename, indices_to_load):
    """Loads specified numerical shares and their hashes from files."""
    loaded_shares = []
    loaded_indices = []
    loaded_hashes = []
    valid_shares = True
    for i in indices_to_load:
        share_filename_npy = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{i}_numerical.npy")
        hash_file = SHARE_HASH_FILE_TEMPLATE.format(i)
        try:
            share_data = np.load(share_filename_npy)
            with open(hash_file, 'r') as f:
                expected_hash = f.read().strip()

            # Verify integrity before using
            actual_hash = calculate_hash(share_data)
            if actual_hash == expected_hash:
                loaded_shares.append(share_data)
                loaded_indices.append(i) # Store the original index (1 to n)
                loaded_hashes.append(actual_hash)
                print(f"Loaded and verified numerical share {i} from {share_filename_npy}")
            else:
                print(f"Error: Hash mismatch for share {i}! Expected {expected_hash}, got {actual_hash}. Share may be corrupt.")
                valid_shares = False
                # Decide how to handle corrupt shares (e.g., skip, raise error)
                # For this example, we'll note it but might try reconstruction anyway if enough valid shares remain
        except FileNotFoundError:
            print(f"Error: Share file or hash file for index {i} not found.")
            valid_shares = False
        except Exception as e:
            print(f"Error loading share {i}: {e}")
            valid_shares = False

    return loaded_shares, loaded_indices, loaded_hashes, valid_shares

# --- Display ---

def display_images(original, shares_to_display, reconstructed, k, n):
    """Displays original, some shares, and reconstructed images."""
    num_shares_display = min(len(shares_to_display), 3) # Display up to 3 shares
    num_cols = 2 + num_shares_display
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))

    # Display Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display Some Shares (Visual representations)
    for i in range(num_shares_display):
        # Load the *visual* share for display purposes
        try:
            share_index = shares_to_display[i][0] # Get the index (1-based)
            visual_share_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_share_{share_index}.png")
            visual_share_img = Image.open(visual_share_path)
            axes[i+1].imshow(visual_share_img, cmap='gray')
            axes[i+1].set_title(f"Share {share_index} (Visual)")
            axes[i+1].axis('off')
        except Exception as e:
             axes[i+1].set_title(f"Share {share_index} (Error Load)")
             print(f"Could not load visual share {share_index} for display: {e}")


    # Display Reconstructed
    if reconstructed is not None:
        axes[num_cols - 1].imshow(reconstructed, cmap='gray')
        axes[num_cols - 1].set_title(f"Reconstructed (from {k}/{n} shares)")
        axes[num_cols - 1].axis('off')
    else:
        axes[num_cols - 1].set_title("Reconstruction Failed")
        axes[num_cols - 1].axis('off')


    plt.tight_layout()
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # --- Parameters ---
    image_path = input("Enter the path to the image file: ") # e.g., 'lena_gray.png' or 'test_image.jpg'
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist. Please provide a valid path.")
        exit()

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    while True:
        try:
            n = int(input("Enter the total number of shares to create (n): "))
            if n > 1:
                break
            else: print("n must be greater than 1.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        try:
            k = int(input(f"Enter the threshold number of shares needed to reconstruct (k <= {n}): "))
            if 1 < k <= n:
                break
            else: print(f"k must be between 2 and {n}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


    # 1. Image Preprocessing
    print("-" * 30)
    print("1. Preprocessing Image...")
    original_image_data = preprocess_image(image_path)

    if original_image_data is not None:
        print(f"Image loaded successfully ({original_image_data.shape[1]}x{original_image_data.shape[0]}).")
        # Calculate and save original hash
        original_hash = calculate_hash(original_image_data)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        with open(ORIGINAL_HASH_FILE, 'w') as f:
            f.write(original_hash)
        print(f"Original image hash ({original_hash[:8]}...) saved to {ORIGINAL_HASH_FILE}")

        # 2. Create Shares (SSS on Pixel Intensity)
        print("-" * 30)
        print("2. Creating Shares...")
        # shares_visual are uint8 for display, shares_numerical are int32/higher for math
        shares_visual, shares_numerical = create_shares(original_image_data, k, n)

        # 3. Secure Storage (Simulated by saving locally with hashes)
        print("-" * 30)
        print("3. Saving Shares and Hashes...")
        # This also calculates and saves hashes for each numerical share
        save_shares_and_hashes(shares_visual, shares_numerical, base_filename)
        print(f"Shares and hashes saved in directory: {OUTPUT_DIR}")
        # Note: Real secure transmission would involve encryption and distributing
        # shares_numerical (or encrypted versions) to different parties/servers.
        # The Max Flow part is a theoretical concept for optimal distribution paths,
        # not typically part of the core SSS/VC algorithm itself.

        # 4. Reconstruction Simulation
        print("-" * 30)
        print("4. Simulating Reconstruction...")
        # Example: Choose k shares randomly (or specify indices)
        # indices_to_reconstruct = random.sample(range(1, n + 1), k)
        indices_str = input(f"Enter {k} share indices (e.g., '1 3 5') to use for reconstruction: ")
        try:
            indices_to_reconstruct = [int(idx) for idx in indices_str.split()]
            if len(indices_to_reconstruct) != k:
                 raise ValueError(f"Please provide exactly {k} indices.")
            if not all(1 <= idx <= n for idx in indices_to_reconstruct):
                 raise ValueError(f"Indices must be between 1 and {n}.")
            if len(set(indices_to_reconstruct)) != k:
                 raise ValueError("Indices must be unique.")

            print(f"Attempting reconstruction using shares: {indices_to_reconstruct}")

            # Load the required *numerical* shares and verify hashes
            shares_for_recon, indices_loaded, hashes_loaded, shares_valid = load_numerical_shares(base_filename, indices_to_reconstruct)

            reconstructed_image = None
            if shares_valid and len(shares_for_recon) == k:
                # Perform reconstruction using Lagrange interpolation
                reconstructed_image = reconstruct_image(shares_for_recon, indices_loaded, k)

                # 5. Integrity Verification
                print("-" * 30)
                print("5. Verifying Integrity...")
                reconstructed_hash = calculate_hash(reconstructed_image)
                print(f"Reconstructed image hash: {reconstructed_hash[:8]}...")
                print(f"Original image hash:      {original_hash[:8]}...")
                if reconstructed_hash == original_hash:
                    print("Integrity Check PASSED: Reconstructed image matches the original.")
                else:
                    print("Integrity Check FAILED: Reconstructed image does not match the original! (Possible corruption or error)")
            else:
                 print("Reconstruction aborted due to missing or invalid shares.")


            # 6. Display Results
            print("-" * 30)
            print("6. Displaying Results...")
            # Prepare list of (index, visual_share_data) for display
            # We load the visual shares here just for the display function
            shares_to_show_for_display = []
            for idx in indices_to_reconstruct:
                 # We don't actually need the visual data inside the display func now,
                 # it loads from file based on index. Just pass the indices used.
                 shares_to_show_for_display.append((idx, None)) # Pass index, placeholder for data

            display_images(original_image_data, shares_to_show_for_display, reconstructed_image, k, n)

        except ValueError as e:
             print(f"Invalid input for indices: {e}")
        except Exception as e:
             print(f"An error occurred during reconstruction or display: {e}")

    else:
        print("Could not load the original image. Exiting.")
