import numpy as np
from PIL import Image
import os
import random
import hashlib
import math
import matplotlib.pyplot as plt
import itertools # For combinations in reconstruction test

# --- Configuration ---
# PRIME = 251 # Use the prime specified in the paper (lossy for pixels 251-255)
# PRIME = 257 # Use a prime > 255 for lossless 8-bit grayscale
PRIME = 251 # <<< Using the paper's prime
print(f"INFO: Using PRIME = {PRIME}. Pixel values > {PRIME-1} will be reduced modulo {PRIME}.")

OUTPUT_DIR = "image_shares_extended"
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
    a = a % m # Ensure a is within mod m
    if a < 0:
       a += m
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        # print(f"DEBUG: No inverse for {a} mod {m}")
        raise ValueError(f'Modular inverse does not exist for {a} mod {m}')
    else:
        return (x % m + m) % m

def generate_polynomial(degree, secret):
    """Generates a random polynomial P(x) of degree 'degree' such that P(0) = secret."""
    # Ensure secret is within the prime field
    secret = int(secret) % PRIME
    coefficients = [secret] + [random.randint(0, PRIME - 1) for _ in range(degree)]
    return [c % PRIME for c in coefficients]

def evaluate_polynomial(coeffs, x):
    """Evaluates the polynomial P(x) = coeffs[0] + coeffs[1]*x + ... modulo PRIME."""
    result = 0
    power_of_x = 1
    x_mod = x % PRIME # Ensure x is within the field if it's large
    for coeff in coeffs:
        term = (coeff * power_of_x) % PRIME
        result = (result + term) % PRIME
        power_of_x = (power_of_x * x_mod) % PRIME
    return result

def lagrange_interpolate(points, x_target=0):
    """
    Finds the y-value corresponding to x_target (usually 0 for the secret)
    using Lagrange interpolation.
    points is a list of (x, y) pairs. Assumes x values are distinct.
    """
    if not points:
        raise ValueError("Need at least one point for interpolation")

    k = len(points)
    xs, ys = zip(*points)

    # Double-check distinctness (should be handled by caller)
    if len(set(xs)) != k:
        # This indicates a logic error elsewhere if it happens
        raise ValueError("X values must be distinct for Lagrange interpolation")

    secret = 0
    x_target_mod = x_target % PRIME

    for i in range(k):
        numerator, denominator = 1, 1
        xi = xs[i] % PRIME # Ensure x is within the field

        for j in range(k):
            if i == j:
                continue
            xj = xs[j] % PRIME # Ensure x is within the field

            numerator = (numerator * (x_target_mod - xj)) % PRIME
            denominator = (denominator * (xi - xj)) % PRIME

        # Compute term = ys[i] * L_i(x_target) mod PRIME
        # L_i(x_target) = numerator * modInverse(denominator, PRIME)
        try:
             inv_denom = modInverse(denominator, PRIME)
             term = (ys[i] * numerator * inv_denom) % PRIME
             secret = (secret + term) % PRIME
        except ValueError as e:
            # This can happen if duplicate x values lead to xi - xj = 0
             print(f"ERROR in Lagrange: {e}. Points: {points}, i={i}, denom={denominator}")
             raise e # Re-raise

    # Ensure result is positive
    return (secret + PRIME) % PRIME


# --- Linear Algebra for Reconstruction with N+1 Share (Modulo PRIME) ---

def precompute_power_sums(n_orig, k_degree, p):
    """Precompute sums of powers: sum(i^j for i=1..n_orig) mod p for j=0..k_degree."""
    power_sums = {} # Store as {degree: sum}
    current_powers = [1] * n_orig # Start with i^0 = 1
    for j in range(k_degree + 1): # For powers 0 to k_degree (polynomial degree k-1 needs powers up to k-1)
        current_sum = sum(current_powers) % p
        power_sums[j] = current_sum
        # Update powers for next iteration: current_powers[i-1] = i^(j+1)
        if j < k_degree:
            for i in range(1, n_orig + 1):
                current_powers[i-1] = (current_powers[i-1] * i) % p
    return power_sums

def solve_linear_system_mod_p(matrix_A, vector_b, p):
    """
    Solves the linear system Ax = b modulo p for x.
    Returns the solution vector x. Uses numpy for basic operations.
    Assumes matrix_A is square and invertible modulo p.
    """
    n = matrix_A.shape[0]
    if matrix_A.shape != (n, n) or vector_b.shape != (n,):
        raise ValueError("Matrix and vector dimensions mismatch.")

    # Calculate determinant modulo p
    det_A = int(round(np.linalg.det(matrix_A))) % p # np.linalg.det returns float

    if det_A == 0:
        raise np.linalg.LinAlgError("Matrix is singular modulo p (determinant is 0)")

    # Calculate inverse of determinant modulo p
    try:
        inv_det_A = modInverse(det_A, p)
    except ValueError:
        raise np.linalg.LinAlgError(f"Determinant {det_A} has no inverse modulo {p}")

    # Calculate adjugate matrix (cofactor matrix transposed) modulo p
    adj_A = np.zeros_like(matrix_A, dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix_A, i, axis=0), j, axis=1)
            cofactor = ((-1)**(i + j)) * int(round(np.linalg.det(minor)))
            adj_A[j, i] = cofactor % p # Transpose happens here (j,i)

    # Calculate inverse matrix: A_inv = (det(A)^-1 * adj(A)) mod p
    inv_A = (inv_det_A * adj_A) % p

    # Calculate solution: x = A_inv * b mod p
    solution_x = (inv_A @ vector_b.astype(int)) % p # Ensure vector_b is int for matmul

    return solution_x

# --- Image Processing and Sharing Functions ---

def preprocess_image(image_path):
    """Loads an image, converts to grayscale, and returns as numpy array."""
    try:
        img = Image.open(image_path).convert('L') # 'L' mode is 8-bit grayscale
        img_data = np.array(img, dtype=np.uint8)
        # Apply modulo PRIME here if needed, or handle in create_shares
        # It's better to handle in create_shares where the secret is processed
        return img_data
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def create_shares(image_data, k, n):
    """
    Splits the image_data into n shares using SSS (k-out-of-n).
    Uses PRIME for calculations.
    Returns a list of n numpy arrays (numerical shares).
    """
    height, width = image_data.shape
    # Numerical shares need to store values up to PRIME-1
    shares_data = [np.zeros_like(image_data, dtype=np.int32) for _ in range(n)]

    print(f"Creating {n} shares (k={k}) for image of size {width}x{height} using PRIME={PRIME}...")
    for r in range(height):
        for c in range(width):
            # The secret is the pixel value MODULO PRIME
            secret_pixel_value = int(image_data[r, c]) % PRIME

            # Generate a polynomial of degree k-1 for this pixel
            coeffs = generate_polynomial(k - 1, secret_pixel_value)

            # Generate n share values for this pixel by evaluating P(x) at x=1, 2, ..., n
            for i in range(n):
                share_x = i + 1 # Use x = 1, 2, ..., n
                share_value = evaluate_polynomial(coeffs, share_x)
                shares_data[i][r, c] = share_value
        if (r + 1) % (height // 10) == 0 or r == height - 1:
             print(f"  Processed row {r+1}/{height}")

    print("Share creation complete.")
    # Return the numerical shares (values 0 to PRIME-1)
    return shares_data

def create_nplus1_share(numerical_shares_n, p):
    """
    Creates the (n+1)th share by summing the first n shares pixel-wise mod p.
    numerical_shares_n: List of n numpy arrays (the original shares).
    p: The prime modulus used.
    """
    if not numerical_shares_n:
        raise ValueError("Need at least one share to create the n+1 share.")

    n = len(numerical_shares_n)
    print(f"Creating the {n+1}-th share by summing shares 1 to {n} (mod {p})...")

    # Initialize the n+1 share
    share_nplus1 = np.zeros_like(numerical_shares_n[0], dtype=np.int32)

    # Sum pixel-wise modulo p
    for share in numerical_shares_n:
        share_nplus1 = (share_nplus1 + share) % p

    print(f"Share {n+1} created.")
    return share_nplus1

def get_visual_share(numerical_share):
     """Converts numerical share (0 to PRIME-1) to visual share (0-255 uint8)."""
     # Simple modulo mapping for visualization
     return (numerical_share % 256).astype(np.uint8)

# --- Hashing for Integrity ---

def calculate_hash(data):
    """Calculates SHA-256 hash of numpy array data."""
    return hashlib.sha256(data.tobytes()).hexdigest()

# --- Saving and Loading ---

def save_shares_and_hashes(all_numerical_shares, base_filename):
    """
    Saves visual shares (PNG), numerical shares (NPY), and hashes (TXT).
    all_numerical_shares: List containing ALL shares (1 to n+1).
    """
    num_total_shares = len(all_numerical_shares)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    share_hashes = []
    for i, numerical_share in enumerate(all_numerical_shares):
        share_index = i + 1 # Share indices are 1-based
        visual_share = get_visual_share(numerical_share)

        # Save visual share (noise-like image)
        share_filename_png = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{share_index}_visual.png")
        try:
            Image.fromarray(visual_share).save(share_filename_png)
            # print(f"Saved visual share {share_index} to {share_filename_png}")
        except Exception as e:
            print(f"Error saving visual share {share_index}: {e}")

        # Save numerical share (needed for reconstruction)
        share_filename_npy = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{share_index}_numerical.npy")
        try:
            np.save(share_filename_npy, numerical_share)
            # print(f"Saved numerical share {share_index} data to {share_filename_npy}")

            # Calculate and store hash of the numerical share
            share_hash = calculate_hash(numerical_share)
            share_hashes.append(share_hash)
            hash_file = SHARE_HASH_FILE_TEMPLATE.format(share_index)
            with open(hash_file, 'w') as f:
                f.write(share_hash)
            # print(f"Saved hash for share {share_index} to {hash_file}")

        except Exception as e:
            print(f"Error saving numerical share {share_index} data or hash: {e}")

    print(f"Saved {num_total_shares} shares (visual+numerical) and hashes.")
    return share_hashes

def load_numerical_shares(base_filename, indices_to_load):
    """Loads specified numerical shares and their hashes from files."""
    loaded_shares = []
    loaded_indices = []
    loaded_hashes = []
    valid_shares = True
    print(f"Attempting to load numerical shares for indices: {indices_to_load}")
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
                loaded_indices.append(i) # Store the original index (1 to n+1)
                loaded_hashes.append(actual_hash)
                print(f" -> Loaded and verified numerical share {i}")
            else:
                print(f"ERROR: Hash mismatch for share {i}! Share is corrupt or tampered.")
                valid_shares = False
                # Stop loading if a share is invalid? Or just skip it?
                # For now, we'll mark as invalid and potentially fail reconstruction later.
        except FileNotFoundError:
            print(f"ERROR: Share file or hash file for index {i} not found.")
            valid_shares = False
        except Exception as e:
            print(f"ERROR loading share {i}: {e}")
            valid_shares = False

    # Check if we loaded enough shares successfully
    if len(loaded_shares) != len(indices_to_load):
        print("ERROR: Failed to load one or more required shares.")
        valid_shares = False

    return loaded_shares, loaded_indices, loaded_hashes, valid_shares

# --- Reconstruction ---

def reconstruct_image(shares_to_reconstruct, share_indices, k, n_orig, p):
    """
    Reconstructs the image from k shares using appropriate method.
    shares_to_reconstruct: List of k numpy arrays (numerical share data).
    share_indices: List of the original indices (1 to n_orig+1) corresponding to the shares.
    k: The threshold.
    n_orig: The number of *original* shares (before adding the n+1 th).
    p: The prime modulus.
    """
    if len(shares_to_reconstruct) != k or len(share_indices) != k:
        raise ValueError(f"Need exactly k={k} shares and indices to reconstruct. Got {len(shares_to_reconstruct)}.")

    height, width = shares_to_reconstruct[0].shape
    reconstructed_data = np.zeros((height, width), dtype=np.int32)

    n_plus_1_index = n_orig + 1
    n_plus_1_share_present = n_plus_1_index in share_indices

    print(f"Reconstructing image from {k} shares (k={k}, n_orig={n_orig}). Share {n_plus_1_index} present: {n_plus_1_share_present}")

    # Precompute power sums if using the linear system method
    power_sums = None
    if n_plus_1_share_present:
        print(" -> Using linear system solving method.")
        try:
            # Degree of polynomial is k-1. We need sums up to power k-1.
            power_sums = precompute_power_sums(n_orig, k - 1, p)
            # print(f"DEBUG: Power sums for n={n_orig}, k-1={k-1}: {power_sums}")
        except Exception as e:
            print(f"ERROR: Failed to precompute power sums: {e}")
            raise e # Cannot proceed without them
    else:
        print(" -> Using Lagrange interpolation method.")


    # Iterate through each pixel position
    for r in range(height):
        for c in range(width):
            # Gather the points (x, y) or equations for this pixel
            points = [] # For Lagrange
            matrix_rows = [] # For Linear System
            vector_b = [] # For Linear System
            value_nplus1 = -1 # Store value if n+1 share is present

            for i in range(k):
                idx = share_indices[i]
                val = int(shares_to_reconstruct[i][r, c])

                if n_plus_1_share_present:
                    # Construct row for linear system M*A = Y
                    if idx == n_plus_1_index:
                         # Row corresponds to equation: n*a0 + sum(i)*a1 + ... = val_n+1
                         row = [power_sums.get(j, 0) for j in range(k)] # j=0..k-1 for powers
                         matrix_rows.append(row)
                         vector_b.append(val)
                         value_nplus1 = val # Keep track for debugging maybe
                    else:
                         # Row corresponds to standard Shamir eval: a0 + idx*a1 + idx^2*a2 + ... = val_idx
                         row = [(idx**j) % p for j in range(k)] # Powers 0 to k-1
                         matrix_rows.append(row)
                         vector_b.append(val)
                else:
                    # Just gather points for Lagrange
                    points.append((idx, val))

            # --- Solve for the secret pixel value (a0) ---
            try:
                if n_plus_1_share_present:
                    # Solve the linear system M * A = Y for A, secret is A[0]
                    matrix_M = np.array(matrix_rows, dtype=int)
                    vector_Y = np.array(vector_b, dtype=int)
                    # print(f"DEBUG ({r},{c}): M=\n{matrix_M}\n Y=\n{vector_Y}") # Careful, very verbose
                    solution_A = solve_linear_system_mod_p(matrix_M, vector_Y, p)
                    reconstructed_pixel_value = solution_A[0] # a0 is the secret
                else:
                    # Use Lagrange interpolation
                    reconstructed_pixel_value = lagrange_interpolate(points, x_target=0)

                reconstructed_data[r, c] = reconstructed_pixel_value

            except (ValueError, np.linalg.LinAlgError) as e:
                # Handle errors like non-invertible matrix or Lagrange issues
                print(f"ERROR during reconstruction at pixel ({r},{c}): {e}")
                # print(f" -> Shares used: {share_indices}")
                # print(f" -> Values at pixel: {[int(s[r,c]) for s in shares_to_reconstruct]}")
                # if n_plus_1_share_present: print(f" -> Matrix M:\n{matrix_M}\n Vector Y:\n{vector_Y}")
                # else: print(f" -> Points: {points}")

                # Set pixel to a default error value (e.g., 0 or midpoint)
                reconstructed_data[r, c] = 128 # Indicate error with gray
                # Optionally: Stop entire reconstruction on first error
                # raise RuntimeError(f"Reconstruction failed at pixel ({r},{c})") from e


        if (r + 1) % (height // 10) == 0 or r == height - 1:
             print(f"  Reconstructed row {r+1}/{height}")

    print("Image reconstruction computation complete.")
    # Clip values to 0-255 and convert to uint8 for the final image
    # Since PRIME=251, values should be 0-250. Clipping > 250 shouldn't be needed.
    reconstructed_image = np.clip(reconstructed_data, 0, 255).astype(np.uint8)
    return reconstructed_image


# --- Display ---

def display_images(original, all_shares_numerical, reconstructed, k, n_orig, indices_used):
    """Displays original, some shares (visual), and reconstructed images."""
    num_total_shares = n_orig + 1
    # Try to display original, n+1 share, one original share, and reconstructed
    shares_to_display_indices = []
    if 1 <= num_total_shares: shares_to_display_indices.append(1) # Show first original share
    if n_orig + 1 <= num_total_shares: shares_to_display_indices.append(n_orig + 1) # Show the combined share

    num_shares_display = len(shares_to_display_indices)
    num_cols = 2 + num_shares_display
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))
    fig.suptitle(f"Shamir ({k}, {n_orig}) Extended to ({k}, {n_orig+1}) Reconstruction | Used Shares: {indices_used}")

    # Display Original
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display Selected Shares (Visual representations)
    share_plot_index = 1
    for idx_to_show in shares_to_display_indices:
         # Get visual share data
         visual_share_img = get_visual_share(all_shares_numerical[idx_to_show - 1]) # List is 0-indexed
         axes[share_plot_index].imshow(visual_share_img, cmap='gray', vmin=0, vmax=255)
         title = f"Share {idx_to_show}"
         if idx_to_show == n_orig + 1: title += " (Combined)"
         axes[share_plot_index].set_title(title)
         axes[share_plot_index].axis('off')
         share_plot_index += 1

    # Display Reconstructed
    if reconstructed is not None:
        axes[num_cols - 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
        axes[num_cols - 1].set_title(f"Reconstructed\n(from {k} shares)")
        axes[num_cols - 1].axis('off')
    else:
        axes[num_cols - 1].set_title("Reconstruction\nFailed/Not Run")
        axes[num_cols - 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # --- Parameters ---
    image_path = input("Enter the path to the image file: ") # e.g., 'lena_gray.png'
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist. Please provide a valid path.")
        exit()

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    while True:
        try:
            n_orig = int(input("Enter the initial number of shares (n >= 2): "))
            if n_orig >= 2:
                break
            else: print("n must be at least 2.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        try:
            k = int(input(f"Enter the threshold (k, where 2 <= k <= {n_orig}): "))
            if 2 <= k <= n_orig:
                break
            else: print(f"k must be between 2 and {n_orig}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


    # 1. Image Preprocessing
    print("-" * 30)
    print("1. Preprocessing Image...")
    original_image_data = preprocess_image(image_path)

    if original_image_data is not None:
        height, width = original_image_data.shape
        print(f"Image loaded successfully ({width}x{height}).")

        # Apply modulo PRIME to original data *before* hashing if PRIME < 256
        # This ensures hash comparison works correctly later if lossy prime used
        original_data_for_hash = original_image_data.astype(np.int32) % PRIME
        original_hash = calculate_hash(original_data_for_hash)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        with open(ORIGINAL_HASH_FILE, 'w') as f:
            f.write(original_hash)
        print(f"Original image hash (using mod {PRIME}, {original_hash[:8]}...) saved.")

        # 2. Create Initial Shares (SSS on Pixel Intensity)
        print("-" * 30)
        print(f"2. Creating Initial {n_orig} Shares (k={k})...")
        # shares_numerical are int32/higher for math, values 0 to PRIME-1
        shares_numerical_n = create_shares(original_image_data, k, n_orig)

        # 3. Create the (n+1)-th Share
        print("-" * 30)
        print(f"3. Creating Share {n_orig + 1} using Participant Increasing Method...")
        share_nplus1 = create_nplus1_share(shares_numerical_n, PRIME)
        all_numerical_shares = shares_numerical_n + [share_nplus1] # Combine all shares

        # 4. Secure Storage Simulation
        print("-" * 30)
        print("4. Saving All Shares and Hashes...")
        save_shares_and_hashes(all_numerical_shares, base_filename)
        print(f"All {n_orig + 1} shares and hashes saved in directory: {OUTPUT_DIR}")

        # 5. Reconstruction Simulation
        print("-" * 30)
        print(f"5. Simulating Reconstruction (Requires k={k} shares from {n_orig + 1} total)...")

        # --- Reconstruction Prompt ---
        reconstructed_image = None
        indices_used_for_recon = []
        while True: # Loop until valid input or quit
             print(f"\nTotal available shares: 1 to {n_orig + 1}")
             indices_str = input(f"Enter exactly {k} unique share indices (e.g., '1 3 {n_orig+1}') or 'q' to quit: ")
             if indices_str.lower() == 'q':
                 break
             try:
                 indices_to_reconstruct = sorted(list(set([int(idx) for idx in indices_str.split()]))) # Get unique, sorted indices

                 if len(indices_to_reconstruct) != k:
                      print(f"Error: Please provide exactly {k} unique indices. You provided {len(indices_to_reconstruct)}.")
                      continue
                 if not all(1 <= idx <= (n_orig + 1) for idx in indices_to_reconstruct):
                      print(f"Error: Indices must be between 1 and {n_orig + 1}.")
                      continue

                 # --- Perform Reconstruction ---
                 print(f"\nAttempting reconstruction using shares: {indices_to_reconstruct}")
                 indices_used_for_recon = indices_to_reconstruct # Store for display

                 # Load the required *numerical* shares and verify hashes
                 shares_for_recon, indices_loaded, hashes_loaded, shares_valid = load_numerical_shares(
                     base_filename, indices_to_reconstruct
                 )

                 if shares_valid and len(shares_for_recon) == k:
                     # Perform reconstruction using appropriate method
                     reconstructed_image = reconstruct_image(
                         shares_for_recon, indices_loaded, k, n_orig, PRIME
                     )

                     # 6. Integrity Verification
                     print("-" * 30)
                     print("6. Verifying Integrity...")
                     # Hash the reconstructed image (after conversion back to uint8)
                     reconstructed_hash = calculate_hash(reconstructed_image.astype(np.int32) % PRIME)
                     print(f"Reconstructed image hash (using mod {PRIME}): {reconstructed_hash[:8]}...")
                     print(f"Original image hash      (using mod {PRIME}): {original_hash[:8]}...")
                     if reconstructed_hash == original_hash:
                         print("Integrity Check PASSED: Reconstructed image matches the original (mod {PRIME}).")
                         if PRIME < 256: print("Note: Minor visual differences possible if original had pixels > 250.")
                     else:
                         print("Integrity Check FAILED: Reconstructed image does NOT match the original (mod {PRIME})!")
                     break # Exit loop after successful reconstruction attempt

                 else:
                      print("Reconstruction aborted due to missing, invalid, or insufficient shares loaded.")
                      # Stay in loop to allow user to try different indices

             except ValueError as e:
                  print(f"Invalid input: {e}. Please enter space-separated integers.")
             except (np.linalg.LinAlgError, RuntimeError) as e:
                  print(f"ERROR during reconstruction: {e}")
                  print("This might happen if the chosen shares lead to a non-invertible system (rare) or other calculation error.")
                  reconstructed_image = None # Ensure image is None if error occurs
                  # Stay in loop to allow user to try different indices
             except Exception as e:
                  print(f"An unexpected error occurred: {e}")
                  reconstructed_image = None
                  # Stay in loop or break depending on severity? For now, stay.


        # 7. Display Results (if reconstruction was attempted)
        if indices_used_for_recon: # Check if user attempted reconstruction
            print("-" * 30)
            print("7. Displaying Results...")
            display_images(original_image_data, all_numerical_shares, reconstructed_image, k, n_orig, indices_used_for_recon)
        else:
            print("\nExited without attempting reconstruction.")

    else:
        print("Could not load the original image. Exiting.")

