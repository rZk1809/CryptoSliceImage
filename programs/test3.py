import numpy as np
from PIL import Image
import os
import random
import hashlib
import math
import matplotlib.pyplot as plt
import itertools
import time
import concurrent.futures
import multiprocessing # To get CPU count

# --- Configuration ---
PRIME = 251 # Use the prime specified in the paper (lossy for pixels 251-255)
# PRIME = 257 # Use a prime > 255 for lossless 8-bit grayscale
print(f"INFO: Using PRIME = {PRIME}. Pixel values > {PRIME-1} will be reduced modulo {PRIME}.")

OUTPUT_DIR = "image_shares_extended_timed_parallel"
ORIGINAL_HASH_FILE = os.path.join(OUTPUT_DIR, "original_hash.txt")
SHARE_HASH_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, "share_{}_hash.txt")

# Determine optimal number of workers for parallel processing
NUM_WORKERS = multiprocessing.cpu_count()
print(f"INFO: Using {NUM_WORKERS} workers for parallel processing.")

# --- Visual Cryptography Note ---
# This implementation uses Shamir's Secret Sharing (SSS) on pixel intensity values.
# While the resulting shares appear as random noise (achieving visual security),
# it does NOT use traditional Visual Cryptography (VC) techniques like
# pixel expansion and physical stacking for binary images.
# Reconstruction here is computational (Lagrange or Linear Algebra).

# --- Shamir's Secret Sharing Core Functions (Modulo PRIME) ---
# (extended_gcd, modInverse, generate_polynomial, evaluate_polynomial remain the same)
def extended_gcd(a, b):
    if a == 0: return (b, 0, 1)
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return (gcd, x, y)

def modInverse(a, m):
    a = a % m
    if a < 0: a += m
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1: raise ValueError(f'Modular inverse does not exist for {a} mod {m}')
    return (x % m + m) % m

def generate_polynomial(degree, secret):
    secret = int(secret) % PRIME
    coefficients = [secret] + [random.randint(0, PRIME - 1) for _ in range(degree)]
    return [c % PRIME for c in coefficients]

def evaluate_polynomial(coeffs, x):
    result = 0
    power_of_x = 1
    x_mod = x % PRIME
    for coeff in coeffs:
        term = (coeff * power_of_x) % PRIME
        result = (result + term) % PRIME
        power_of_x = (power_of_x * x_mod) % PRIME
    return result

# --- Lagrange Interpolation (Same as before) ---
def lagrange_interpolate(points, x_target=0):
    if not points: raise ValueError("Need points for interpolation")
    k = len(points)
    xs, ys = zip(*points)
    if len(set(xs)) != k: raise ValueError("X values must be distinct")
    secret = 0
    x_target_mod = x_target % PRIME
    for i in range(k):
        num, den = 1, 1
        xi = xs[i] % PRIME
        for j in range(k):
            if i == j: continue
            xj = xs[j] % PRIME
            num = (num * (x_target_mod - xj)) % PRIME
            den = (den * (xi - xj)) % PRIME
        try:
            inv_den = modInverse(den, PRIME)
            term = (ys[i] * num * inv_den) % PRIME
            secret = (secret + term) % PRIME
        except ValueError as e:
            print(f"ERROR in Lagrange: {e}. Points: {points}, i={i}, denom={den}")
            raise e
    return (secret + PRIME) % PRIME


# --- Linear Algebra for Reconstruction with N+1 Share (Same as before) ---
# (precompute_power_sums, solve_linear_system_mod_p remain the same)
def precompute_power_sums(n_orig, k_degree, p):
    power_sums = {}
    current_powers = [1] * n_orig
    for j in range(k_degree + 1):
        current_sum = sum(current_powers) % p
        power_sums[j] = current_sum
        if j < k_degree:
            for i in range(1, n_orig + 1):
                current_powers[i-1] = (current_powers[i-1] * i) % p
    return power_sums

def solve_linear_system_mod_p(matrix_A, vector_b, p):
    n = matrix_A.shape[0]
    det_A = int(round(np.linalg.det(matrix_A))) % p
    if det_A == 0: raise np.linalg.LinAlgError("Matrix singular mod p")
    try: inv_det_A = modInverse(det_A, p)
    except ValueError: raise np.linalg.LinAlgError(f"Det {det_A} no inverse mod {p}")
    adj_A = np.zeros_like(matrix_A, dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix_A, i, axis=0), j, axis=1)
            cofactor = ((-1)**(i + j)) * int(round(np.linalg.det(minor)))
            adj_A[j, i] = cofactor % p
    inv_A = (inv_det_A * adj_A) % p
    solution_x = (inv_A @ vector_b.astype(int)) % p
    return solution_x

# --- Image Processing and Sharing Functions ---

def preprocess_image(image_path):
    """Loads image, converts to grayscale, returns numpy array."""
    try:
        img = Image.open(image_path).convert('L')
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- Parallel Share Creation ---

def _create_shares_chunk_processor(args):
    """Worker function for creating shares for a chunk of rows."""
    img_chunk, k, n, p, start_row, height, width = args
    chunk_height = img_chunk.shape[0]
    # Initialize share chunks for this part
    share_chunks = [np.zeros((chunk_height, width), dtype=np.int32) for _ in range(n)]

    for r_local in range(chunk_height): # Row within the chunk
        r_global = start_row + r_local # Global row index
        for c in range(width):
            secret_pixel_value = int(img_chunk[r_local, c]) % p
            coeffs = generate_polynomial(k - 1, secret_pixel_value)
            for i in range(n):
                share_x = i + 1
                share_value = evaluate_polynomial(coeffs, share_x)
                share_chunks[i][r_local, c] = share_value
    # Return the list of share chunks for this part
    return share_chunks

def create_shares_parallel(image_data, k, n):
    """
    Splits image_data into n shares using SSS (k-out-of-n) in parallel.
    """
    height, width = image_data.shape
    # Final shares array initialized once
    shares_data = [np.zeros_like(image_data, dtype=np.int32) for _ in range(n)]
    print(f"Creating {n} shares (k={k}) in PARALLEL using {NUM_WORKERS} workers...")

    # Divide image data into chunks (e.g., by rows)
    chunk_size = max(1, height // NUM_WORKERS) # Avoid chunk_size 0
    tasks = []
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        # Ensure we don't create an empty chunk if height is multiple of chunk_size
        if start_row == end_row: continue
        img_chunk = image_data[start_row:end_row, :]
        tasks.append((img_chunk, k, n, PRIME, start_row, height, width)) # task[4] is start_row

    results_temp = [] # Store results temporarily before sorting
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Create a map from future object to its start_row
        futures_map = {executor.submit(_create_shares_chunk_processor, task): task[4] for task in tasks}

        for future in concurrent.futures.as_completed(futures_map):
            start_row = futures_map[future] # Get the correct start_row for this future
            try:
                # Result is a list of n share chunks for that task
                chunk_result = future.result()
                # Append the start_row and the result list
                results_temp.append((start_row, chunk_result))
                print(f"  Worker finished chunk starting at row {start_row}")
            except Exception as e:
                print(f"Error in worker process for chunk starting at row {start_row}: {e}")
                # Handle error, maybe raise it to stop everything
                raise e

    # Sort results by start_row to assemble correctly
    results_temp.sort(key=lambda x: x[0])

    # Assemble the full shares from the chunks
    print("Assembling results...")
    for start_row, chunk_result_list in results_temp:
        # chunk_result_list contains n numpy arrays (one for each share's chunk)
        if not chunk_result_list: # Should not happen if worker succeeded
             print(f"Warning: Empty result for chunk starting at row {start_row}")
             continue
        chunk_height = chunk_result_list[0].shape[0]
        if chunk_height == 0: # Skip empty chunks if they somehow occur
             continue
        end_row = start_row + chunk_height
        # Ensure slicing indices are within bounds (should be if logic is correct)
        if start_row < 0 or end_row > height:
             print(f"Error: Calculated slice [{start_row}:{end_row}] out of bounds for height {height}.")
             raise IndexError("Slice calculation error during assembly.")

        # Assign the chunk to the correct slice in each of the final share arrays
        for share_idx in range(n):
            # Get the specific share chunk from the list returned by the worker
            source_chunk = chunk_result_list[share_idx]
            # Define the target slice in the final share array
            target_slice = shares_data[share_idx][start_row:end_row, :]

            # --- Debugging Print ---
            # print(f"  Assembling share {share_idx}, rows [{start_row}:{end_row}]: target_shape={target_slice.shape}, source_shape={source_chunk.shape}")

            # Perform the assignment
            try:
                 target_slice[:] = source_chunk
            except ValueError as e:
                 print(f"\n--- BROADCAST ERROR ---")
                 print(f"Failed to assign chunk to share {share_idx}, rows [{start_row}:{end_row}]")
                 print(f"Target slice shape: {target_slice.shape}")
                 print(f"Source chunk shape: {source_chunk.shape}")
                 print(f"Original error: {e}")
                 print(f"-----------------------\n")
                 # Re-raise the error to stop execution, as this indicates a fundamental problem
                 raise e


    print("Parallel share creation complete.")
    return shares_data
# --- Sequential Share Creation (for comparison) ---
def create_shares_sequential(image_data, k, n):
    height, width = image_data.shape
    shares_data = [np.zeros_like(image_data, dtype=np.int32) for _ in range(n)]
    print(f"Creating {n} shares (k={k}) SEQUENTIALLY...")
    for r in range(height):
        for c in range(width):
            secret_pixel_value = int(image_data[r, c]) % PRIME
            coeffs = generate_polynomial(k - 1, secret_pixel_value)
            for i in range(n):
                share_x = i + 1
                share_value = evaluate_polynomial(coeffs, share_x)
                shares_data[i][r, c] = share_value
        # Optional: Print progress
        if (r + 1) % (max(1, height // 10)) == 0 or r == height - 1:
             print(f"  Processed row {r+1}/{height}")
    print("Sequential share creation complete.")
    return shares_data

# --- Create N+1 Share (Sequential is fast enough) ---
def create_nplus1_share(numerical_shares_n, p):
    n = len(numerical_shares_n)
    print(f"Creating the {n+1}-th share by summing shares 1 to {n} (mod {p})...")
    share_nplus1 = np.zeros_like(numerical_shares_n[0], dtype=np.int32)
    for share in numerical_shares_n:
        share_nplus1 = (share_nplus1 + share) % p
    print(f"Share {n+1} created.")
    return share_nplus1

# --- Get Visual Share (Same as before) ---
def get_visual_share(numerical_share):
     return (numerical_share % 256).astype(np.uint8)

# --- Hashing for Integrity (Same as before) ---
def calculate_hash(data):
    return hashlib.sha256(data.tobytes()).hexdigest()

# --- Saving and Loading (Same as before) ---
def save_shares_and_hashes(all_numerical_shares, base_filename):
    num_total_shares = len(all_numerical_shares)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    share_hashes = []
    for i, numerical_share in enumerate(all_numerical_shares):
        share_index = i + 1
        visual_share = get_visual_share(numerical_share)
        # Save visual share
        share_filename_png = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{share_index}_visual.png")
        try: Image.fromarray(visual_share).save(share_filename_png)
        except Exception as e: print(f"Error saving visual share {share_index}: {e}")
        # Save numerical share
        share_filename_npy = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{share_index}_numerical.npy")
        try:
            np.save(share_filename_npy, numerical_share)
            share_hash = calculate_hash(numerical_share)
            share_hashes.append(share_hash)
            hash_file = SHARE_HASH_FILE_TEMPLATE.format(share_index)
            with open(hash_file, 'w') as f: f.write(share_hash)
        except Exception as e: print(f"Error saving numerical share {share_index} data or hash: {e}")
    print(f"Saved {num_total_shares} shares (visual+numerical) and hashes.")
    return share_hashes

def load_numerical_shares(base_filename, indices_to_load):
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
            with open(hash_file, 'r') as f: expected_hash = f.read().strip()
            actual_hash = calculate_hash(share_data)
            if actual_hash == expected_hash:
                loaded_shares.append(share_data)
                loaded_indices.append(i)
                loaded_hashes.append(actual_hash)
                print(f" -> Loaded and verified numerical share {i}")
            else:
                print(f"ERROR: Hash mismatch for share {i}!")
                valid_shares = False
        except FileNotFoundError:
            print(f"ERROR: Share file or hash file for index {i} not found.")
            valid_shares = False
        except Exception as e:
            print(f"ERROR loading share {i}: {e}")
            valid_shares = False
    if len(loaded_shares) != len(indices_to_load):
        print("ERROR: Failed to load one or more required shares.")
        valid_shares = False
    return loaded_shares, loaded_indices, loaded_hashes, valid_shares

# --- Parallel Reconstruction ---

def _reconstruct_chunk_processor(args):
    """Worker function for reconstructing a chunk of rows."""
    share_chunks, share_indices, k, n_orig, p, start_row, end_row, width, n_plus_1_present, power_sums = args

    chunk_height = end_row - start_row
    reconstructed_chunk = np.zeros((chunk_height, width), dtype=np.int32)
    n_plus_1_index = n_orig + 1

    for r_local in range(chunk_height): # Row within the chunk
        for c in range(width):
            # Gather points or build system for this pixel (r_local, c) within the chunks
            points = [] # For Lagrange
            matrix_rows = [] # For Linear System
            vector_b = [] # For Linear System

            for i in range(k): # Iterate through the k selected shares
                idx = share_indices[i]
                # Get value from the i-th share chunk at local row r_local, column c
                val = int(share_chunks[i][r_local, c])

                if n_plus_1_present:
                    if idx == n_plus_1_index:
                         row = [power_sums.get(j, 0) for j in range(k)]
                         matrix_rows.append(row)
                         vector_b.append(val)
                    else:
                         row = [(idx**j) % p for j in range(k)]
                         matrix_rows.append(row)
                         vector_b.append(val)
                else:
                    points.append((idx, val))

            # Solve for the secret pixel value
            try:
                if n_plus_1_present:
                    matrix_M = np.array(matrix_rows, dtype=int)
                    vector_Y = np.array(vector_b, dtype=int)
                    solution_A = solve_linear_system_mod_p(matrix_M, vector_Y, p)
                    reconstructed_pixel_value = solution_A[0]
                else:
                    reconstructed_pixel_value = lagrange_interpolate(points, x_target=0)
                reconstructed_chunk[r_local, c] = reconstructed_pixel_value
            except (ValueError, np.linalg.LinAlgError) as e:
                # Log error and set default value
                # print(f"WARN: Recon error at pixel ({start_row + r_local},{c}): {e}. Setting to 128.")
                reconstructed_chunk[r_local, c] = 128 # Indicate error
            # Note: Catching errors here might hide widespread issues.
            # Consider re-raising serious errors if needed.

    return (start_row, reconstructed_chunk)


def reconstruct_image_parallel(shares_to_reconstruct, share_indices, k, n_orig, p):
    """Reconstructs the image from k shares in parallel."""
    if len(shares_to_reconstruct) != k or len(share_indices) != k:
        raise ValueError(f"Need exactly k={k} shares/indices. Got {len(shares_to_reconstruct)}.")

    height, width = shares_to_reconstruct[0].shape
    reconstructed_data = np.zeros((height, width), dtype=np.int32) # Final image

    n_plus_1_index = n_orig + 1
    n_plus_1_present = n_plus_1_index in share_indices

    print(f"Reconstructing image PARALLEL (k={k}, n_orig={n_orig}). Share {n_plus_1_index} present: {n_plus_1_present}")

    power_sums = None
    if n_plus_1_present:
        print(" -> Using linear system solving method.")
        try: power_sums = precompute_power_sums(n_orig, k - 1, p)
        except Exception as e: raise RuntimeError(f"Failed to precompute power sums: {e}") from e
    else:
        print(" -> Using Lagrange interpolation method.")

    # Divide work into chunks
    chunk_size = max(1, height // NUM_WORKERS)
    tasks = []
    for i in range(0, height, chunk_size):
        start_row = i
        end_row = min(i + chunk_size, height)
        # Extract the relevant row chunks from EACH of the k input shares
        share_chunks_for_task = [share[start_row:end_row, :] for share in shares_to_reconstruct]
        tasks.append((share_chunks_for_task, share_indices, k, n_orig, p, start_row, end_row, width, n_plus_1_present, power_sums))

    results = {} # Use dict to store results keyed by start_row
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_reconstruct_chunk_processor, task): task[5] for task in tasks} # key future by start_row
        for future in concurrent.futures.as_completed(futures):
            start_row = futures[future]
            try:
                s_row, recon_chunk = future.result()
                results[s_row] = recon_chunk # Store chunk associated with its start row
                print(f"  Worker finished recon chunk starting at row {s_row}")
            except Exception as e:
                print(f"Error in reconstruction worker for chunk starting {start_row}: {e}")
                raise e # Stop on error

    # Assemble the full reconstructed image
    print("Assembling reconstruction results...")
    for start_row, recon_chunk in sorted(results.items()): # Sort by start_row
         end_row = start_row + recon_chunk.shape[0]
         reconstructed_data[start_row:end_row, :] = recon_chunk

    print("Parallel image reconstruction complete.")
    reconstructed_image = np.clip(reconstructed_data, 0, 255).astype(np.uint8)
    return reconstructed_image


# --- Sequential Reconstruction (for comparison) ---
def reconstruct_image_sequential(shares_to_reconstruct, share_indices, k, n_orig, p):
    if len(shares_to_reconstruct) != k or len(share_indices) != k:
        raise ValueError(f"Need exactly k={k} shares/indices. Got {len(shares_to_reconstruct)}.")
    height, width = shares_to_reconstruct[0].shape
    reconstructed_data = np.zeros((height, width), dtype=np.int32)
    n_plus_1_index = n_orig + 1
    n_plus_1_present = n_plus_1_index in share_indices
    print(f"Reconstructing image SEQUENTIAL (k={k}, n_orig={n_orig}). Share {n_plus_1_index} present: {n_plus_1_present}")
    power_sums = None
    if n_plus_1_present:
        print(" -> Using linear system solving method.")
        try: power_sums = precompute_power_sums(n_orig, k - 1, p)
        except Exception as e: raise RuntimeError(f"Failed to precompute power sums: {e}") from e
    else: print(" -> Using Lagrange interpolation method.")

    # Pixel iteration
    for r in range(height):
        for c in range(width):
            points = []
            matrix_rows = []
            vector_b = []
            for i in range(k):
                idx = share_indices[i]
                val = int(shares_to_reconstruct[i][r, c])
                if n_plus_1_present:
                    if idx == n_plus_1_index:
                         row = [power_sums.get(j, 0) for j in range(k)]
                         matrix_rows.append(row); vector_b.append(val)
                    else:
                         row = [(idx**j) % p for j in range(k)]
                         matrix_rows.append(row); vector_b.append(val)
                else: points.append((idx, val))
            try:
                if n_plus_1_present:
                    matrix_M = np.array(matrix_rows, dtype=int)
                    vector_Y = np.array(vector_b, dtype=int)
                    solution_A = solve_linear_system_mod_p(matrix_M, vector_Y, p)
                    reconstructed_pixel_value = solution_A[0]
                else: reconstructed_pixel_value = lagrange_interpolate(points, x_target=0)
                reconstructed_data[r, c] = reconstructed_pixel_value
            except (ValueError, np.linalg.LinAlgError) as e:
                # print(f"WARN: Seq Recon error at pixel ({r},{c}): {e}. Setting to 128.")
                reconstructed_data[r, c] = 128 # Indicate error
        # Progress indicator
        if (r + 1) % (max(1,height // 10)) == 0 or r == height - 1:
             print(f"  Reconstructed row {r+1}/{height}")

    print("Sequential image reconstruction complete.")
    reconstructed_image = np.clip(reconstructed_data, 0, 255).astype(np.uint8)
    return reconstructed_image


# --- Display (Same as before) ---
def display_images(original, all_shares_numerical, reconstructed, k, n_orig, indices_used):
    num_total_shares = n_orig + 1
    shares_to_display_indices = []
    if 1 <= num_total_shares: shares_to_display_indices.append(1)
    if n_orig + 1 <= num_total_shares: shares_to_display_indices.append(n_orig + 1)
    num_shares_display = len(shares_to_display_indices)
    num_cols = 2 + num_shares_display
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))
    fig.suptitle(f"Shamir ({k}, {n_orig})->({k}, {n_orig+1}) | Used: {indices_used}")
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255); axes[0].set_title("Original"); axes[0].axis('off')
    share_plot_index = 1
    for idx_to_show in shares_to_display_indices:
         visual_share_img = get_visual_share(all_numerical_shares[idx_to_show - 1])
         axes[share_plot_index].imshow(visual_share_img, cmap='gray', vmin=0, vmax=255)
         title = f"Share {idx_to_show}" + (" (Comb.)" if idx_to_show == n_orig + 1 else "")
         axes[share_plot_index].set_title(title); axes[share_plot_index].axis('off')
         share_plot_index += 1
    if reconstructed is not None:
        axes[num_cols - 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
        axes[num_cols - 1].set_title(f"Reconstructed\n(from {k} shares)"); axes[num_cols - 1].axis('off')
    else: axes[num_cols - 1].set_title("Reconstruction\nFailed"); axes[num_cols - 1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    image_path = input("Enter the path to the image file: ")
    if not os.path.exists(image_path): print(f"Error: File '{image_path}' not found."); exit()
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    # --- Corrected Input Loop for n_orig ---
    while True:
        try:
            n_orig = int(input("Enter initial number of shares (n >= 2): "))
            # Check the value *after* successful conversion inside the try block
            if n_orig >= 2:
                break  # Exit loop if valid
            else:
                print("n must be at least 2.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# --- Corrected Input Loop for k ---
    while True:
        try:
            k = int(input(f"Enter threshold (k, 2 <= k <= {n_orig}): "))
             # Check the value *after* successful conversion inside the try block
            if 2 <= k <= n_orig:
                break # Exit loop if valid
            else:
                print(f"k must be between 2 and {n_orig}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 1. Preprocessing
    print("-" * 30); print("1. Preprocessing Image...")
    original_image_data = preprocess_image(image_path)
    if original_image_data is None: exit()
    height, width = original_image_data.shape
    print(f"Image loaded ({width}x{height}).")
    original_data_for_hash = original_image_data.astype(np.int32) % PRIME
    original_hash = calculate_hash(original_data_for_hash)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(ORIGINAL_HASH_FILE, 'w') as f: f.write(original_hash)
    print(f"Original hash (mod {PRIME}, {original_hash[:8]}...) saved.")

    # --- 2/3. Share Creation (Encryption) ---
    print("-" * 30); print(f"2/3. Creating {n_orig + 1} Total Shares (k={k})...")

    # --- Parallel Encryption ---
    start_time_p_enc = time.time()
    shares_numerical_n_p = create_shares_parallel(original_image_data, k, n_orig)
    share_nplus1_p = create_nplus1_share(shares_numerical_n_p, PRIME)
    end_time_p_enc = time.time()
    time_p_enc = end_time_p_enc - start_time_p_enc
    all_numerical_shares_p = shares_numerical_n_p + [share_nplus1_p]
    print(f"Parallel Encryption Time: {time_p_enc:.4f} seconds")

    # --- Sequential Encryption (Optional, for comparison) ---
    run_sequential_enc = input("Run sequential encryption for comparison? (y/n): ").lower() == 'y'
    time_s_enc = -1
    if run_sequential_enc:
        start_time_s_enc = time.time()
        shares_numerical_n_s = create_shares_sequential(original_image_data, k, n_orig)
        share_nplus1_s = create_nplus1_share(shares_numerical_n_s, PRIME)
        end_time_s_enc = time.time()
        time_s_enc = end_time_s_enc - start_time_s_enc
        print(f"Sequential Encryption Time: {time_s_enc:.4f} seconds")
        # Verify parallel and sequential results match (optional but recommended)
        # Check if all_numerical_shares_p == (shares_numerical_n_s + [share_nplus1_s])
        # This requires careful comparison due to potential floating point issues if not using int32
        print("Note: Verification between parallel/sequential shares not implemented.")

    # Use parallel shares for saving and reconstruction
    all_numerical_shares = all_numerical_shares_p

    # 4. Saving Shares
    print("-" * 30); print("4. Saving All Shares and Hashes...")
    save_shares_and_hashes(all_numerical_shares, base_filename)

    # 5. Reconstruction (Decryption)
    print("-" * 30); print(f"5. Simulating Reconstruction (Requires k={k} shares)...")
    reconstructed_image_p = None
    reconstructed_image_s = None
    indices_used_for_recon = []
    time_p_dec = -1
    time_s_dec = -1

    while True:
        print(f"\nTotal available shares: 1 to {n_orig + 1}")
        indices_str = input(f"Enter {k} unique share indices (e.g., '1 3 {n_orig+1}') or 'q' to quit: ")
        if indices_str.lower() == 'q': break
        try:
            indices_to_reconstruct = sorted(list(set([int(idx) for idx in indices_str.split()])))
            if len(indices_to_reconstruct) != k: print(f"Error: Need exactly {k} indices."); continue
            if not all(1 <= idx <= (n_orig + 1) for idx in indices_to_reconstruct): print(f"Error: Indices out of range [1, {n_orig+1}]."); continue
            indices_used_for_recon = indices_to_reconstruct

            # Load shares for reconstruction
            print(f"\nAttempting reconstruction using shares: {indices_to_reconstruct}")
            shares_for_recon, idx_loaded, _, shares_valid = load_numerical_shares(base_filename, indices_to_reconstruct)
            if not shares_valid or len(shares_for_recon) != k: print("Recon aborted: Invalid/missing shares."); continue

            # --- Parallel Decryption ---
            start_time_p_dec = time.time()
            reconstructed_image_p = reconstruct_image_parallel(shares_for_recon, idx_loaded, k, n_orig, PRIME)
            end_time_p_dec = time.time()
            time_p_dec = end_time_p_dec - start_time_p_dec
            print(f"Parallel Decryption Time: {time_p_dec:.4f} seconds")

            # --- Sequential Decryption (Optional) ---
            run_sequential_dec = input("Run sequential decryption for comparison? (y/n): ").lower() == 'y'
            if run_sequential_dec:
                start_time_s_dec = time.time()
                reconstructed_image_s = reconstruct_image_sequential(shares_for_recon, idx_loaded, k, n_orig, PRIME)
                end_time_s_dec = time.time()
                time_s_dec = end_time_s_dec - start_time_s_dec
                print(f"Sequential Decryption Time: {time_s_dec:.4f} seconds")
                # Verify agreement
                if reconstructed_image_p is not None and reconstructed_image_s is not None:
                     if np.array_equal(reconstructed_image_p, reconstructed_image_s):
                         print(" -> Sequential and Parallel reconstructions match.")
                     else:
                         print(" -> WARNING: Sequential and Parallel reconstructions DIFFER!")

            # --- Integrity Check (using parallel result) ---
            print("-" * 30); print("6. Verifying Integrity...")
            if reconstructed_image_p is not None:
                recon_hash = calculate_hash(reconstructed_image_p.astype(np.int32) % PRIME)
                print(f"Recon hash (mod {PRIME}): {recon_hash[:8]}...")
                print(f"Orig hash  (mod {PRIME}): {original_hash[:8]}...")
                if recon_hash == original_hash: print("Integrity Check PASSED.")
                else: print("Integrity Check FAILED!")
            else: print("Integrity check skipped (reconstruction failed).")
            break # Exit loop after successful attempt

        except ValueError as e: print(f"Invalid input: {e}")
        except (np.linalg.LinAlgError, RuntimeError, Exception) as e:
             print(f"ERROR during reconstruction processing: {e}")
             # Allow user to try again with different shares

    # 7. Display Results
    if indices_used_for_recon:
        print("-" * 30); print("7. Displaying Results (using parallel reconstruction)...")
        display_images(original_image_data, all_numerical_shares, reconstructed_image_p, k, n_orig, indices_used_for_recon)
        # Print final timing summary
        print("\n--- Timing Summary ---")
        print(f"Parallel Encryption:   {time_p_enc:.4f} sec")
        if run_sequential_enc: print(f"Sequential Encryption: {time_s_enc:.4f} sec")
        print(f"Parallel Decryption:   {time_p_dec:.4f} sec")
        if run_sequential_dec: print(f"Sequential Decryption: {time_s_dec:.4f} sec")
        if run_sequential_enc and time_s_enc > 0: print(f" -> Encryption Speedup: {time_s_enc / time_p_enc:.2f}x")
        if run_sequential_dec and time_s_dec > 0: print(f" -> Decryption Speedup: {time_s_dec / time_p_dec:.2f}x")
    else: print("\nExited without attempting reconstruction.")
