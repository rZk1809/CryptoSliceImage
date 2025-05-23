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
import getpass # For password input

# --- Cryptography Imports ---
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag # Import specific exception

# --- Configuration ---
PRIME = 251
print(f"INFO: Using PRIME = {PRIME}. Pixel values > {PRIME-1} will be reduced modulo {PRIME}.")
OUTPUT_DIR = "image_shares_final_corrected" # New dir name
ORIGINAL_HASH_FILE = os.path.join(OUTPUT_DIR, "original_hash.txt")
SHARE_HASH_FILE_TEMPLATE = os.path.join(OUTPUT_DIR, "share_{}_hash.txt")
SALT_SIZE = 16
AES_KEY_SIZE = 32 # 256-bit key
ITERATIONS = 100000 # PBKDF2 iterations

try:
    NUM_WORKERS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_WORKERS = 4
print(f"INFO: Using {NUM_WORKERS} workers for parallel processing.")

# --- Notes ---
# VC Note: Uses SSS on pixel values (computational recon), not traditional stack-to-reveal VC.
# Max Flow Note: Network modeling concept, not implemented here.

# --- SSS Core Functions ---
def extended_gcd(a, b):
    if a == 0: return (b, 0, 1)
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1; y = x1
    return (gcd, x, y)

def modInverse(a, m):
    a = a % m;
    if a < 0: a += m
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1: raise ValueError(f'Mod inverse does not exist for {a} mod {m}')
    return (x % m + m) % m

def generate_polynomial(degree, secret):
    secret = int(secret) % PRIME
    coeffs = [secret] + [random.randint(0, PRIME - 1) for _ in range(degree)]
    return [c % PRIME for c in coeffs]

def evaluate_polynomial(coeffs, x):
    result = 0; power_of_x = 1; x_mod = x % PRIME
    for coeff in coeffs:
        term = (coeff * power_of_x) % PRIME; result = (result + term) % PRIME
        power_of_x = (power_of_x * x_mod) % PRIME
    return result

# --- Lagrange Interpolation ---
def lagrange_interpolate(points, x_target=0):
    if not points: raise ValueError("Need points for interpolation")
    k = len(points); xs, ys = zip(*points)
    if len(set(xs)) != k: raise ValueError("X values must be distinct")
    secret = 0; x_target_mod = x_target % PRIME
    for i in range(k):
        num, den = 1, 1; xi = xs[i] % PRIME
        for j in range(k):
            if i == j: continue
            xj = xs[j] % PRIME; num = (num * (x_target_mod - xj)) % PRIME; den = (den * (xi - xj)) % PRIME
        try: inv_den = modInverse(den, PRIME); term = (ys[i] * num * inv_den) % PRIME; secret = (secret + term) % PRIME
        except ValueError as e: print(f"ERR Lagrange: {e}. Pts:{points}, i:{i}, den:{den}"); raise e
    return (secret + PRIME) % PRIME

# --- Linear Algebra for N+1 Recon ---
def precompute_power_sums(n_orig, k_degree, p):
    power_sums = {}; current_powers = [1] * n_orig
    for j in range(k_degree + 1):
        current_sum = sum(current_powers) % p; power_sums[j] = current_sum
        if j < k_degree:
            for i in range(1, n_orig + 1): current_powers[i-1] = (current_powers[i-1] * i) % p
    return power_sums

def solve_linear_system_mod_p(matrix_A, vector_b, p):
    n = matrix_A.shape[0]; det_A = int(round(np.linalg.det(matrix_A))) % p
    if det_A == 0: raise np.linalg.LinAlgError("Matrix singular mod p")
    try: inv_det_A = modInverse(det_A, p)
    except ValueError: raise np.linalg.LinAlgError(f"Det {det_A} no inverse mod {p}")
    adj_A = np.zeros_like(matrix_A, dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix_A, i, axis=0), j, axis=1)
            cofactor = ((-1)**(i + j)) * int(round(np.linalg.det(minor)))
            adj_A[j, i] = cofactor % p
    inv_A = (inv_det_A * adj_A) % p; solution_x = (inv_A @ vector_b.astype(int)) % p
    return solution_x

# --- Image Preprocessing ---
def preprocess_image(image_path):
    try: img = Image.open(image_path).convert('L'); return np.array(img, dtype=np.uint8)
    except Exception as e: print(f"ERR processing image {image_path}: {e}"); return None

# --- Parallel Share Creation Worker ---
def _create_shares_chunk_processor(args):
    img_chunk, k, n, p, start_row, height, width = args
    chunk_height = img_chunk.shape[0]
    share_chunks = [np.zeros((chunk_height, width), dtype=np.int32) for _ in range(n)]
    for r_local in range(chunk_height):
        for c in range(width):
            secret = int(img_chunk[r_local, c]) % p; coeffs = generate_polynomial(k - 1, secret)
            for i in range(n):
                share_x = i + 1; val = evaluate_polynomial(coeffs, share_x)
                share_chunks[i][r_local, c] = val
    return share_chunks

# --- Parallel Share Creation Manager ---
def create_shares_parallel(image_data, k, n):
    height, width = image_data.shape
    shares_data = [np.zeros_like(image_data, dtype=np.int32) for _ in range(n)]
    print(f"Creating {n} shares (k={k}) PARALLEL ({NUM_WORKERS} workers)...")
    chunk_size = max(1, height // NUM_WORKERS); tasks = []
    for i in range(0, height, chunk_size):
        start_row = i; end_row = min(i + chunk_size, height)
        if start_row == end_row: continue
        img_chunk = image_data[start_row:end_row, :]; tasks.append((img_chunk, k, n, PRIME, start_row, height, width))
    results_temp = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures_map = {executor.submit(_create_shares_chunk_processor, task): task[4] for task in tasks}
        for future in concurrent.futures.as_completed(futures_map):
            start_row = futures_map[future]
            try: chunk_result = future.result(); results_temp.append((start_row, chunk_result)); # print(f"  Worker finished create chunk @{start_row}")
            except Exception as e: print(f"ERR create worker @{start_row}: {e}"); raise e
    results_temp.sort(key=lambda x: x[0])
    print("Assembling created shares...")
    for start_row, chunk_result_list in results_temp:
        if not chunk_result_list: continue
        chunk_height = chunk_result_list[0].shape[0]; end_row = start_row + chunk_height
        if chunk_height == 0 or start_row < 0 or end_row > height: print(f"ERR: Invalid slice [{start_row}:{end_row}]"); raise IndexError("Slice calc error")
        for share_idx in range(n):
            source_chunk = chunk_result_list[share_idx]; target_slice = shares_data[share_idx][start_row:end_row, :]
            try: target_slice[:] = source_chunk
            except ValueError as e: print(f"BROADCAST ERR: Share {share_idx}, rows [{start_row}:{end_row}], Target {target_slice.shape}, Source {source_chunk.shape}, Err: {e}"); raise e
    print("Parallel share creation complete.")
    return shares_data

# --- Sequential Share Creation ---
def create_shares_sequential(image_data, k, n):
    height, width = image_data.shape
    shares_data = [np.zeros_like(image_data, dtype=np.int32) for _ in range(n)]
    print(f"Creating {n} shares (k={k}) SEQUENTIALLY...")
    for r in range(height):
        for c in range(width):
            secret = int(image_data[r, c]) % PRIME; coeffs = generate_polynomial(k - 1, secret)
            for i in range(n):
                share_x = i + 1; val = evaluate_polynomial(coeffs, share_x)
                shares_data[i][r, c] = val
    print("Sequential share creation complete.")
    return shares_data

# --- Parallel Reconstruction Worker ---
def _reconstruct_chunk_processor(args):
    share_chunks, share_indices, k, n_orig, p, start_row, end_row, width, n_plus_1_present, power_sums = args
    chunk_height = end_row - start_row; reconstructed_chunk = np.zeros((chunk_height, width), dtype=np.int32)
    n_plus_1_index = n_orig + 1
    for r_local in range(chunk_height):
        for c in range(width):
            points = []; matrix_rows = []; vector_b = []
            for i in range(k):
                idx = share_indices[i]; val = int(share_chunks[i][r_local, c])
                if n_plus_1_present:
                    if idx == n_plus_1_index: row = [power_sums.get(j, 0) for j in range(k)]; matrix_rows.append(row); vector_b.append(val)
                    else: row = [(idx**j) % p for j in range(k)]; matrix_rows.append(row); vector_b.append(val)
                else: points.append((idx, val))
            try:
                if n_plus_1_present: matrix_M = np.array(matrix_rows,dtype=int); vector_Y = np.array(vector_b,dtype=int); solution_A = solve_linear_system_mod_p(matrix_M, vector_Y, p); recon_val = solution_A[0]
                else: recon_val = lagrange_interpolate(points, x_target=0)
                reconstructed_chunk[r_local, c] = recon_val
            except (ValueError, np.linalg.LinAlgError): reconstructed_chunk[r_local, c] = 128
    return (start_row, reconstructed_chunk)

# --- Parallel Reconstruction Manager ---
def reconstruct_image_parallel(shares_to_reconstruct, share_indices, k, n_orig, p):
    if len(shares_to_reconstruct) != k or len(share_indices) != k: raise ValueError(f"Need k={k} shares. Got {len(shares_to_reconstruct)}.")
    height, width = shares_to_reconstruct[0].shape; reconstructed_data = np.zeros((height, width), dtype=np.int32)
    n_plus_1_index = n_orig + 1; n_plus_1_present = n_plus_1_index in share_indices
    print(f"Reconstructing PARALLEL (k={k}, n={n_orig+1}). N+1 Present: {n_plus_1_present}")

    # *** CORRECTED power_sums block ***
    power_sums = None
    if n_plus_1_present:
        print(" -> Lin sys method.")
        try:
            power_sums = precompute_power_sums(n_orig, k - 1, p)
        except Exception as e:
            print(f"ERROR: Failed to precompute power sums: {e}")
            raise RuntimeError(f"Power sum precomputation failed") from e
    else:
        print(" -> Lagrange method.")
    # *** END CORRECTED block ***

    chunk_size = max(1, height // NUM_WORKERS); tasks = []
    for i in range(0, height, chunk_size):
        start_row = i; end_row = min(i + chunk_size, height)
        if start_row == end_row: continue
        share_chunks_for_task = [share[start_row:end_row, :] for share in shares_to_reconstruct]
        tasks.append((share_chunks_for_task, share_indices, k, n_orig, p, start_row, end_row, width, n_plus_1_present, power_sums))
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures_map = {executor.submit(_reconstruct_chunk_processor, task): task[5] for task in tasks}
        for future in concurrent.futures.as_completed(futures_map):
            start_row = futures_map[future]
            try: s_row, recon_chunk = future.result(); results[s_row] = recon_chunk; # print(f"  Worker finished recon chunk @{s_row}")
            except Exception as e: print(f"ERR recon worker @{start_row}: {e}"); raise e
    print("Assembling reconstructed image...")
    for start_row, recon_chunk in sorted(results.items()): end_row = start_row + recon_chunk.shape[0]; reconstructed_data[start_row:end_row, :] = recon_chunk
    print("Parallel image reconstruction complete.")
    reconstructed_image = np.clip(reconstructed_data, 0, 255).astype(np.uint8)
    return reconstructed_image

# --- Sequential Reconstruction ---
def reconstruct_image_sequential(shares_to_reconstruct, share_indices, k, n_orig, p):
    if len(shares_to_reconstruct) != k or len(share_indices) != k: raise ValueError(f"Need k={k} shares. Got {len(shares_to_reconstruct)}.")
    height, width = shares_to_reconstruct[0].shape; reconstructed_data = np.zeros((height, width), dtype=np.int32)
    n_plus_1_index = n_orig + 1; n_plus_1_present = n_plus_1_index in share_indices
    print(f"Reconstructing SEQUENTIAL (k={k}, n={n_orig+1}). N+1 Present: {n_plus_1_present}")

    # *** CORRECTED power_sums block ***
    power_sums = None
    if n_plus_1_present:
        print(" -> Lin sys method.")
        try:
            power_sums = precompute_power_sums(n_orig, k - 1, p)
        except Exception as e:
            print(f"ERROR: Failed to precompute power sums: {e}")
            raise RuntimeError(f"Power sum precomputation failed") from e
    else:
        print(" -> Lagrange method.")
    # *** END CORRECTED block ***

    print("Processing pixels sequentially...")
    for r in range(height):
        for c in range(width):
            points = []; matrix_rows = []; vector_b = []
            for i in range(k):
                idx = share_indices[i]; val = int(shares_to_reconstruct[i][r, c])
                if n_plus_1_present:
                    if idx == n_plus_1_index: row = [power_sums.get(j, 0) for j in range(k)]; matrix_rows.append(row); vector_b.append(val)
                    else: row = [(idx**j) % p for j in range(k)]; matrix_rows.append(row); vector_b.append(val)
                else: points.append((idx, val))
            try:
                if n_plus_1_present: matrix_M = np.array(matrix_rows,dtype=int); vector_Y = np.array(vector_b,dtype=int); solution_A = solve_linear_system_mod_p(matrix_M, vector_Y, p); recon_val = solution_A[0]
                else: recon_val = lagrange_interpolate(points, x_target=0)
                reconstructed_data[r, c] = recon_val
            except (ValueError, np.linalg.LinAlgError): reconstructed_data[r, c] = 128
    print("Sequential image reconstruction complete.")
    reconstructed_image = np.clip(reconstructed_data, 0, 255).astype(np.uint8)
    return reconstructed_image

# --- Create N+1 Share ---
def create_nplus1_share(numerical_shares_n, p):
    n = len(numerical_shares_n); share_nplus1 = np.zeros_like(numerical_shares_n[0], dtype=np.int32)
    for share in numerical_shares_n: share_nplus1 = (share_nplus1 + share) % p
    return share_nplus1

# --- Get Visual Share ---
def get_visual_share(numerical_share):
     return (numerical_share % 256).astype(np.uint8)

# --- Hashing for Integrity ---
def calculate_hash(data):
    if isinstance(data, np.ndarray): data_bytes = data.tobytes()
    elif isinstance(data, bytes): data_bytes = data
    else: raise TypeError("Data must be numpy array or bytes for hashing")
    return hashlib.sha256(data_bytes).hexdigest()

# --- Encryption / Decryption Helpers ---
def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=AES_KEY_SIZE, salt=salt, iterations=ITERATIONS, backend=default_backend())
    return kdf.derive(password.encode('utf-8'))

def encrypt_data(plaintext_bytes: bytes, password: str) -> bytes:
    salt = os.urandom(SALT_SIZE); key = derive_key(password, salt)
    aesgcm = AESGCM(key); nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
    return salt + nonce + ciphertext

def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
    try:
        salt = encrypted_data[:SALT_SIZE]; nonce = encrypted_data[SALT_SIZE:SALT_SIZE+12]
        ciphertext = encrypted_data[SALT_SIZE+12:]; key = derive_key(password, salt)
        aesgcm = AESGCM(key); plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext_bytes
    except InvalidTag: raise ValueError("Decryption failed: Authentication error (likely wrong password or corrupt data)")
    except Exception as e: raise ValueError(f"Decryption failed: {e}") from e

# --- Saving and Loading (with Encryption) ---
_session_password = None

def save_shares_and_hashes(all_numerical_shares, base_filename):
    global _session_password
    num_total_shares = len(all_numerical_shares)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR); print(f"Created dir: {OUTPUT_DIR}")
    if _session_password is None:
         print("\n--- Encryption ---")
         _session_password = getpass.getpass("Set password to ENCRYPT numerical shares: ")
         password_confirm = getpass.getpass("Confirm password: ")
         if _session_password != password_confirm: print("Passwords do not match. Aborting."); _session_password = None; raise ValueError("Password mismatch")
         print("Password set for saving.")
    share_hashes = []; print("Saving shares (Numerical data ENCRYPTED)...")
    for i, numerical_share in enumerate(all_numerical_shares):
        share_index = i + 1; visual_share = get_visual_share(numerical_share)
        hash_file = SHARE_HASH_FILE_TEMPLATE.format(share_index) # Define hash_file path early
        try: # Hash PLAINTEXT
            share_hash = calculate_hash(numerical_share); share_hashes.append(share_hash)
            with open(hash_file, 'w') as f: f.write(share_hash)
        except Exception as e: print(f"ERR saving hash {share_index}: {e}"); continue
        try: # Encrypt and Save numerical
            share_filename_enc = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{share_index}_numerical.enc")
            encrypted_bundle = encrypt_data(numerical_share.tobytes(), _session_password)
            with open(share_filename_enc, 'wb') as f: f.write(encrypted_bundle)
        # *** CORRECTED Exception Block ***
        except Exception as e:
            print(f"ERR encrypt/save share {share_index}: {e}")
            # Clean up hash file if encrypted save failed?
            try:
                os.remove(hash_file)
                print(f"  Removed potentially incomplete hash file: {hash_file}")
            except OSError:
                pass # Ignore error if hash file wasn't created or couldn't be removed
            # Decide whether to continue to the next share or stop everything
            continue # Skip saving this share and move to the next
        # *** END CORRECTED Block ***
        try: # Save visual
            share_filename_png = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{share_index}_visual.png")
            Image.fromarray(visual_share).save(share_filename_png)
        except Exception as e: print(f"WARN saving visual share {share_index}: {e}")
    print(f"Saved {num_total_shares} shares (Encrypted Num + Visual) and Hashes.")
    return share_hashes

def load_numerical_shares(base_filename, indices_to_load):
    global _session_password
    loaded_shares = []; loaded_indices = []; loaded_hashes = []; valid_shares = True
    if _session_password is None:
         print("\n--- Decryption ---")
         _session_password = getpass.getpass("Enter password to DECRYPT shares: ")
    print(f"Loading/Decrypting shares: {indices_to_load}")
    for i in indices_to_load:
        share_filename_enc = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{i}_numerical.enc")
        hash_file = SHARE_HASH_FILE_TEMPLATE.format(i); numerical_share_plaintext = None
        try: # Decrypt
            with open(share_filename_enc, 'rb') as f: encrypted_bundle = f.read()
            decrypted_bytes = decrypt_data(encrypted_bundle, _session_password)
            try: # Infer shape/dtype and reshape
                 visual_share_path = os.path.join(OUTPUT_DIR, f"{base_filename}_share_{i}_visual.png")
                 vis_img = Image.open(visual_share_path); expected_shape = (vis_img.height, vis_img.width)
                 expected_dtype = np.int32; num_elements = expected_shape[0] * expected_shape[1]; bytes_per_element = np.dtype(expected_dtype).itemsize
                 if len(decrypted_bytes) != num_elements * bytes_per_element: raise ValueError(f"Byte count mismatch {len(decrypted_bytes)} vs {num_elements * bytes_per_element}")
                 numerical_share_plaintext = np.frombuffer(decrypted_bytes, dtype=expected_dtype).reshape(expected_shape)
            except FileNotFoundError: print(f"ERR: Visual share {i} missing, cannot infer shape."); raise
            except ValueError as shape_err: print(f"ERR reshaping share {i}: {shape_err}"); raise
            with open(hash_file, 'r') as f: expected_hash = f.read().strip() # Verify Hash
            actual_hash = calculate_hash(numerical_share_plaintext)
            if actual_hash == expected_hash:
                loaded_shares.append(numerical_share_plaintext); loaded_indices.append(i); loaded_hashes.append(actual_hash)
                print(f" -> Decrypted and verified share {i}")
            else: print(f"ERR: Hash mismatch for DECRYPTED share {i}!"); valid_shares = False
        except FileNotFoundError: print(f"ERR: Share/Hash file missing for index {i}."); valid_shares = False
        except (ValueError, TypeError, InvalidTag) as e: print(f"ERR processing share {i}: {e}"); valid_shares = False; _session_password = None
        except Exception as e: print(f"ERR loading share {i}: {e}"); valid_shares = False
    if len(loaded_shares) != len(indices_to_load): print("ERR: Failed load/decrypt/verify one or more shares."); valid_shares = False
    return loaded_shares, loaded_indices, loaded_hashes, valid_shares

# --- Display ---
def display_images(original, all_shares_numerical_plain, reconstructed, k, n_orig, indices_used):
    num_total_shares = n_orig + 1; shares_to_display_indices = []
    if 1 <= num_total_shares: shares_to_display_indices.append(1)
    if n_orig + 1 <= num_total_shares: shares_to_display_indices.append(n_orig + 1)
    num_shares_display = len(shares_to_display_indices); num_cols = 2 + num_shares_display
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))
    fig.suptitle(f"Shamir ({k},{n_orig})->({k},{n_orig+1}) | Used: {indices_used} | Encrypted")
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255); axes[0].set_title("Original"); axes[0].axis('off')
    share_plot_index = 1
    for idx_to_show in shares_to_display_indices:
         visual_share_img = get_visual_share(all_shares_numerical_plain[idx_to_show - 1])
         axes[share_plot_index].imshow(visual_share_img, cmap='gray', vmin=0, vmax=255)
         title = f"Share {idx_to_show}" + (" (Comb.)" if idx_to_show == n_orig + 1 else "")
         axes[share_plot_index].set_title(title); axes[share_plot_index].axis('off'); share_plot_index += 1
    if reconstructed is not None: axes[num_cols - 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255); axes[num_cols - 1].set_title(f"Reconstructed\n(from {k} shares)"); axes[num_cols - 1].axis('off')
    else: axes[num_cols - 1].set_title("Reconstruction\nFailed"); axes[num_cols - 1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    _session_password = None # Reset password

    # --- Parameters Input ---
    image_path = input("Enter path to image file: ")
    if not os.path.exists(image_path): print(f"ERR: File '{image_path}' not found."); exit()
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    # *** CORRECTED Input Loop for n_orig ***
    while True:
        try:
            n_orig = int(input("Enter initial number of shares (n >= 2): "))
            if n_orig >= 2: break
            else: print("n must be >= 2.")
        except ValueError: print("Invalid integer.")
    # *** CORRECTED Input Loop for k ***
    while True:
        try:
            k = int(input(f"Enter threshold (k, 2 <= k <= {n_orig}): "))
            if 2 <= k <= n_orig: break
            else: print(f"k must be between 2 and {n_orig}.")
        except ValueError: print("Invalid integer.")

    # 1. Preprocessing
    print("-" * 30); print("1. Preprocessing Image...")
    original_image_data = preprocess_image(image_path)
    if original_image_data is None: exit()
    height, width = original_image_data.shape; print(f"Image loaded ({width}x{height}).")
    original_data_for_hash = original_image_data.astype(np.int32) % PRIME; original_hash = calculate_hash(original_data_for_hash)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(ORIGINAL_HASH_FILE, 'w') as f: f.write(original_hash); print(f"Original hash (mod {PRIME}) saved.")

    # --- 2/3. Share Creation ---
    print("-" * 30); print(f"2/3. Creating {n_orig + 1} Shares (k={k})...")
    start_time_p_enc = time.time()
    shares_numerical_n_p = create_shares_parallel(original_image_data, k, n_orig)
    share_nplus1_p = create_nplus1_share(shares_numerical_n_p, PRIME)
    end_time_p_enc = time.time(); time_p_enc = end_time_p_enc - start_time_p_enc
    all_numerical_shares_plain_p = shares_numerical_n_p + [share_nplus1_p]
    print(f"Parallel Share Generation Time: {time_p_enc:.4f} sec")
    # Optional Sequential Run
    run_sequential_enc = input("Run sequential generation for comparison? (y/n): ").lower() == 'y'
    time_s_enc = -1
    if run_sequential_enc: start_time_s_enc = time.time(); create_shares_sequential(original_image_data, k, n_orig); end_time_s_enc = time.time(); time_s_enc = end_time_s_enc - start_time_s_enc; print(f"Sequential Gen Time: {time_s_enc:.4f} sec")

    # 4. Saving Shares (WITH ENCRYPTION)
    print("-" * 30); print("4. Saving Shares (Encrypting Numerical)...")
    try: save_shares_and_hashes(all_numerical_shares_plain_p, base_filename)
    except ValueError as e: print(f"Save failed: {e}"); exit()

    # 5. Reconstruction (Decryption Phase)
    print("-" * 30); print(f"5. Simulating Reconstruction (k={k})...")
    reconstructed_image_p = None; reconstructed_image_s = None
    indices_used_for_recon = []; time_p_dec = -1; time_s_dec = -1

    while True: # Recon attempt loop
        print(f"\nAvailable shares: 1 to {n_orig + 1}")
        indices_str = input(f"Enter {k} unique share indices (e.g., '1 3 {n_orig+1}') or 'q' to quit: ")
        if indices_str.lower() == 'q': break
        try: # Parse indices
            indices_to_reconstruct = sorted(list(set([int(idx) for idx in indices_str.split()])))
            if len(indices_to_reconstruct) != k: print(f"ERR: Need {k} indices."); continue
            if not all(1 <= idx <= (n_orig + 1) for idx in indices_to_reconstruct): print(f"ERR: Indices out of range."); continue
            indices_used_for_recon = indices_to_reconstruct
            try: # Load/Decrypt & Reconstruct
                 print(f"\nAttempting load/decrypt for shares: {indices_to_reconstruct}")
                 shares_for_recon, idx_loaded, _, shares_valid = load_numerical_shares(base_filename, indices_to_reconstruct)
                 if not shares_valid or len(shares_for_recon) != k: print("Recon aborted: Invalid/missing/decryption failed shares."); continue
                 # Parallel Recon
                 start_time_p_dec = time.time(); reconstructed_image_p = reconstruct_image_parallel(shares_for_recon, idx_loaded, k, n_orig, PRIME); end_time_p_dec = time.time(); time_p_dec = end_time_p_dec - start_time_p_dec; print(f"Parallel Decrypt+Recon Time: {time_p_dec:.4f} sec")
                 # Optional Sequential Recon
                 run_sequential_dec = input("Run sequential decrypt/recon for comparison? (y/n): ").lower() == 'y'
                 if run_sequential_dec: start_time_s_dec = time.time(); reconstructed_image_s = reconstruct_image_sequential(shares_for_recon, idx_loaded, k, n_orig, PRIME); end_time_s_dec = time.time(); time_s_dec = end_time_s_dec - start_time_s_dec; print(f"Sequential Decrypt+Recon Time: {time_s_dec:.4f} sec")
                 # Integrity Check
                 print("-" * 30); print("6. Verifying Integrity...")
                 if reconstructed_image_p is not None:
                     recon_hash = calculate_hash(reconstructed_image_p.astype(np.int32) % PRIME)
                     print(f"Recon hash (mod {PRIME}): {recon_hash[:8]}...")
                     print(f"Orig hash  (mod {PRIME}): {original_hash[:8]}...")
                     if recon_hash == original_hash: print("Integrity Check PASSED.")
                     else: print("Integrity Check FAILED!")
                 else: print("Integrity check skipped (recon failed).")
                 break # Exit recon loop on success
            except (ValueError, TypeError, InvalidTag) as e: print(f"Error during share load/decrypt: {e}"); print("Check password/files. Try again.") # Let user retry password
            except (np.linalg.LinAlgError, RuntimeError) as e: print(f"Error during recon calculation: {e}. Try different shares?") # Math error
            except Exception as e: print(f"Unexpected error in recon phase: {e}") # Other errors
        except ValueError as e: print(f"Invalid index input: {e}") # Index parsing error

    # 7. Display Results
    if indices_used_for_recon and reconstructed_image_p is not None:
        print("-" * 30); print("7. Displaying Results...")
        display_images(original_image_data, all_numerical_shares_plain_p, reconstructed_image_p, k, n_orig, indices_used_for_recon)
        print("\n--- Timing Summary ---")
        print(f"Parallel Generation:   {time_p_enc:.4f} sec")
        if run_sequential_enc: print(f"Sequential Generation: {time_s_enc:.4f} sec")
        print(f"Parallel Decrypt+Recon:{time_p_dec: >8.4f} sec") # Align output
        if run_sequential_dec: print(f"Sequential Decrypt+Recon:{time_s_dec: >8.4f} sec")
        if run_sequential_enc and time_s_enc > 0 and time_p_enc > 0 : print(f" -> Generation Speedup: {time_s_enc / time_p_enc:.2f}x")
        if run_sequential_dec and time_s_dec > 0 and time_p_dec > 0 : print(f" -> Recon Speedup:      {time_s_dec / time_p_dec:.2f}x")
    else: print("\nExited without successful reconstruction.")

    _session_password = None
    print("\nSession password cleared. End of script.")
