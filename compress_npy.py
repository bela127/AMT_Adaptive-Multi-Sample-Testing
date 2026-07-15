import os
import argparse
import numpy as np

def determine_best_dtype(arr, sample=False):
    """
    Analyzes the array and returns the smallest safe numpy dtype.
    Optimized to use fast contiguous block-reads (Head & Tail) for dry runs.
    """
    original_dtype = arr.dtype
    if original_dtype == np.bool_:
        return None

    # --- BLOCK SAMPLING FOR ULTRA-FAST DRY RUNS ---
    if sample:
        # Flatten the view to simplify slicing multi-dimensional structures
        flat_view = arr.ravel()
        total_elements = flat_view.size

        if total_elements <= 2000:
            # Array is tiny, just pull the whole thing in one shot
            test_arr = flat_view[:]
        else:
            # Pull the first 1,000 elements and the last 1,000 elements.
            # Because these are contiguous blocks, the OS and network layer 
            # can stream them instantly in just 2 quick I/O operations.
            head = flat_view[:1000]
            tail = flat_view[-1000:]
            test_arr = np.concatenate([head, tail])
    else:
        # Full, exhaustive check for live mode (runs completely in RAM)
        test_arr = arr

    # 1. Quick min/max check on the block sample
    try:
        min_val = np.min(test_arr)
        max_val = np.max(test_arr)
    except Exception:
        return None

    # Check for boolean compatibility
    if min_val == 0 and max_val == 1:
        if np.issubdtype(original_dtype, np.floating):
            if not np.all((test_arr == 0) | (test_arr == 1)):
                return None
        return np.bool_ if original_dtype != np.bool_ else None

    # 2. Determine target integer type candidate
    target_dtype = None
    if -128 <= min_val <= 127 and -128 <= max_val <= 127:
        target_dtype = np.int8
    elif -32768 <= min_val <= 32767 and -32768 <= max_val <= 32767:
        target_dtype = np.int16
    elif -2147483648 <= min_val <= 2147483647 and -2147483648 <= max_val <= 2147483647:
        target_dtype = np.int32
    else:
        target_dtype = np.int64

    # FAIL FAST: Skip heavy math if we aren't saving space
    if target_dtype is None or np.dtype(target_dtype).itemsize >= np.dtype(original_dtype).itemsize:
        return None
        
    # 3. Decimal check for floats
    if np.issubdtype(original_dtype, np.floating):
        if not np.all(np.mod(test_arr, 1) == 0):
            return None 

    return target_dtype

def inspect_and_compress_folder(folder_path, dry_run=True, recursive=True):
    print(f"Scanning folder: {folder_path}")
    print(f"Recursion:     {'Enabled' if recursive else 'Disabled (Top-level only)'}")
    print(f"Mode:          {'[DRY RUN] Fast Preview (Sampling Enabled)' if dry_run else '[LIVE MODE] Exhaustive verification and modification'}\n")
    input("Press Enter to continue or Ctrl+C to abort...")
    
    total_saved_bytes = 0
    file_count = 0

    if recursive:
        iterator = os.walk(folder_path)
    else:
        try:
            iterator = [(folder_path, [], os.listdir(folder_path))]
        except Exception as e:
            print(f"[ERROR] Could not read directory: {e}")
            return

    for root, _, files in iterator:
        for file in files:
            if not file.lower().endswith('.npy'):
                continue
                
            file_path = os.path.join(root, file)
            file_count += 1
            display_path = os.path.relpath(file_path, folder_path)
            
            try:
                # Open via mmap_mode to keep the initial footprint at 0 bytes
                arr_meta = np.load(file_path, mmap_mode='r')
                original_dtype = arr_meta.dtype
                
                # --- STRATEGY SPLIT ---
                if dry_run:
                    # Pass the memory map directly, but tell it to sample.
                    # This will execute almost instantly over network/slow storage.
                    target_dtype = determine_best_dtype(arr_meta, sample=True)
                else:
                    # Live Mode: Load fully into RAM *locally* for complete safety verification
                    print(f"[*] Reading {display_path} completely into memory for verification...")
                    full_arr = np.load(file_path)
                    target_dtype = determine_best_dtype(full_arr, sample=False)
                
                if target_dtype is None:
                    if dry_run:
                        print(f"[-] {display_path}: Likely already optimized or contains continuous float data. Skipping.")
                    else:
                        print(f"[-] {display_path}: Verified un-compressible. Skipping.")
                    continue
                
                original_size = os.path.getsize(file_path)
                new_itemsize = np.dtype(target_dtype).itemsize
                estimated_new_size = (arr_meta.nbytes / arr_meta.itemsize) * new_itemsize
                saved_bytes = original_size - estimated_new_size
                total_saved_bytes += max(0, saved_bytes)
                
                print(f"[+] {display_path}: Optimization found. {original_dtype} -> {target_dtype.__name__}")
                print(f"    Size: {original_size / (1024**2):.2f} MB -> {estimated_new_size / (1024**2):.2f} MB")
                
                if not dry_run:
                    # full_arr is already loaded and verified safely in RAM, convert it and save
                    converted_data = full_arr.astype(target_dtype)
                    np.save(file_path, converted_data)
                    print(f"    [SUCCESS] Written safely to disk.")
                    
            except Exception as e:
                print(f"[ERROR] Could not process {display_path}: {e}")

    print("\n" + "="*50)
    print(f"Scan complete. Processed {file_count} .npy file(s).")
    print(f"Total {'estimated' if dry_run else 'actual'} storage savings: {total_saved_bytes / (1024**3):.2f} GB")
    if dry_run and total_saved_bytes > 0:
        print("Run again with '--run' to safely apply changes.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downcast redundant float/int numpy arrays using fast sampling previews.")
    parser.add_argument("folder", type=str, help="Path to the directory containing .npy files")
    parser.add_argument("--run", action="store_false", dest="dry_run", help="Deactivate dry-run mode and rewrite files")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not look into subdirectories")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        inspect_and_compress_folder(args.folder, dry_run=args.dry_run, recursive=args.recursive)
    else:
        print(f"Error: {args.folder} is not a valid directory.")