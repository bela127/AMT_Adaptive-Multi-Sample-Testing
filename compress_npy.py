import os
import argparse
import numpy as np

def determine_best_dtype(arr):
    """
    Analyzes the array and returns the smallest safe numpy dtype,
    or None if it's already optimized or contains floating-point decimals.
    """
    original_dtype = arr.dtype
    
    # If it's already a boolean, no optimization needed
    if original_dtype == np.bool_:
        return None
        
    # Check if the array has fractional parts (is it actually a float?)
    # For integer dtypes, this is always true. For float dtypes, we check if all values are whole numbers.
    if np.issubdtype(original_dtype, np.floating):
        if not np.all(np.mod(arr, 1) == 0):
            return None # Contains actual decimals, cannot downcast to int/bool safely
            
    # Find the data range
    try:
        min_val = np.min(arr)
        max_val = np.max(arr)
    except Exception:
        return None # Fallback for empty or unreadable arrays

    # Check for boolean compatibility (strictly 0 and 1)
    if min_val == 0 and max_val == 1:
        return np.bool_ if original_dtype != np.bool_ else None

    # Determine smallest integer type
    # NumPy integer bounds:
    # int8:   -128 to 127
    # int16:  -32,768 to 32,767
    # int32:  -2,147,483,648 to 2,147,483,647
    # int64:  Beyond int32
    
    target_dtype = None
    if -128 <= min_val <= 127 and -128 <= max_val <= 127:
        target_dtype = np.int8
    elif -32768 <= min_val <= 32767 and -32768 <= max_val <= 32767:
        target_dtype = np.int16
    elif -2147483648 <= min_val <= 2147483647 and -2147483648 <= max_val <= 2147483647:
        target_dtype = np.int32
    else:
        target_dtype = np.int64

    # If the target dtype is the same size or larger than the original, skip it
    if target_dtype and np.dtype(target_dtype).itemsize < np.dtype(original_dtype).itemsize:
        return target_dtype
        
    return None

def inspect_and_compress_folder(folder_path, dry_run=True, recursive=True):
    print(f"Scanning folder: {folder_path}")
    print(f"Recursion:     {'Enabled' if recursive else 'Disabled (Top-level only)'}")
    print(f"Mode:          {'[DRY RUN] Previewing changes only' if dry_run else '[LIVE MODE] Modifying files'}\n")
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
                # Use mmap_mode to check structure instantly without full memory footprint
                arr_meta = np.load(file_path, mmap_mode='r')
                original_dtype = arr_meta.dtype
                
                target_dtype = determine_best_dtype(arr_meta)
                
                if target_dtype is None:
                    print(f"[-] {display_path}: Already optimized or contains continuous float data ({original_dtype}). Skipping.")
                    continue
                
                original_size = os.path.getsize(file_path)
                
                # Calculate new size based on the target itemsize
                new_itemsize = np.dtype(target_dtype).itemsize
                estimated_new_size = (arr_meta.nbytes / arr_meta.itemsize) * new_itemsize
                
                saved_bytes = original_size - estimated_new_size
                total_saved_bytes += max(0, saved_bytes) # Guard against edge cases where header sizes vary
                
                print(f"[+] {display_path}: Found optimization. {original_dtype} -> {target_dtype.__name__}")
                print(f"    Size: {original_size / (1024**2):.2f} MB -> {estimated_new_size / (1024**2):.2f} MB")
                
                if not dry_run:
                    actual_data = np.load(file_path)
                    converted_data = actual_data.astype(target_dtype)
                    np.save(file_path, converted_data)
                    print(f"    [SUCCESS] Converted to {target_dtype.__name__}.")
                    
            except Exception as e:
                print(f"[ERROR] Could not process {display_path}: {e}")

    print("\n" + "="*50)
    print(f"Scan complete. Processed {file_count} .npy file(s).")
    print(f"Total estimated storage savings: {total_saved_bytes / (1024**3):.2f} GB")
    if dry_run and total_saved_bytes > 0:
        print("Run again with '--run' to apply these changes permanently.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downcast redundant float/int numpy arrays to smaller int variants or boolean.")
    parser.add_argument("folder", type=str, help="Path to the directory containing .npy files")
    parser.add_argument("--run", action="store_false", dest="dry_run", help="Deactivate dry-run mode and rewrite files")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not look into subdirectories")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        inspect_and_compress_folder(args.folder, dry_run=args.dry_run, recursive=args.recursive)
    else:
        print(f"Error: {args.folder} is not a valid directory.")