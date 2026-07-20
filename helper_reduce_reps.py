import os
import numpy as np
from amt.configuration import Config  # Adjust import path if needed

def truncate_repetitions_to_2500(target_dir):
    if not os.path.isdir(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    all_files = [f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('reject_')]
    print(f"Found {len(all_files)} .npy files to check in '{target_dir}'.\n")

    for file_name in all_files:
        old_path = os.path.join(target_dir, file_name)
        
        # Strip '.npy' to allow clean int parsing for reps inside load_from_name
        clean_name_for_config = file_name[:-4] 
        
        conf = Config()
        try:
            conf.load_from_name(clean_name_for_config)
        except Exception as e:
            print(f"Skipping {file_name}: Failed to parse filename config. ({e})")
            continue

        if conf.reps == 5000:
            print(f"Processing: {file_name}")
            
            try:
                raw_data = np.load(old_path, allow_pickle=True)
                
                # Check the second dimension (axis=1) for the 5000 repetitions
                if raw_data.ndim < 2 or raw_data.shape[1] != 5000:
                    print(f"  -> Warning: Filename specifies 5000 reps, but array axis-1 has shape {raw_data.shape[1] if raw_data.ndim > 1 else 'N/A'}. Skipping.")
                    continue
                
                # Slice the first 2500 entries along the repetition axis (axis=1)
                truncated_data = raw_data[:, :2500, ...]

                # Create the new configuration and clone it
                new_conf = conf.clone()
                new_conf.reps = 2500
                
                prefix = f"reject_{new_conf.get_test_name()}"
                
                # Maintain prefix string consistency for custom names (e.g., ebp.mixture.infinite)
                orig_mode = file_name.split("test.mode-")[1].split("_sel.mode-")[0]
                curr_mode = prefix.split("test.mode-")[1].split("_sel.mode-")[0]
                
                if orig_mode != curr_mode:
                    prefix = prefix.replace(f"test.mode-{curr_mode}", f"test.mode-{orig_mode}")
                
                new_file_name = f"{prefix}.npy"
                new_path = os.path.join(target_dir, new_file_name)
                
                # Save the truncated data and clean up the old file
                np.save(new_path, truncated_data)
                print(f"  -> Successfully saved as: {new_file_name}")
                
            except Exception as e:
                print(f"  -> Error handling data file processing for {file_name}: {e}")
        else:
            print(f"Skipping: {file_name} (Reps already = {conf.reps})")

if __name__ == "__main__":
    TARGET_FOLDER = "./exp_results/test_res/p_and_p_diff"
    truncate_repetitions_to_2500(TARGET_FOLDER)