import os
import re
import argparse
from pathlib import Path

from amt.configuration import Config

def migrate_filename_directory(target_directory: str, dry_run: bool = True, recursive: bool = True):
    """
    Scans a directory, parses files using the legacy pattern via Config,
    and updates them to the new version (1) configuration naming convention.
    """
    dir_path = Path(target_directory)
    if not dir_path.exists():
        print(f"Error: Directory '{target_directory}' does not exist.")
        return

    print(f"--- Starting Migration Sweep in: {dir_path} ---")
    print(f"    Execution Mode: {'[DRY RUN]' if dry_run else '[LIVE UPDATE]'}")
    print(f"    Scope:          {'Recursive' if recursive else 'Current Directory Only'}\n")
    input("Press Enter to continue or Ctrl+C to abort...")
    
    matched_files = 0
    renamed_files = 0
    skipped_files = 0

    # Universal regex capturing any arbitrary prefix and token sequences containing 'mode-'
    file_pattern = re.compile(r"^([a-zA-Z0-9_\-\.]+?)_(.*mode-.*)(\.npy)$")

    # Select target generator depending on recursive flag
    file_generator = dir_path.rglob("*.npy") if recursive else dir_path.glob("*.npy")

    for file_item in file_generator:
        if not file_item.is_file():
            continue

        match = file_pattern.match(file_item.name)
        if not match:
            print(f"[SKIP] No pattern match for: {file_item.name}")
            skipped_files += 1
            continue

        matched_files += 1
        prefix = match.group(1)          # e.g., 'contingency', 'test', 'reject', 'any_custom_prefix'
        base_config_str = match.group(2) # The raw parameter token sequence
        suffix = match.group(3)          # e.g., '.npy'

        try:
            # 1. Spawn a clean config slate and load keys
            conf = Config()
            conf.load_from_name(f"dummy_{base_config_str}")
            
            # 2. DEFINITIVE CHECK: Look directly at the parameter layout contents
            if "test.mode-" in base_config_str:
                new_base = conf.get_test_name(version=1)
            else:
                new_base = conf.get_sel_name(version=1)

            new_filename = f"{prefix}_{new_base}{suffix}"
            new_file_path = file_item.parent / new_filename

            # 3. Identity optimization triggers perfectly now for all file types
            if file_item.name == new_filename:
                print(f"[SKIP] Already compliant: {file_item.name} in '{file_item.parent.relative_to(dir_path) if recursive else '.'}'")
                skipped_files += 1
                continue

            print(f"[MIGRATE] Match Found in '{file_item.parent.relative_to(dir_path) if recursive else '.'}':")
            print(f"  OLD: {file_item.name}")
            print(f"  NEW: {new_filename}")

            if not dry_run:
                # Safe cross-over protection
                if new_file_path.exists():
                    print(f"  WARNING: Collision destination exists! Skipping: {new_filename}")
                    skipped_files += 1
                    continue
                file_item.rename(new_file_path)
            
            renamed_files += 1

        except Exception as e:
            print(f"[ERROR] Failed parsing matrix identifier: {file_item.name}. Reason: {e}")
            skipped_files += 1

    print("\n--- Migration Execution Summary ---")
    print(f"Total Simulation Targets Found: {matched_files}")
    print(f"Successfully Migrated Files:   {renamed_files}")
    print(f"Skipped / Already Clean Files: {skipped_files}")
    if dry_run and renamed_files > 0:
        print("Note: This was a dry run. Run with --run to write changes to disk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate legacy configuration file naming templates to Version 1 layouts.")
    parser.add_argument("folder", type=str, help="Path to the directory containing .npy files")
    parser.add_argument("--run", action="store_false", dest="dry_run", help="Deactivate dry-run mode and rewrite files")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not look into subdirectories")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        migrate_filename_directory(args.folder, dry_run=args.dry_run, recursive=args.recursive)
    else:
        print(f"Error: {args.folder} is not a valid directory.")