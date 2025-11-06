import os
import torch
import glob
import logging

# --- Configuration ---
# !!! IMPORTANT: Set this to the directory containing your .pt files !!!
PT_FILES_DIRECTORY = "D:/ADGA-MIL/CLAM+Dynamic/Numbers of neighbors/RRT_data/tcga-subtyping/TCGA-NSCLC PLIP/pt_files" 
# Example: PT_FILES_DIRECTORY = "/home/user/data/tcga_features/pt_files"
# --- End Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def modify_pt_files(directory):
    """
    Loads all .pt files in the given directory, removes the leading dimension 
    if it's size 1, and saves the modified tensor back to the original file.

    Args:
        directory (str): The path to the directory containing .pt files.
    """
    if not os.path.isdir(directory):
        logging.error(f"Error: Directory not found: {directory}")
        return

    # Find all .pt files in the directory (non-recursive)
    # Use os.path.join for cross-platform compatibility
    search_pattern = os.path.join(directory, '*.pt')
    pt_files = glob.glob(search_pattern)

    if not pt_files:
        logging.warning(f"No .pt files found in directory: {directory}")
        return

    logging.info(f"Found {len(pt_files)} .pt files in {directory}. Starting processing...")
    
    modified_count = 0
    skipped_count = 0

    for file_path in pt_files:
        try:
            # Load the tensor - use map_location='cpu' for safety
            tensor = torch.load(file_path, map_location=torch.device('cpu'))

            original_shape = tensor.shape
            
            # Check if the tensor has 3 dimensions and the first dimension is 1
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                # Remove the leading dimension
                modified_tensor = tensor.squeeze(0)
                
                # Ensure the resulting tensor is 2D (safety check)
                if modified_tensor.ndim != 2:
                     logging.warning(f"Skipping {os.path.basename(file_path)}: Squeezing resulted in unexpected shape {modified_tensor.shape}")
                     skipped_count += 1
                     continue

                # Save the modified tensor back to the original file path
                torch.save(modified_tensor, file_path)
                logging.info(f"Modified: {os.path.basename(file_path)} (Shape changed from {original_shape} to {modified_tensor.shape})")
                modified_count += 1
            else:
                # Tensor shape is already as desired or doesn't match [1, N, D]
                logging.info(f"Skipped: {os.path.basename(file_path)} (Shape {original_shape} already correct or doesn't match pattern)")
                skipped_count += 1

        except Exception as e:
            logging.error(f"Error processing file {os.path.basename(file_path)}: {e}")
            skipped_count += 1 # Count errors as skipped

    logging.info("-" * 30)
    logging.info("Processing finished.")
    logging.info(f"Total files processed: {len(pt_files)}")
    logging.info(f"Files modified: {modified_count}")
    logging.info(f"Files skipped/unchanged: {skipped_count}")

# --- Run the modification ---
if __name__ == "__main__":
    # Make sure the user has set the directory
    if PT_FILES_DIRECTORY == "/path/to/your/pt_files":
         print("ERROR: Please set the 'PT_FILES_DIRECTORY' variable in the script to the correct path.")
    else:
        # Ask for confirmation before proceeding, due to overwrite risk
        confirm = input(f"This script will OVERWRITE .pt files in '{PT_FILES_DIRECTORY}'.\nHave you backed up your data? (yes/no): ")
        if confirm.lower() == 'yes':
            modify_pt_files(PT_FILES_DIRECTORY)
        else:
            print("Operation cancelled. Please back up your data first.")
