import os
import sys
import argparse
import shb_video_tools as shb
import easyocr

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Analyse all subfolders within a root folder from a SHB experiment.")
    p.add_argument("root", help="The root folder containing experiment subfolders (e.g., 'camera_data').")
    return p.parse_args()

def main():
    """
    Main function to orchestrate the analysis of all subfolders.
    """
    args = parse_args()
    root_folder = args.root

    # Check if the provided root folder exists
    if not os.path.isdir(root_folder):
        print(f"Error: Root folder not found at '{root_folder}'", file=sys.stderr)
        sys.exit(1) # Exit the script with an error code

    # --- Create the main results directory, e.g., 'camera_data_results' ---
    folder_name = os.path.basename(os.path.normpath(root_folder))
    main_output_dir = f"{folder_name}_results"
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Results will be saved in: '{main_output_dir}'")

    # --- Initialize the EasyOCR reader once to avoid reloading the model ---
    print("Initializing EasyOCR reader... (this may take a moment)")
    # You can customize the languages as needed, e.g., ['en'] for English
    reader = easyocr.Reader(['en'])
    print("Reader initialized.")

    # --- Find and process all immediate subdirectories in the root folder ---
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    if not subfolders:
        print(f"Warning: No subfolders found in '{root_folder}'.")
        return

    print(f"Found {len(subfolders)} subfolders to analyse.")

    # --- Process each subfolder by calling the function from the library ---
    for subfolder in subfolders:
        try:
            print(f"Delegating analysis of '{subfolder}' to shb_video_tools...")
            # Pass the initialized reader to the analysis function
            shb.analyse_folder(subfolder, main_output_dir, reader)
        except AttributeError:
            print(f"Error: 'analyse_folder' function not found in 'shb_video_tools' library.", file=sys.stderr)
            print("Please ensure the function is defined in your library.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            # Catches other potential errors during the analysis of a folder
            print(f"An error occurred while analysing '{subfolder}': {e}", file=sys.stderr)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    # This block ensures the main function runs when the script is executed
    main()