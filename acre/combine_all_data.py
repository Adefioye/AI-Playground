import os
import json

# Define input and output directories
input_folders = ["text/Comp", "text/IID", "text/Sys"]
output_folder = "text/combined"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each file type (train, test, val)
for file_type in ["train", "test", "val"]:
    combined_data = []
    idx_counter = 0  # Initialize counter for idx

    # Iterate over each input folder
    for folder in input_folders:
        input_file_path = os.path.join(folder, f"{file_type}.jsonl")
        
        # Read the input file
        with open(input_file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                
                # Update the idx to start from 0 and be sequential
                data["idx"] = idx_counter
                idx_counter += 1
                
                # Add the updated data to the combined list
                combined_data.append(data)

    # Write the combined data to the output file
    output_file_path = os.path.join(output_folder, f"{file_type}.jsonl")
    with open(output_file_path, "w") as f:
        for data in combined_data:
            f.write(json.dumps(data) + "\n")

    print(f"Processed {file_type}.jsonl: {len(combined_data)} entries")