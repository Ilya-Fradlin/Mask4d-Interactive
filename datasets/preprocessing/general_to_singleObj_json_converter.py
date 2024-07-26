# import json
# from tqdm import tqdm

# # Load the original JSON data
# with open("/globalwork/fradlin/mask4d-interactive/processed/semantic_kitti/full_validation_list.json", "r") as f:
#     data = json.load(f)

# new_data = {}

# # Iterate through each key in the original JSON
# for key, value in tqdm(data.items(), desc="Processing scenes"):
#     base_scene_name = key
#     labels = value["unique_panoptic_labels"]

#     # Create a new entry for each unique label
#     for i, label in enumerate(labels, start=1):
#         new_key = f"{base_scene_name}_{i}"
#         new_value = value.copy()
#         new_value["unique_panoptic_labels"] = [label]
#         new_data[new_key] = new_value

# # Save the new JSON data
# with open("/globalwork/fradlin/mask4d-interactive/processed/semantic_kitti/full_single_validation_list.json", "w") as f:
#     json.dump(new_data, f, indent=2)

import json
from tqdm import tqdm

# Load the original JSON data
with open("/globalwork/fradlin/mask4d-interactive/processed/semantic_kitti/full_single_validation_list.json", "r") as f:
    data = json.load(f)

new_data = {}

# Iterate through each key in the original JSON
for key, value in tqdm(data.items(), desc="Processing scenes"):
    # Modify the "clicks" dictionary to keep only keys "1" and "0"
    new_clicks = {k: v for k, v in value["clicks"].items() if k in ["1", "0"]}

    # Modify the "obj" dictionary to keep only key "1" with the first unique_panoptic_label
    first_label = value["unique_panoptic_labels"][0]
    new_obj = {"1": first_label}

    # Update the value with modified "clicks" and "obj"
    new_value = value.copy()
    new_value["clicks"] = new_clicks
    new_value["obj"] = new_obj

    # Add the modified entry to the new_data dictionary
    new_data[key] = new_value

# Save the new JSON data
print("Processing complete. saving...")
with open("/globalwork/fradlin/mask4d-interactive/processed/semantic_kitti/full_single_validation_list.json", "w") as f:
    json.dump(new_data, f, indent=2)

print("saving complete. New data saved to 'full_single_validation_list.json'.")
