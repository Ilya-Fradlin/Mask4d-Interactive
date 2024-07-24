import json


def downsample_json(input_file, output_file, n):
    # Read the input JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Create a new dictionary to store the downsampled data
    downsampled_data = {}

    # Loop through the data and keep every nth entry
    keys = list(data.keys())
    for i in range(0, len(keys), n):
        key = keys[i]
        downsampled_data[key] = data[key]

    # Write the downsampled data to the output JSON file
    with open(output_file, "w") as f:
        json.dump(downsampled_data, f, indent=4)


# Example usage
input_file = "full_validation_list.json"
output_file = "short_validation_list.json"
n = 12  # Keep every 12th entry
downsample_json(input_file, output_file, n)
