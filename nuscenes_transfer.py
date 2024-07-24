import json
import os
import subprocess

from tqdm import tqdm

# Load the JSON file
with open("/home/fradlin/Github/Mask4D-Interactive/datasets/preprocessing/nuscenes_validation_list.json", "r") as file:
    data = json.load(file)

ssh_passphrase = "Luizapassphrase97!"


# Function to create directories on the destination server
def create_directories(destination_path):
    parent_directory = os.path.dirname(destination_path)
    command = f"ssh fradlin1@juwels-booster.fz-juelich.de 'mkdir -p {parent_directory}'"
    subprocess.run(command, shell=True, check=True)


# Iterate through the items and generate rsync commands
for key, value in tqdm(data.items()):
    filepath = value.get("filepath")
    label_filepath = value.get("label_filepath")

    for path in [filepath, label_filepath]:
        if path:
            # Construct the destination path
            destination_path = path.replace("/globalwork/datasets/", "/p/scratch/objectsegvideo/ilya/")

            # Ensure the parent directories exist on the destination server
            create_directories(destination_path)

            # Construct the rsync command
            rsync_command = f"rsync -a --ignore-existing -e ssh --progress {path} fradlin1@juwels-booster.fz-juelich.de:{destination_path}"

            # Print the rsync command (or execute it using subprocess)
            # print(rsync_command)
            subprocess.run(rsync_command, shell=True, check=True)
