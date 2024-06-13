import csv
import re


def extract_unique_corrupted_scenes(input_file, output_file):
    seen_scenes = set()
    pattern = re.compile(r"The corrupted scene is: '.+/sequences/(\d+)/velodyne/(\d+).bin")

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["scene", "sequence"])  # Write header

        for line in infile:
            if "The corrupted scene is:" in line:
                match = pattern.search(line)
                if match:
                    scene, sequence = match.groups()
                    scene_info = (sequence, scene)
                    if scene_info not in seen_scenes:
                        seen_scenes.add(scene_info)
                        writer.writerow([scene, sequence])


if __name__ == "__main__":
    input_file = "/home/fradlin/Github/Mask4D-Interactive/outputs/9950693_train_things_only_test.txt"  # Replace with the actual input file path
    output_file = "corrupted_scenes.csv"  # Replace with the desired output file path
    extract_unique_corrupted_scenes(input_file, output_file)


# python extract_unique_corrupted_scenes.py
