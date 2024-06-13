import csv
import re


def extract_unique_corrupted_scenes(input_file, output_file):
    seen_scenes = set()
    pattern = re.compile(r"The corrupted scene is: '.+/sequences/(\d+)/velodyne/(\d+).bin")

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["sequence", "scene", "line"])  # Write header

        for line in infile:
            if "The corrupted scene is:" in line:
                match = pattern.search(line)
                if match:
                    sequence, scene = match.groups()
                    scene_info = (sequence, scene)
                    if scene_info not in seen_scenes:
                        seen_scenes.add(scene_info)
                        writer.writerow([sequence, scene, line.strip()])


if __name__ == "__main__":
    input_file = "path_to_input_file.txt"  # Replace with the actual input file path
    output_file = "corrupted_scenes.csv"  # Replace with the desired output file path
    extract_unique_corrupted_scenes(input_file, output_file)


# python extract_unique_corrupted_scenes.py
