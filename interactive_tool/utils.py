try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")
import torch
import numpy as np

# constants and flags
USE_TRAINING_CLICKS = False
OBJECT_CLICK_COLOR = [0.2, 0.81, 0.2]  # colors between 0 and 1 for open3d
BACKGROUND_CLICK_COLOR = [0.81, 0.2, 0.2]  # colors between 0 and 1 for open3d
UNSELECTED_OBJECTS_COLOR = [0.4, 0.4, 0.4]
SELECTED_OBJECT_COLOR = [0.2, 0.81, 0.2]
obj_color = {1: [1, 211, 211], 2: [233, 138, 0], 3: [41, 207, 2], 4: [244, 0, 128], 5: [194, 193, 3], 6: [121, 59, 50], 7: [254, 180, 214], 8: [239, 1, 51], 9: [125, 0, 237], 10: [229, 14, 241]}

ply_dtypes = dict([(b"int8", "i1"), (b"char", "i1"), (b"uint8", "u1"), (b"uchar", "u1"), (b"int16", "i2"), (b"short", "i2"), (b"uint16", "u2"), (b"ushort", "u2"), (b"int32", "i4"), (b"int", "i4"), (b"uint32", "u4"), (b"uint", "u4"), (b"float32", "f4"), (b"float", "f4"), (b"float64", "f8"), (b"double", "f8")])

# Numpy reader format
valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


def get_obj_color(obj_idx, normalize=False):

    r, g, b = obj_color[obj_idx]

    if normalize:
        r /= 256
        g /= 256
        b /= 256

    return [r, g, b]


def find_nearest(coordinates, value):
    distance = torch.cdist(coordinates, torch.tensor([value]).to(coordinates.device), p=2)
    return distance.argmin().tolist()


def mean_iou_single(pred, labels):
    truepositive = pred * labels
    intersection = torch.sum(truepositive == 1)
    uni = torch.sum(pred == 1) + torch.sum(labels == 1) - intersection

    iou = intersection / uni
    return iou


def mean_iou_scene(pred, labels):

    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids != 0]
    obj_num = len(obj_ids)
    iou_sample = 0.0
    iou_dict = {}
    for obj_id in obj_ids:
        obj_iou = mean_iou_single(pred == obj_id, labels == obj_id)
        iou_dict[int(obj_id)] = float(obj_iou)
        iou_sample += obj_iou

    iou_sample /= obj_num

    return iou_sample, iou_dict


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, "rb") as plyfile:

        # Check if the file start with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start whith the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [("k", ext + "u1"), ("v1", ext + "i4"), ("v2", ext + "i4"), ("v3", ext + "i4")]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data["v1"], faces_data["v2"], faces_data["v3"])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        # Find point element
        if b"element vertex" in line:
            current_element = "vertex"
            line = line.split()
            num_points = int(line[2])

        elif b"element face" in line:
            current_element = "face"
            line = line.split()
            num_faces = int(line[2])

        elif b"property" in line:
            if current_element == "vertex":
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == "vertex":
                if not line.startswith("property list uchar int"):
                    raise ValueError("Unsupported faces property : " + line)

    return num_points, num_faces, vertex_properties
