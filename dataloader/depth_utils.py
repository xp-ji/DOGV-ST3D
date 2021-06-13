import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def get_path_list_form_file_list(data_path, file_list_file):
    """
    get_path_list_form_file_list:
    :param file_format:
    :param file_list_file:
    :return:
    """

    file_dir_list = []
    label_list = []
    frame_list = []

    with open(file_list_file, "r") as flp:
        for line in flp.readlines():
            flp_line = line.strip("\n").split("\t")

            # default format is "<$file_format>/t<$label>/t<[frames,height,width]>"
            target_file_path = data_path + "/" + flp_line[0]
            target_label = int(flp_line[1])
            target_length = int(flp_line[2])

            file_dir_list.append(target_file_path)
            label_list.append(target_label)
            frame_list.append(target_length)
        flp.close()

    return file_dir_list, label_list, frame_list

def read_depth_maps(depth_file_dir, depth_format, img_loader=None,
                    file_idx=None, pre_depth_clip=None, pre_region_crop=None):

    if isinstance(file_idx[0], list):  # nested list
        depth_file_list = [[depth_format.format(idx=idx + 1) for idx in idx_inlist] for idx_inlist in file_idx]
    else:
        depth_file_list = [depth_format.format(idx=idx + 1) for idx in file_idx]

    img_array_list = []

    for img_i in depth_file_list:
        img_path = depth_file_dir + "/" + img_i
        assert os.path.isfile(img_path), "{:s} does not exist!".format(img_path)

        img = img_loader(img_path)

        # if add_noise
        img_array_list.append(img)

    img_array = np.concatenate([np.expand_dims(x, 0) for x in img_array_list], axis=0)

    # if depth offset enhancement
    if pre_region_crop is not None:
        h_s, h_e, w_s, w_e = pre_region_crop
        img_array_crop = img_array[:, h_s: h_e, w_s: w_e]

    else:
        img_array_crop = img_array

    if pre_depth_clip is not None:
        depth_min, depth_max = pre_depth_clip
        img_array_crop = img_array_crop.astype(np.float)  # uint16 -> float

        img_array_crop = (img_array_crop - depth_min) / (depth_max-depth_min)  # depth value clip
        img_array_crop = np.clip(img_array_crop, 0, 1)
        img_array_crop = img_array_crop * 255.0

    img_array_crop = img_array_crop[:, np.newaxis] # FxCxHxW

    return img_array_crop

    
