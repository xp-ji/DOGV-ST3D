from torch.utils.data import Dataset
from transforms.group_transforms import dynamic_temporal_sparse_sampling
from transforms.group_transforms import group_data_transforms
from dataloader.depth_utils import read_depth_maps, cv2_loader, get_path_list_form_file_list

class DepthDataset(Dataset):
    def __init__(self, data_path, intype, index_file,
                 short_param, long_param):
        short_transform = group_data_transforms(short_param, intype)
        self.short_transform = short_transform
        self.short_param = short_param
        self.long_param = long_param

        self.depth_format = "{idx:08d}.png"
        self.data_path = data_path

        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(self.data_path, index_file)

    def __getitem__(self, index):
        target_file_dir = self.file_list[index]
        target_label = self.label_list[index]
        target_length = self.frame_list[index]

        target_data = self._loading_data(target_file_dir, target_length)

        return target_data, target_label, target_length

    def __len__(self):
        return len(self.file_list)

    def _loading_data(self, file_path, file_count):
        file_idx = dynamic_temporal_sparse_sampling(file_count, self.long_param)

        pre_depth_clip = self.short_param["pre_depth_clip"]
        pre_region_crop = self.short_param["pre_region_crop"]

        data_processed = read_depth_maps(file_path, self.depth_format,
                                         img_loader=cv2_loader,
                                         file_idx=file_idx,
                                         pre_depth_clip=pre_depth_clip,
                                         pre_region_crop=pre_region_crop)

        if self.short_transform is not None:
            data_processed = self.short_transform(data_processed)

        return data_processed

class NTURGBD(DepthDataset):
    def __init__(self, data_path, intype, index_file,
                 short_param, long_param, is_masked=True):
        short_transform = group_data_transforms(short_param, intype)
        self.short_transform = short_transform
        self.short_param = short_param
        self.long_param = long_param

        if is_masked:
            self.depth_format = "MDepth-{idx:08d}.png"
            self.data_path = data_path + "_masked"
        else:
            self.depth_format = "Depth-{idx:08d}.png"
            self.data_path = data_path

        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(self.data_path, index_file)

class PKUMMD(DepthDataset):
    def __init__(self, data_path, intype, index_file,
                 short_param, long_param, is_masked=False):
        short_transform = group_data_transforms(short_param, intype)
        self.short_transform = short_transform
        self.short_param = short_param
        self.long_param = long_param

        if is_masked:
            raise ValueError("Segment option is not supported by PKU_MMD dataset!")
        else:
            self.depth_format = "Depth-{idx:06d}.png"
            self.data_path = data_path

        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(self.data_path, index_file)

class UOWCombined3D(DepthDataset):
    def __init__(self, data_path, intype, index_file,
                 short_param, long_param, is_masked=False):
        short_transform = group_data_transforms(short_param, intype)
        self.short_transform = short_transform
        self.short_param = short_param
        self.long_param = long_param

        if is_masked:
            raise ValueError("Segment option is not supported by UOW Combined dataset!")
        else:
            self.depth_format = "Depth-{idx:06d}.png"
            self.data_path = data_path

        self.file_list, self.label_list, self.frame_list = \
            get_path_list_form_file_list(self.data_path, index_file)
        

def depth_dataset(opt, config, subset):
    short_param = config[subset]['short_param']
    long_param = config[subset]['long_param']
    intype = opt.intype
    data_path = config['data_path']
    index_file = config[subset]['file_list']

    if opt.dataset == 'NTU_RGBD' or opt.dataset == 'NTU_120':
        target_dataset = NTURGBD(data_path, intype, index_file, short_param, long_param)

    elif opt.dataset == 'PKU_MMD':
        target_dataset = PKUMMD(data_path, intype, index_file, short_param, long_param)

    elif opt.dataset == 'UOW_Combined3D':
        target_dataset = UOWCombined3D(data_path, intype, index_file, short_param, long_param)

    else:
        raise ValueError("Unknown dataset: '{:s}'".format(opt.dataset))

    return target_dataset
    
