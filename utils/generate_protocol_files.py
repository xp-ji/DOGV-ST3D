import json
import os
import random
from utils.trivial_definition import separator_line
from configs.param_config import ConfigClass
import numpy as np

def aggregate_pku_label_list(label_dir):
    txt_list = os.listdir(label_dir)
    txt_list = list(filter(lambda x: x.endswith('.txt'), txt_list))
    txt_list.sort()

    label_sequence = []
    seqs_count = 0
    for txt_i in txt_list:
        txt_file = label_dir + "/" + txt_i
        with open(txt_file) as lf:
            for line in lf.readlines():
                label_line = line.strip("\n").split(",")

                label_line = [int(i) for i in label_line]

                # video_name, action_no, frame_start, frame_end
                assert label_line[2]>label_line[1], "{:s} Action {:d}: end frame should be large than start frame".format(txt_file,  label_line[0])

                label_sequence.append([txt_i.replace(".txt", ""),
                                       label_line[0], label_line[1], label_line[2]])

                seqs_count += 1
            lf.close()
    assert seqs_count == len(label_sequence)

    return label_sequence

def generate_protocol_for_UOWCombined3D(dataset_param, logger, expand_train_list=True):
    """
    generate_protocol_for_UOWCombined3D
    :param dataset_param:
    :return:
    """
    data_Name = "UOWCombined3D"
    dataset_dir = dataset_param["data_dir"]
    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["file_format"]["Skeleton"] = dataset_param["data_type"]["Skeleton"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    train_subs = list(range(1, data_attr["n_subs"] + 1))[::2]  # odd for train
    train_exps = list(range(1, data_attr["n_exps"] + 1))[::2]  # odd for train
    random_train_subs = random.sample(list(range(1, data_attr["n_subs"] + 1)), data_attr["n_subs"]//2+1)

    for p_i, protocol_i in enumerate(protocols):

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        # constant check
        if dataset_param["data_constant_check"]:
            depth_samples_list = os.listdir(data_type["Depth"])
            skeleton_samples_list = os.listdir(data_type["Skeleton"])
            skeleton_samples_list = [x[0:14] for x in skeleton_samples_list]

            samples_list = list(set(depth_samples_list) & set(skeleton_samples_list))

        else:
            samples_list = os.listdir(data_type["Depth"])

        samples_list.sort()

        data_path = data_type["Depth"]
        action_list = []

        if protocol_i == "random_cross_subjects":
            logger.info("subject id ({:d}) for training: ".format(len(random_train_subs)))
            logger.info(random_train_subs)
            logger.info(separator_line(dis_len="half"))

        for filename in samples_list:
            action_class = int(
                filename[filename.find('a') + 1:filename.find('a') + 4])

            action_list.append(action_class)

        action_list = list(set(action_list))
        action_list.sort()

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes, len(data_attr["n_names"])))

        for filename in samples_list:
            action_class = int(
                filename[filename.find('a') + 1:filename.find('a') + 4])

            subject_id = int(
                filename[filename.find('s') + 1:filename.find('s') + 4])
            exp_id = int(
                filename[filename.find('e') + 1:filename.find('e') + 4])

            if protocol_i == "cross_subjects":
                istraining = (subject_id in train_subs)

            elif protocol_i == "cross_samples":
                istraining = (exp_id in train_exps)

            elif protocol_i == "random_cross_subjects":
                istraining = (subject_id in random_train_subs)

            elif protocol_i == "random_cross_samples":
                istraining = random.random() < 0.5

            else:
                raise ValueError()

            img_path = data_path + "/" + filename

            img_count = 0

            for img_name in os.listdir(img_path):
                if ".png" in img_name:
                    img_count += 1

            #img_count = len(os.listdir(img_path))

            assert img_count > 0, ValueError("Empty folder!")


            action_label = action_list.index(action_class)

            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=filename,
                                                                    label=action_label,
                                                                    frames=img_count)

            if istraining:
                train_list.append(line_str)
            else:
                test_list.append(line_str)


        if expand_train_list:
            # expand the train_list
            tile_times = int(2.5e4 // len(train_list) + 1)
            train_list = np.tile(train_list, tile_times)

        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file


def generate_protocol_for_NTU_RGBD(dataset_param, logger):
    """
    generate_protocol_for_MSRAction3D:
    :param dataset_param:
    :return:
    """
    data_Name = "NTU_RGB+D"
    dataset_dir = dataset_param["data_dir"]

    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["file_format"]["Skeleton"] = dataset_param["data_type"]["Skeleton"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    for p_i, protocol_i in enumerate(protocols):

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        action_list = []

        data_path = dataset_param["data_type"]["Depth"] + "_masked_clip"

        file_list = os.listdir(data_path)

        for filename in file_list:

            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(
                filename[filename.find('C') + 1:filename.find('C') + 4])

            if protocol_i == "cross_view":
                istraining = (camera_id in protocol_param["train_cam"])

            elif protocol_i == "cross_subjects":
                istraining = (subject_id in protocol_param["train_subs"])

            else:
                raise ValueError()

            img_path = data_path + "/" + filename

            img_count = 0

            for img_name in os.listdir(img_path):
                if ".png" in img_name:
                    img_count += 1

            #img_count = len(os.listdir(img_path))

            assert img_count > 0, ValueError("Empty folder!")

            action_list.append(action_class)

            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=filename,
                                                                    label=action_class - 1,
                                                                    frames=img_count)

            if istraining:
                train_list.append(line_str)
            else:
                test_list.append(line_str)

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes,
                                                                                             len(data_attr["n_names"])))
        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file

def generate_protocol_for_PKU_MMD(dataset_param, logger):
    """
    generate_protocol_for_PKU_MMD:
    :param dataset_param:
    :return:
    """
    data_Name = "PKU-MMD"
    dataset_dir = dataset_param["data_dir"]

    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    for p_i, protocol_i in enumerate(protocols):

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        action_list = []

        label_sequence = aggregate_pku_label_list(protocol_param["Label"])

        data_path = dataset_param["data_type"]["Depth"]

        spilt_file = protocol_param["spilt_file"]

        txt_content = []
        with open(spilt_file,  "r") as sf:
            for line in sf.readlines():
                txt_content.append(line.strip("\n"))
            sf.close()

        videos_for_train = txt_content[1].replace(" ", "").split(",")[0:-1]
        videos_for_test = txt_content[3].replace(" ", "").split(",")[0:-1]

        for sequence_i in label_sequence:
            video_name, action_no, frms_start, frms_end = sequence_i
            img_dir = data_path + "/" + video_name

            if not os.path.isdir(img_dir):
                raise ValueError("{:s} does not exist!".format(img_dir))

            if video_name in videos_for_train:
                istraining = True

            elif video_name in videos_for_test:
                istraining = False
            else:
                istraining = None
            # istraining = (video_name in videos_for_train)


            if dataset_param["data_exist_check"]:
                for i in range(frms_start, frms_end):
                    img_name = "{:s}/depth-{:06d}.png".format(img_dir, i)

                    if not os.path.isfile(img_name):
                        raise ValueError("{:s} does not exist!".format(img_name))

            action_list.append(action_no)

            line_str = "{file_str:s}\t{label:d}\t{frms_start:d}\t{frms_end:d}".format(file_str=video_name,
                                                                                      label=action_no-1,
                                                                                      frms_start=frms_start,
                                                                                      frms_end=frms_end)
            if istraining is not None:
                if istraining:
                    train_list.append(line_str)
                else:
                    test_list.append(line_str)

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes,
                                                                                             len(data_attr["n_names"])))
        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file

def generate_protocol_for_PKU_MMD_Spilt(dataset_param, logger, expand_train_list=True):
    """
    generate_protocol_for_PKU_MMD:
    :param dataset_param:
    :return:
    """
    data_Name = "PKU-MMD"
    dataset_dir = dataset_param["data_dir"]

    data_attr = dataset_param["data_attr"]
    data_type = dataset_param["data_type"]

    protocols = list(dataset_param["eval_protocols"].keys())

    # AssertionError for none protocols
    assert len(protocols) > 0, "evaluation protocol should be declared!"

    # Preparing new dict for json file
    data_param_dict = {}
    data_param_dict["data_Name"] = data_Name
    data_param_dict["data_Path"] = dataset_dir
    data_param_dict["file_format"] = {}
    data_param_dict["file_format"]["Depth"] = dataset_param["data_type"]["Depth"]
    data_param_dict["eval_Protocol"] = {}
    data_param_dict["action_Names"] = data_attr["n_names"]

    file_list_dir = dataset_param["file_list_dir"]

    if not os.path.exists(file_list_dir):
        os.makedirs(file_list_dir)

    for p_i, protocol_i in enumerate(protocols):
        # if p_i<1:
        #     continue

        protocol_param = dataset_param["eval_protocols"][protocol_i]
        train_list = []
        test_list = []

        action_list = []

        data_path = dataset_param["data_type"]["Depth"]

        spilt_file = protocol_param["spilt_file"]

        txt_content = []
        with open(spilt_file,  "r") as sf:
            for line in sf.readlines():
                txt_content.append(line.strip("\n"))
            sf.close()

        videos_for_train = txt_content[1].replace(" ", "").split(",")[0:-1]
        videos_for_test = txt_content[3].replace(" ", "").split(",")[0:-1]

        file_list = os.listdir(data_path)

        file_list.sort()

        count_i = 0

        for filename in file_list:

            video_name = filename[filename.find('V') + 1:filename.find('V') + 7]
            views = filename[filename.find('-') + 1:filename.find('_')]
            action_class = int(filename[filename.find('A') + 1:filename.find('A') + 3])

            img_path = data_path + "/" + filename

            img_count = 0

            for img_name in os.listdir(img_path):
                if ".png" in img_name:
                    img_count += 1

            assert img_count > 0, ValueError("Empty folder for {:s}!".format(filename))

            img_test = img_path + "/Depth-{:06d}.png".format(img_count-1)

            if not os.path.isfile(img_test):
                print(img_test)
                count_i += 1



            if video_name in videos_for_train:
                istraining = True
            elif video_name in videos_for_test:
                istraining = False
            else:
                istraining = None

            # istraining = (video_name in videos_for_train)

            if img_count<16:
                continue

            action_list.append(action_class)

            line_str = "{file_str:s}\t{label:d}\t{frames:d}".format(file_str=filename,
                                                                    label=action_class - 1,
                                                                    frames=img_count)

            if istraining is not None:
                if istraining:
                    train_list.append(line_str)
                else:
                    test_list.append(line_str)

        # print("----")
        # print(count_i)
        #
        # raise RuntimeError

        if expand_train_list:
            # expand the train_list
            tile_times = int(2.5e4 // len(train_list) + 1)
            train_list = np.tile(train_list, tile_times)

        # constant check for num classes
        num_classes = len(set(action_list))

        if num_classes != len(data_attr["n_names"]):
            logger.warn("Warning: num classes: {:d} is not equal to class names: {}.".format(num_classes,
                                                                                             len(data_attr["n_names"])))
        # protocol name
        protocol_item = "{:d}_{:s}".format(p_i+1, protocol_i)

        logger.info(protocol_item+": ")

        # Write train list to file
        train_list_file = protocol_param["file_list_train"].replace("<$protocol_item>",
                                                                    protocol_item)
        with open(train_list_file, "w") as trlf:
            for train_line in train_list:
                trlf.write(train_line + "\n")
            trlf.close()
            logger.info("    Train filelist has been stored in '{:s}'".format(train_list_file))

        # Write test list to file
        test_list_file = protocol_param["file_list_test"].replace("<$protocol_item>",
                                                                  protocol_item)
        with open(test_list_file, "w") as telf:
            for test_line in test_list:
                telf.write(test_line + "\n")
            telf.close()
            logger.info("    Test filelist has been stored in '{:s}'".format(test_list_file))

        logger.info("    => Summary: {:d} samples for training and {:d} samples for test.".format(len(train_list),
                                                                                        len(test_list)))
        logger.info("    => Number of classes: {:d}".format(num_classes))

        logger.info(separator_line(dis_len="half"))

        assert len(train_list) > 0 and len(test_list) > 0, "Target dataset has no samples to read."
        data_param_dict["eval_Protocol"][protocol_item] = {}
        data_param_dict["eval_Protocol"][protocol_item]["train"] = train_list_file
        data_param_dict["eval_Protocol"][protocol_item]["test"] = test_list_file

    # num_classes to eval_Protocol
    data_param_dict["num_classes"] = num_classes


    # write protocol param to json file
    data_param_dict_file = dataset_param["eval_config_file"].format(file_list_dir, data_Name)

    with open(data_param_dict_file, 'w') as jsf:
        json.dump(data_param_dict, jsf, indent=4)
        logger.info("Evaluation protocols have been stored in '{:s}'".format(data_param_dict_file))
        logger.info(separator_line())

    return data_param_dict_file

def generate_protocol_files(data_name, dataset_param, logger):
    """generate_protocol_files
    :param data_name:
    :param dataset_param:
    :return:
    """
    if data_name == "NTU_RGB+D":
        return generate_protocol_for_NTU_RGBD(dataset_param, logger)

    elif data_name == "PKU_MMD":
        return generate_protocol_for_PKU_MMD_Spilt(dataset_param, logger)

    elif "UOW_Combined3D" in data_name:
        return generate_protocol_for_UOWCombined3D(dataset_param, logger)

    else:
        raise ValueError("Unknown dataset: '{:s}'".format(data_name))


if __name__ == '__main__':

    config = ConfigClass()

    data_name = "NTU_RGB+D"
    dataset_param = config.get_dataset_param(data_name)
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    generate_protocol_files(data_name, dataset_param, logger)

    # generate_protocol_for_UOWCombined3D(dataset_param, logger)

