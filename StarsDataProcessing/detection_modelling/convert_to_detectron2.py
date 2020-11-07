import glob
import json
import os
import random

bbox_mode = {"XYXY_ABS": 0, "XYWH_ABS": 1}  # from detectron definition
class_dict = {"vehicle": 0, "bike": 1, "pedestrian": 2}


def get_image_record(image_path):
    # Assuming imagename.png is the image and imagename.json is the annotation file
    v = image_path
    jsonfile = open(v[:-4] + ".json")
    record = {}
    annotations = json.load(jsonfile)
    jsonfile.close()

    filename = v
    image_name = v[v.rfind(os.sep) + 1 :]
    # height, width = cv2.imread(filename).shape[:2] #bypassing as our height and width are constant
    height = 1080
    width = 1920

    record["file_name"] = filename
    record["image_id"] = image_name
    record["height"] = height
    record["width"] = width

    # annos = v["regions"]
    objs = []
    for anno in annotations:
        # assert not anno["region_attributes"]
        # anno = anno["shape_attributes"]
        # px = anno["all_points_x"]
        # py = anno["all_points_y"]
        # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        # poly = [p for x in poly for p in x]
        classname = anno["classname"]
        obj = {
            "bbox": anno["bounding_box2d"],
            "bbox_mode": bbox_mode[anno["bounding_box2d_format"]],
            "segmentation": [],
            "category_id": class_dict[classname],
            "iscrowd": 0,
        }
        objs.append(obj)
    record["annotations"] = objs
    return record


def get_dict(dataset_dir="output_data/test1", train_test_ratio=0.8):

    # train_test_ratio = 0.8

    imgs = glob.glob(dataset_dir + "/*.png")
    num_images = len(imgs)
    train_count = int(train_test_ratio * num_images)
    image_idx_list = list(range(num_images))
    if train_test_ratio < 1:
        random.shuffle(image_idx_list)
    else:
        imgs = sorted(imgs)  # ensure the glob images are sorted(helps in visualization)

    train_idx = image_idx_list[0:train_count]
    val_idx = image_idx_list[train_count:]

    train_set = []
    val_set = []

    for idx in train_idx:
        img_file = imgs[idx]
        record = get_image_record(img_file)
        train_set.append(record)

    for idx in val_idx:
        img_file = imgs[idx]
        record = get_image_record(img_file)
        val_set.append(record)

    return train_set, val_set

    # return dataset_dicts


def adhocConvertFolder(folder_path, to_file=False):
    """Convert one folder as it is to detectron2 format, without any train,val splits
    
    Arguments:
        folder_path {[type]} -- [description]
    """
    train, _ = get_dict(folder_path, train_test_ratio=1.0)
    if to_file:
        json.dump(train, open("adhoc_dataset.json", "w"))
    else:
        return train


if __name__ == "__main__":
    # user input
    folders_to_convert = [
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_0",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_0_1",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_1",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_1_1",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_2",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_2_1",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_3",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_3_1",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_4",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_5",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_6",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_7",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_8",
        "/home/stars/Code/carla_utils/new_output_data/14 April 2020 16_59_39 2FPS/output_data/round_9",
    ]
    train_dict_list = []
    val_dict_list = []
    for folder in folders_to_convert:
        train, val = get_dict(folder)
        train_dict_list += train
        val_dict_list += val
    print(len(train_dict_list), len(val_dict_list))
    json.dump(train_dict_list, open("Temp_stars_carla_train.json", "w"))
    json.dump(val_dict_list, open("Temp_stars_carla_val.json", "w"))

# 1920 train and 480 test
# 2688 train 672 test
