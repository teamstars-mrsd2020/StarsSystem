
# ensure to activate an environment that has detectron2 installed
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import BoxMode
import json

import os


def get_train_dicts():
    data = json.load(open("stars_carla_train.json"))
    for im in data:
        anno = im['annotations']
        for an in anno:
            an['bbox_mode'] = BoxMode(an['bbox_mode'])
    return data


def get_val_dicts():
    data = json.load(open("stars_carla_val.json"))
    for im in data:
        anno = im['annotations']
        for an in anno:
            an['bbox_mode'] = BoxMode(an['bbox_mode'])
    return data


# register datasets
DatasetCatalog.register("stars_carla_train", get_train_dicts)
DatasetCatalog.register("stars_carla_val", get_val_dicts)

MetadataCatalog.get("stars_carla_train").thing_classes = ["vehicle", "bike", "pedestrian"]
MetadataCatalog.get("stars_carla_val").thing_classes = ["vehicle", "bike", "pedestrian"]


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, True, "./coco_eval_output")


# create cfg
def setup():
    # todo: check shortest edge length
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("stars_carla_train",)
    cfg.DATASETS.TEST = ("stars_carla_val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 6000 # number of iterations, num epochs = iter*batch_size/train_set_size
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # save a checkpoint every 2000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.MIN_SIZE_TRAIN = (800,1080)
    cfg.INPUT.MAX_SIZE_TRAIN = 1920
    cfg.INPUT.MAX_SIZE_TEST = 1920
    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.TEST.EVAL_PERIOD = 500 
    cfg.VIS_PERIOD = 500
    return cfg


def main():
    cfg = setup()
    # print(cfg.INPUT.FORMAT)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    # main()
    num_gpus = 4
    launch(
        main,
        num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url = "tcp://127.0.0.1:12312"
    )
