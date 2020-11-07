from detectron2 import model_zoo
from detectron2.config import get_cfg


def getTrainCfg():
    pass


def getEvalCfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("stars_carla_train",)
    cfg.DATASETS.TEST = ("stars_carla_val",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1080)
    cfg.INPUT.MAX_SIZE_TRAIN = 1920
    cfg.INPUT.MAX_SIZE_TEST = 1920
    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.freeze()
    return cfg
