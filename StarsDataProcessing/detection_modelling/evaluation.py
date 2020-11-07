# ensure to activate an environment that has detectron2 installed
import copy
import json
import math
import os
import torch
from collections import defaultdict
from tqdm import tqdm
import cv2
from sklearn.metrics import precision_recall_curve
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures.boxes import BoxMode
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor

from json_utils import NumpyEncoder
from stars_config import getEvalCfg

import time
import matplotlib.pyplot as plt
import numpy as np


class ClassWisePRCurve:
    """Accumulator Class for classwise PR Curve data
    """

    def __init__(self, classes_list):
        self.classes_list = classes_list
        self.pr_data = {}
        for c in self.classes_list:
            self.pr_data[c] = {"y_gt": [], "y_conf": []}
        plt.ion()
        min_x = 0
        max_x = 1
        self.figure, self.ax = plt.subplots()
        (self.lines,) = self.ax.plot([], [], ".", markersize=1)
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_xlabel("Recall")
        self.ax.set_ylabel("Precision")
        self.ax.grid()

        # self.annot = None
        # if(self.annot is not None):
        #     self.annot.remove()
        
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

    def addRawData(self, classname, y_gt, y_conf):
        """Accumulator for PR curve
        This data is the class wise results for each class formatted as a binary classifier
        
        Arguments:
            classname {str} -- Class name
            y_gt {list} -- list of the True prediction
            y_conf {list} -- list of the corresponding confidences for each prediction
        """
        self.pr_data[classname]["y_gt"] += y_gt
        self.pr_data[classname]["y_conf"] += y_conf

    def getPRCurveForClass(self, classname):
        """Get the PR Curve for a particular class
        
        Arguments:
            classname {str} -- Class name
        
        Returns:
            (3-tuple) -- array of precisions, recalls, and the threshold steps
        """
        # print(self.pr_data[classname]["y_gt"], self.pr_data[classname]["y_conf"])

        if(len(self.pr_data[classname]["y_gt"])>0):
            precision, recall, thresholds = precision_recall_curve(
                self.pr_data[classname]["y_gt"], self.pr_data[classname]["y_conf"]
            )
        else:
            precision, recall, thresholds = [], [], [] 
        return precision, recall, thresholds

    def getPRCurves(self):
        """Get PR Curves for all classes
        
        Returns:
            {dict} -- Dictional containg the precisions, recalls and threshold steps for each class
        """
        res = {}
        for c in self.classes_list:
            p, r, t = self.getPRCurveForClass(c)
            res[c] = {"precisions": p, "recalls": r, "threshold_steps": t}
        return res

    def updateAnnot(self, ind):
        th = round(100 * np.mean(np.take(self.data['threshold_steps'],ind['ind'])),2)
        x,y = self.lines.get_data()
        self.annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
        text = "Threshold: {}".format(th)
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.lines.contains(event)
            if cont:
                self.updateAnnot(ind)
                self.annot.set_visible(True)
                self.figure.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.figure.canvas.draw_idle()

    def plotCurve(self, data):
        xdata = data["recalls"]
        ydata = data["precisions"]
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()

    def plotPRCurveForClass(self, classname, visualize=True):
        """Plot PR Curve for class
        
        Arguments:
            classname {str} -- Class name
            visualize {bool} -- True: View plot in pop-up
                                False: Save plot as image
        """
        p, r, t = self.getPRCurveForClass(classname)
        self.data = {"precisions": p, "recalls": r, "threshold_steps": t}
        
        self.annot.set_visible(False) 
        if visualize:
            self.plotCurve(self.data)
            self.figure.canvas.mpl_connect("motion_notify_event", self.hover)
            # time.sleep(0.5)
            self.figure.canvas.flush_events()
        else:
            self.plotCurve(self.data)
            self.figure.canvas.flush_events()
            plt.savefig(f"PR_Curve_{classname}.png")


class ClassWiseStats:
    """Accumulator class for Precision and Recall Stats(at a single value of confidence)
    """

    def __init__(self, classes_list):
        self.classes_list = classes_list
        self.class_wise_scores = {
            classname: {"tp": 0, "fp": 0, "fn": 0} for classname in classes_list
        }

    def addStats(self, classname, tp, fp, fn):
        """Accumulate the counts of stats relevant for P,R Calculation
        
        Arguments:
            classname {str} -- name of the class
            tp {int} -- True Positive Count
            fp {int} -- False Positive Count
            fn {int} -- False Negative Count
        """
        self.class_wise_scores[classname]["tp"] += tp
        self.class_wise_scores[classname]["fp"] += fp
        self.class_wise_scores[classname]["fn"] += fn

    def getPrecision(self, classname):
        """Calculate Precision for the given class
            Precision = TP/(TP+FP)

        Arguments:
            classname {str} -- class name(must be present in classes list)
        
        Returns:
            precision -- the precision of the class
        """
        tp = self.class_wise_scores[classname]["tp"]
        fp = self.class_wise_scores[classname]["fp"]
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = math.nan
        return precision

    def getRecall(self, classname):
        """Calculate Precision for the given class
            Recall = TP/(TP+FN)

        Arguments:
            classname {str} -- class name(must be present in classes list)
        
        Returns:
            recall -- the recall of the class
        """
        tp = self.class_wise_scores[classname]["tp"]
        fn = self.class_wise_scores[classname]["fn"]
        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        else:
            recall = math.nan
        return recall

    def getAllStats(self):
        """Return the Precision and Recall for all classes as a dictionary
            Returned dictionary has classnames as keys and another dict as the value
            the second level dictionary has the values in the keys - `Precision` and `Recall`
        
        Returns:
            class_wise_pr{dict} -- Dictionary with class wise pr
        """
        class_wise_pr = {}
        for c in self.classes_list:
            class_wise_pr[c] = {
                "Precision": self.getPrecision(c),
                "Recall": self.getRecall(c),
            }
        return class_wise_pr


class StarsEvaluator:
    """Custom Evaluator for Object Detection models
    Written for Team S.T.A.R.S - CMU MRSD Capstone.
    `github/t27`
    """

    def __init__(
        self,
        model,
        classes_list,
        train_json,
        val_json,
        batch_size=8,
        confidence_thres=0.7,
        iou_thres=0.5,
        model_input_format="BGR",
        progress_bar=True,
    ):
        self.train_dataset = self.get_data_dicts(train_json)
        self.test_dataset = self.get_data_dicts(val_json)
        self.model = model
        self.classes_list = classes_list
        self.batch_size = batch_size
        self.class_wise_stats = ClassWiseStats(self.classes_list)
        self.pr_curve_generator = ClassWisePRCurve(self.classes_list)
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.model_input_format = model_input_format
        self.progress_bar = progress_bar

    def getStatsSummary(self):
        return (
            self.class_wise_stats.getAllStats(),
            self.pr_curve_generator.getPRCurves(),
        )

    def plotPRCurveForClass(self, classname):
        self.pr_curve_generator.plotPRCurveForClass(classname)

    def evaluateAndGetTestNumbers(self):
        return self.evaluateDataset(self.test_dataset)

    def getTrainNumbers(self):
        return self.evaluateDataset(self.train_dataset)

    def evaluateDatasetIterative(self, dataset):
        for ob in dataset:
            image_path = ob["file_name"]
            annotations = ob["annotations"]

            cv_image = cv2.imread(image_path)
            # our model is currently trained using the opencv representation(default detectron)
            if self.model_input_format == "RGB":
                cv_image = cv_image[:, :, ::-1]

            image = torch.as_tensor(cv_image.astype("float32").transpose(2, 0, 1))
            gt_boxes = []
            gt_classes = []
            for anno in annotations:
                gt_boxes.append(anno["bbox"])
                gt_classes.append(self.classes_list[anno["category_id"]])

            # This function evaluates an image adds the relevant stats to the Stats and PRCurve objects
            det_boxes, det_scores, det_classes = self.evaluateSingleImage(
                image, gt_boxes, gt_classes
            )
            yield det_boxes, det_scores, det_classes, gt_boxes, gt_classes, cv_image

        # return (
        #     self.class_wise_stats.getAllStats(),
        #     self.pr_curve_generator.getPRCurves(),
        # )

    def evaluateDataset(self, dataset):
        if self.progress_bar:
            dataset_it = tqdm(dataset)
        else:
            dataset_it = dataset
        for ob in dataset_it:
            image_path = ob["file_name"]
            annotations = ob["annotations"]

            image = cv2.imread(image_path)
            # our model is currently trained using the opencv representation(default detectron)
            if self.model_input_format == "RGB":
                image = image[:, :, ::-1]

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            gt_boxes = []
            gt_classes = []
            for anno in annotations:
                gt_boxes.append(anno["bbox"])
                gt_classes.append(self.classes_list[anno["category_id"]])

            # This function evaluates an image adds the relevant stats to the Stats and PRCurve objects
            self.evaluateSingleImage(image, gt_boxes, gt_classes)

        return (
            self.class_wise_stats.getAllStats(),
            self.pr_curve_generator.getPRCurves(),
        )

        # TODO: add batch processing of the image

    def evaluateSingleImage(self, image, gt_boxes, gt_classes):
        """Evaluate a single image and get the numbers against GT
        
        Arguments:
            image {Tensor} -- torch Tensor of the image 
            gt_boxes {list} -- List of GT boxes
            gt_classes {list} -- List of classes that correspond to the gt_boxes
        """

        # for each image, make a separate
        # read image
        # compare annotation
        det_boxes, det_scores, det_classes = self.getDetections(image)
        # print(det_classes)
        class_wise_preds = defaultdict(list)
        class_wise_gt = defaultdict(list)
        for idx, classnum in enumerate(det_classes):
            classname = self.classes_list[classnum[0]]
            class_wise_preds[classname].append((det_boxes[idx], det_scores[idx][0]))

        for idx, classval in enumerate(gt_classes):
            class_wise_gt[classval].append(gt_boxes[idx])

        for c in self.classes_list:
            tp, fp, fn, pr_curve_data = self.getConfusionMatrixForClass(
                class_wise_gt[c], class_wise_preds[c]
            )
            self.class_wise_stats.addStats(c, tp, fp, fn)
            self.pr_curve_generator.addRawData(
                c, pr_curve_data["y_gt"], pr_curve_data["y_conf"]
            )
        return det_boxes, det_scores, det_classes

    def getConfusionMatrixForClass(self, gt, preds):

        """Calculate True Positives, False Positives and False Negatives
        This method doesn't consider class, so ensure the class wise filtering
        is done before. Only pass GT and Preds of the same class to this method
        
        Arguments:
            gt {list} -- Array of GT boxes
            preds {list of 2-tuples} -- Array of predicted boxes and their confidence

        Returns:
            [3-tuple] -- TruePositiveCount,FalsePositiveCount,FalseNegativeCount
        """

        # TODO: ensure the preds are NMS'ed so that we dont have overlapping predictions of the same class
        gt_boxes = copy.deepcopy(gt)

        tp, fp = ([], [])

        y_gt = []
        y_confs = []
        # for representing this as a binary classification problem(for PR Curves),
        # tp - pred = 1, gt=1
        # fp - pred = 1, gt=0
        # fn - pred = 0, gt=1

        for pred_box, conf in preds:
            # find the GT box with the highest IOU, remove that box
            # need a minimum of this iou to be considered useful
            max_iou = self.iou_thres
            max_iou_idx = None
            for idx, gt_box in enumerate(gt_boxes):
                iou = self.getIOU(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = idx
            if max_iou_idx is not None:
                del gt_boxes[max_iou_idx]
                # since we found a succesful match, this is a true positive
                if conf > self.confidence_thres:
                    tp.append(pred_box)
                # for PR Curve
                y_gt.append(1)
                y_confs.append(conf)
            else:
                if conf > self.confidence_thres:
                    fp.append(pred_box)
                y_gt.append(0)
                y_confs.append(conf)

        for z in gt_boxes:
            y_gt.append(1)
            y_confs.append(0)  # zero confidence

        pr_curve_data = {"y_gt": y_gt, "y_conf": y_confs}

        tp_count = len(tp)
        # number of predictions that didn't match any gt box ar false positives
        fp_count = len(fp)

        # number of gt boxes that didnt match a pred box are false negatives
        fn_count = len(gt_boxes)

        return tp_count, fp_count, fn_count, pr_curve_data

    def getIOU(self, box1, box2):
        """Calculate Intersection over Union
        
        Arguments:
            box1 {list} -- list of [xmin ymin xmax ymax]
            box2 {list} -- list of [xmin ymin xmax ymax]
        """
        # Intersection
        i_xmin = max(box1[0], box2[0])
        i_ymin = max(box1[1], box2[1])
        i_xmax = min(box1[2], box2[2])
        i_ymax = min(box1[3], box2[3])
        if i_xmax > i_xmin and i_ymax > i_ymin:
            intersection = (i_xmax - i_xmin) * (i_ymax - i_ymin)
        else:
            intersection = 0
            return 0

        def boxarea(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        union = boxarea(box1) + boxarea(box2) - intersection

        return intersection / union

    def getDetections(self, image):
        with torch.no_grad():
            c, h, w = image.shape
            inp = [{"image": image, "height": h, "width": w}]
            out = self.model(inp)[0]
            det_boxes = out["instances"].pred_boxes.tensor.to("cpu").numpy()
            det_scores = out["instances"].scores.to("cpu").unsqueeze(1).numpy()
            det_classes = out["instances"].pred_classes.to("cpu").unsqueeze(1).numpy()
            return det_boxes, det_scores, det_classes

    def get_data_dicts(self, json_file):
        data = json.load(open(json_file))

        for im in data:
            anno = im["annotations"]
            for an in anno:
                an["bbox_mode"] = BoxMode(an["bbox_mode"])
        return data


def prepareEvaluator(
    confidence_thres=0.7, iou_thres=0.5, chkpts_file="outputmodel_final.pth",
):
    classes_dict = {0: "vehicle", 1: "bike", 2: "pedestrian"}
    # # create cfg
    cfg = getEvalCfg()

    # pred = DefaultPredictor(cfg)

    model = build_model(cfg)  # building custom model, doesnt load chkpts automatically
    model.eval()
    DetectionCheckpointer(model).load(chkpts_file)

    evaluator = StarsEvaluator(
        model,
        list(classes_dict.values()),
        "stars_carla_train.json",
        "stars_carla_val.json",
        iou_thres=iou_thres,
        confidence_thres=confidence_thres,
    )

    return evaluator


def main():

    evaluator = prepareEvaluator(chkpts_file="final_training/output/model_final.pth")
    class_wise_results, pr_curves = evaluator.evaluateAndGetTestNumbers()
    json.dump(
        class_wise_results,
        open("class_wise_results.json", "w"),
        indent=2,
        cls=NumpyEncoder,
    )
    json.dump(
        pr_curves, open("pr_curves_raw_data.json", "w"), indent=2, cls=NumpyEncoder
    )
    ## Plot PR Curve


if __name__ == "__main__":
    main()


# Spec

# ---- for each image -------
# get gt boxes, and predicted boxes
#       put the gt boxes and predicted boxes in to class wise buckets
# for each class, conside confidence threshold as filter to calculate Positives
# use IOU threshold to identify true and false positives(decrement GT for every matched box), unmatched gt boxes are False negatives
