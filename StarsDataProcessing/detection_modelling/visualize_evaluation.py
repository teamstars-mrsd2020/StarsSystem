import cv2
import time
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

# custom imports
import evaluation
import convert_to_detectron2

# alias SVD1="cd /home/stars/Code/carla_utils/training && conda activate detector && python visualize_evaluation.py"

## TUNABLE Parameters
confidence_thres = 0.70  # 70% (increasing this increases precision, decreases recall)
iou_thres = 0.5  # 50% (increasing this increases precision and decreases recall)
PLOT_PR_CURVE = True
##
# dataset_folder = "/home/stars/Code/carla_utils/new_output_test_data/19 April/SVD_Data/round_2"
dataset_folder = (
    "/home/stars/Code/carla_utils/new_output_test_data/19 April/SVD_Data/round_5"  # 11
)
# folder with raw images and annotations
checkpoints_path = (
    "/home/stars/Code/carla_utils/training/final_training/output/model_final.pth"
)
def main():
    # only for Adhoc testing, use the pytest runner in the StarsTest for actual tests
    run(dataset_folder, checkpoints_path, confidence_thres, iou_thres, PLOT_PR_CURVE)

def stars_test_1(cfg):
    # cfg: the Stars Run Config file, containing all the information including the run id and the paths etc
    checkpoints_path = "./final_training/output/model_final.pth"
    dataset_folder = cfg["detection_gt"]
    run(dataset_folder,checkpoints_path)

def run(
    dataset_folder,
    checkpoints_path,
    confidence_thres=0.70,
    iou_thres=0.5,
    PLOT_PR_CURVE=True,
):
    import ipdb;ipdb.set_trace()
    evaluator = evaluation.prepareEvaluator(
        confidence_thres=confidence_thres,
        iou_thres=iou_thres,
        chkpts_file=checkpoints_path,
    )
    round_x_dataset = convert_to_detectron2.adhocConvertFolder(dataset_folder)
    round_x_dataset = round_x_dataset[:300]
    t_prev = time.time()
    fps = 0
    num_frames = len(round_x_dataset)
    pbar = pbar = tqdm(total=len(round_x_dataset))

    for result in evaluator.evaluateDatasetIterative(round_x_dataset):

        det_boxes, det_scores, det_classes, gt_boxes, gt_classes, cv_image = result

        det_classes = [evaluator.classes_list[x[0]] for x in det_classes]

        thres_boxes = []
        thres_classes = []

        for box, score, cls in zip(det_boxes, det_scores, det_classes):
            if score > confidence_thres:
                thres_boxes.append(box)
                thres_classes.append(cls)

        gt_labels = [f"gt_{label}" for label in gt_classes]
        viz = Visualizer(cv_image, None)
        blacks = ["black"] * len(gt_labels)
        viz.overlay_instances(boxes=gt_boxes, assigned_colors=blacks)
        viz.overlay_instances(boxes=thres_boxes, labels=thres_classes)
        out = viz.output.get_image()
        # TODO: Color GT in black and add real time stats(class wise precision, recall and FPS/ETA) as text on the frame(sky)

        if fps > 0:
            class_wise_stats, class_wise_pr_curves = evaluator.getStatsSummary()
            p_vehicle = class_wise_stats["vehicle"]["Precision"]
            r_vehicle = class_wise_stats["vehicle"]["Recall"]

            if PLOT_PR_CURVE and pbar.n > (num_frames * 0.5):
                evaluator.plotPRCurveForClass("vehicle")

            # show_str = f"Vehicle: Precision = {100*p_vehicle:.2f}%; Recall = {100*r_vehicle:.2f}%;                                                      FPS={fps:.2f}"
            show_str = f"Vehicle: Precision = {100*p_vehicle:.2f}%; Recall = {100*r_vehicle:.2f}%; FPS={fps:.2f}"

            cv2.putText(
                img=out,
                text=show_str,
                org=(0, 100),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=2,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            print(
                f"Vehicle: Precision = {100*p_vehicle:.2f}%; Recall = {100*r_vehicle:.2f}%;"
            )
            cv2.namedWindow("Stars Detector Evaluation", cv2.WINDOW_NORMAL)
            # Window positioning
            windowscale = 0.8
            win_w = int(1920 * windowscale)
            win_h = int(1080 * windowscale)
            cv2.resizeWindow("Stars Detector Evaluation", (win_w, win_h))
            winx = int((1920 - (win_w)) / 2)  # center the window
            winy = 10
            cv2.moveWindow("Stars Detector Evaluation", winx, winy)
            cv2.imshow("Stars Detector Evaluation", out)

            cv2.waitKey(delay=10)

        t = time.time()
        dt = t - t_prev
        fps = 1 / dt
        t_prev = t
        # # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        #     break
        pbar.update(1)

    cv2.waitKey(0)
    pbar.close()


if __name__ == "__main__":
    # main()
    import os
    os.chdir("/home/StarsSystem/StarsDataProcessing")
    import json
    run_id = 1
    with open("../data/runs/run_" + str(run_id) + ".json") as f:
        cfg = json.load(f)
    # def stars_test_1(cfg):
    #     # cfg: the Stars Run Config file, containing all the information including the run id and the paths etc
    checkpoints_path = "../StarsDataProcessing/detection_modelling/final_training/output/model_final.pth"
    dataset_folder = cfg["detection_gt"]
    run(dataset_folder,checkpoints_path)
