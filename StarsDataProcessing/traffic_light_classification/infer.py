import numpy as np
import cv2
import tqdm
from torchvision import transforms
import torch
# from torchsummary import summary
from .model import TLClassification
from PIL import Image
import pandas as pd

path_prefix = "../StarsDataProcessing/traffic_light_classification/"

def test(model, device, frame, preprocess):

    model.eval()
    
    with torch.no_grad():

        frame = preprocess(frame).to(device)
        frame = frame.unsqueeze(0)
        output = model(frame)

        pred = torch.argmax(output)
        pred = pred.cpu().numpy()
        
    return pred
           
def load_model(device):
    
    model = TLClassification()
    # print(model)
    model = model.to(device)
    # summary(model, (3, 40, 40) )
    # model = nn.DataParallel(model)
    model_path = path_prefix + "checkpoints/final_model.h5"
    model.load_state_dict(torch.load(model_path))

    return model

def get_TL_status(video_path, bboxes, csv_save_path, second_TL=False, debug=False):
        
    tl_state_map = {0:"red", 1:"yellow", 2:"green"}

    preprocess = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(device)

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width  = cap.get(3) # float
    height = cap.get(4) # float

    if debug:
        out = cv2.VideoWriter(path_prefix + 'debug/last_vid.mp4', 
            cv2.VideoWriter_fourcc('M'
            ,'J','P','G'), 30.0, (int(width),int(height)))

    assert len(bboxes) == 2 or len(bboxes) == 4
    
    # if len(bboxes) == 2:
    #     bboxes = bboxes * 2

    if second_TL:
        offset = 2
    else:
        offset = 0

    tl_id = []
    frame_ids = []
    states = []

    count = -1
    for frame_id in tqdm.tqdm(range(length)):
        
        ret, frame = cap.read()

        if not ret:
            break
        
        if debug:
            frame_debug = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        for tid, bbox in enumerate(bboxes):

            c_x = bbox[0] + ( bbox[2]/2 )
            c_y = bbox[1] + ( bbox[3]/2 )
            c_x = int(c_x)
            c_y = int(c_y)
            side = int(min(bbox[2], bbox[3]))

            data_frame = frame.crop((c_x-side, c_y-side, c_x+side, c_y+side)) 
            count += 1
            pred = test(model, device, data_frame, preprocess)
            
            if pred == 2:
                color_class = (0, 255, 0)
            elif pred == 1:
                color_class = (0, 255, 255)
            elif pred == 0:
                color_class = (0, 0, 255)
            
            if debug:
                cv2.rectangle(frame_debug, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color_class, 5)
            
            tl_id.append(tid + offset)
            frame_ids.append(frame_id+1)
            states.append(tl_state_map[int(pred)])
        
        if debug:    
            # cv2.imshow('frame', frame_debug)
            out.write(frame_debug)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
    cap.release()
    
    if debug:
        out.release()
        cv2.destroyAllWindows()

    d = {'frame_id': frame_ids, 'tl_id': tl_id, 'tl_state': states}
    df = pd.DataFrame(data=d)
    df.to_csv(csv_save_path)


def main():
    
    tl_state_map = {0:"red", 1:"yellow", 2:"green"}

    preprocess = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(device)

    cap = cv2.VideoCapture("./inputs/tl_violation.MP4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width  = cap.get(3) # float
    height = cap.get(4) # float
    out = cv2.VideoWriter('outputs/tl_violation_processed.mp4', 
        cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (int(width),int(height)))

    tl_id_0_0 = (2908, 759, 36, 67)
    tl_id_0_1 = (3141, 742, 37, 72)
    tl_id_1_0 = (1167, 783, 35, 73)
    tl_id_1_1 = (1314, 790, 32, 68)

    bboxes = [tl_id_0_0, tl_id_1_0, tl_id_0_1, tl_id_1_1]
    
    tl_id = []
    frame_ids = []
    states = []

    count = 0
    for frame_id in range(length):
        
        ret, frame = cap.read()

        if not ret:
            break

        frame_debug = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        for tid, bbox in enumerate(bboxes):

            c_x = bbox[0] + ( bbox[2]/2 )
            c_y = bbox[1] + ( bbox[3]/2 )
            c_x = int(c_x)
            c_y = int(c_y)
            side = int(min(bbox[2], bbox[3]))

            data_frame = frame.crop((c_x-side, c_y-side, c_x+side, c_y+side)) 
            count += 1
            pred = test(model, device, data_frame, preprocess)
            
            if pred == 2:
                color_class = (0, 255, 0)
            elif pred == 1:
                color_class = (0, 255, 255)
            elif pred == 0:
                color_class = (0, 0, 255)
            
            cv2.rectangle(frame_debug, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color_class, 5)
            
            # if tid < 2:    
            tl_id.append(tid)
            frame_ids.append(frame_id+1)
            states.append(tl_state_map[int(pred)])
        
        # for j in range(2):
        #     tl_id.append(j+2)
        #     frame_ids.append(frame_id+1)
            # states.append(tl_state_map[0])
    
        cv2.imshow('frame', frame_debug)
        out.write(frame_debug)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    d = {'frame_id': frame_ids, 'tl_id': tl_id, 'tl_state': states}
    df = pd.DataFrame(data=d)
    df.to_csv("outputs/tl_violation.csv")

    # from IPython import embed; embed()

if __name__ == '__main__':
    main()
