import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image

# tl_state_map = {-1:"off", 0:"red", 1:"yellow", 2:"green"}

def process_once(base_paths, data_type="train"):

    uid = 0
    labels = np.empty((0,), int)
    
    if (not isinstance(base_paths, list)):
        base_paths = [base_paths]
    
    for base_path in base_paths:

        clips_path = glob.glob(base_path + "/*")
        clips_path.sort()

        for clip_path in clips_path:

            ann_path = clip_path + "/frameAnnotationsBOX.csv"
            frame_base_path = clip_path + "/frames/"
            
            df = pd.read_csv(ann_path, sep=";") 

            for i in range(df.shape[0]):

                ann = df["Annotation tag"][i]
                
                if "go" == ann:
                    label = 2
                elif "warning" == ann:
                    label = 1
                elif "stop" == ann:
                    label = 0
                else:
                    label = -1
                    print(ann)

                if label < 0:
                    continue

                fp = df["Filename"][i].split("/")[-1]
                im_path = frame_base_path + fp
                img = cv2.imread(im_path)
                # print(im_path)
                s_x = df['Upper left corner X'][i]
                s_y = df['Upper left corner Y'][i]
                e_x = df['Lower right corner X'][i]
                e_y = df['Lower right corner Y'][i]

                c_x = (s_x + e_x) /2
                c_y = (s_y + e_y) /2
                c_x = int(c_x)
                c_y = int(c_y)
                side = int(max(e_y-s_y, e_x-s_x)/2)

                try:
                    save_path = "./processed_data/" + str(data_type) + "/frames/img_" + str(uid) + ".jpg"
                    # print(save_path)
                    cv2.imwrite(save_path, img[c_y-side:min(c_y+side, img.shape[0]), c_x-side:min(c_x+side, img.shape[1])])    
                    # cv2.imwrite(save_path, img[s_y:e_y, s_x:e_x])    
                    labels = np.append(labels, label)
                    uid += 1
                
                except:
                    pass
                
            
    np.save("./processed_data/" + str(data_type) + "/labels.npy", labels)
    

class TLDataset(Dataset):
    def __init__(self, base_path):
        self.labels = np.load(base_path + "labels.npy")
        
        self.preprocess = transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.data = []

        for i in range(self.labels.shape[0]):
            self.data.append(base_path + "frames/img_" + str(i) + ".jpg")
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        path = self.data[idx]
        
        img = Image.open(path)
        # img = cv2.imread(path)
        
        img = self.preprocess(img)
        return img, self.labels[idx]

class TLFineTuneDataset(Dataset):
    def __init__(self, base_path):
        
        self.data = []
        self.labels = []

        classes = glob.glob(base_path + "/*")
        classes.sort()

        for class_ in classes:
            l = int(class_.split("/")[-1])
            images = glob.glob(class_ + "/*" )
            for image in images:    
                self.data.append(image)
                self.labels.append(l)

        self.preprocess = transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        
        img = Image.open(path)
        # img = cv2.imread(path)
        
        img = self.preprocess(img)
        return img, self.labels[idx] 
    

if __name__ == "__main__":
    # process_once("./data/dayTrain", data_type="train")
    # process_once(["./data/daySequence"], data_type="test")
    # dataset = TLDataset("./processed_data/train/")

    dataset = TLFineTuneDataset("./new_dataset/")

    from IPython import embed; embed()
    