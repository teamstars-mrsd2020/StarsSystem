import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random 

import tqdm

from dataset import *
from model import *

import wandb

from torchsummary import summary

run = wandb.init(project="traffic")

config = wandb.config
config.batch_size = 128
config.test_batch_size = config.batch_size
config.epochs = 50
config.lr = 0.0001
config.no_cuda = False
config.seed = 320
config.log_interval = 10
config["num_workers"] = 24
wandb.save("model.py")

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader, total=int(len(train_loader)))):
        
        data, target = data.float().to(device), target.to(device)

        optimizer.zero_grad()
        
        output = model(data)

        loss = criterion(output, target)
        
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * data.shape[0]
    
    training_loss = running_loss / len(train_loader.dataset)
    
    log_obj = {"loss": training_loss, "epoch": epoch}
    print(log_obj)
    wandb.log(log_obj)

def test(args, model, device, test_loader, criterion):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(test_loader, total=int(len(test_loader)))):
            
            data, target = data.float().to(device), target.to(device)
            
            output = model(data)

            test_loss += (criterion(output, target).item() * data.shape[0])
          
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    log_obj = {
        "Dev Accuracy": 100. * correct / len(test_loader.dataset),
        "Dev Loss": test_loss / len(test_loader.dataset),
    }
    
    print(log_obj)
    wandb.log(log_obj)
    


def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if use_cuda else {}
    
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)
    # torch.backends.cudnn.deterministic = True

    # Train
    train_dataset = TLFineTuneDataset("./new_dataset/")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, drop_last=True, **kwargs)

    # Dev
    # dev_dataset = TLDataset("./processed_data/test/")
    # dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config.test_batch_size, **kwargs)

    print("Data Loading Complete.")

    print("Training size: " , len(train_loader.dataset))
    # print("Dev size: " , len(dev_loader.dataset))

    model = TLClassification()
    print(model)
    model = model.to(device)
    summary(model, (3, 40, 40) )
    
    # model = nn.DataParallel(model)
    model_path = "./checkpoints/final_model.h5"
    model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.75)
    
    wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        
        wandb.log({"lr": scheduler.get_last_lr()})
        
        train(config, model, device, train_loader, optimizer, epoch, criterion)
        # test(config, model, device, dev_loader, criterion)
        scheduler.step()
        
        model_name = "checkpoints/model_epoch_" + str(epoch) + "_" + str(run.name) + ".h5"
        torch.save(model.state_dict(), model_name)
        wandb.save(model_name)

if __name__ == '__main__':
    main()