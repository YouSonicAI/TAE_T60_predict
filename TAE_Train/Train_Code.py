import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

import torchvision.transforms

#from workspace_utils import active_session
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')
import valdata_meant60
from valdata_meant60 import Val_meanT60
from torch.utils.tensorboard import SummaryWriter
# 决定使用哪块GPU进行训练
import os
import glob
from new_data_load import Dataset_dict,collate_fn
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#device_ids = [0, 1]
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from TAE_Model import ACE_Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60
import argparse
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--trained_epoch',type=int,default=0)

    parser.add_argument('--save_dir', type=str, default='D:/TAE/TAE_Train/save_model')

    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--start_freq',type=int,default=7) # 8
    parser.add_argument('--end_freq',type=int,default=22) # 28

    parser.add_argument('--image',type=int,default=3) # 0 1 2 3 4 5 
    parser.add_argument('--gt',type=int,default=9)   # 0 3 6 9  12  15 


    args = parser.parse_args()
    return args

## The code is used to predict T60 among 125  250 500 1000 2000 4000
## You only modify the image and gt of the parse_args function

def load_checkpoint(checkpoint_path=None,trained_epoch=None,model=None,device=None):
    save_model = torch.load(checkpoint_path,map_location=device)

    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']

    print('model loaded from %s' % checkpoint_path)
    return_epoch = 0
    if not trained_epoch is None:
        return_epoch = trained_epoch
    else:
        return_epoch = trained_epoch_load

    return model,return_epoch


def net_sample_output():
    for i, sample in enumerate(val_loader):

        images = sample['image']
        ddr = sample['ddr']
        t60 = sample['t60']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.tensor(images.clone().detach(), dtype=torch.float32, device=device)

        # forward pass to get net output
        t60_predict = net(images)

        # break after first image is tested
        if i == 0:
            return images, t60_predict, t60

def val_net(net,epoch,val_loader,writer):
    with torch.no_grad():
        net.eval()
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(val_loader)
        for j,datas in enumerate(progress_bar):

            meanT60 = datas['MeanT60'].to(torch.float32).to(device)
            images = datas['image'].to(torch.float32).to(device)
            gt_t60 = datas['t60'].to(torch.float32).to(device)
            valid_len = datas['validlen']

         
            

            images = images.squeeze(1)
            feature = images.transpose(0,1)[args.image].unsqueeze(1)


            
            output_pts = net(feature.to(device))
            print("output_pts", output_pts)



            
            gt_t60_reshape = datas['t60'].to(torch.float32).to(device)
            
            gt_t60 = gt_t60_reshape.transpose(0, 1)[args.gt].unsqueeze(1)

           


            loss = criterion(output_pts, gt_t60)
            bias = torch.sum((gt_t60 - output_pts)) / output_pts.shape[0]
           

            total_mean_loss += loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss/len(val_loader)
        mean_bias = float(total_mean_bias)/len(val_loader)
        writer.add_scalar('val/mean_loss',mean_loss,epoch)
        writer.add_scalar('val/mean_bias',mean_bias,epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))






def train_net(start_epoch,n_epochs,train_loader,val_loader,batch_size,args):
    dt = datetime.datetime.now()
    model_dir = args.save_dir
    #model_dir = os.path.join("save_model/CRNN","train_0518_1133")
    #model_dir = os.path.join("save_model/CRNN/", dt.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #start_epoch=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = torch.optim.lr_scheduler.StepLR(optimizer,
                                         step_size=30,
                                         gamma=0.1,last_epoch=start_epoch)

    print("lr at beginning:",lr.get_last_lr())
    net.train()

    lr_list = list()
    nn = 0

    print("the training process from epoch{}...".format(start_epoch))
    dt = datetime.datetime.now()

    log_dir = os.path.join("./log",dt.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    for epoch in range(start_epoch,n_epochs):
        print("Training:epoch ", epoch,"lr:",lr.get_last_lr()[-1])
        lr_list.append(lr.get_last_lr())
        total_mean_loss = 0
        total_mean_bias = 0
        #计算做了几个batch，防止最后那部分数据没有做反向传播
        c_batch = 0
        progress_bar = tqdm(train_loader)

        #k,是字典名字，datas要么是dict,要么是0
        for batch_i, datas in enumerate(progress_bar):#tqdm(enumerate(train_loader)):

            meanT60 = datas['MeanT60'].to(torch.float32).to(device)
            images =datas['image'].to(torch.float32).to(device)
            

            # print("images_shape", images.shape)
            images = images.squeeze(1)
            feature = images.transpose(0,1)[args.image].unsqueeze(1)
            # print("images_1000.shape", images_2000.shape)


            # print(" images_1000 ",  images_1000 )
            
            

            
           
            output_pts = net(feature) #需要确认维度变化后对应数值是否是对的

            print("output_pts",  output_pts)


            gt_t60_reshape = datas['t60'].to(torch.float32).to(device)
               
            gt_t60 = gt_t60_reshape.transpose(0, 1)[args.gt].unsqueeze(1)

           

            loss = criterion(output_pts, gt_t60)
            print("loss", loss)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=0.1, norm_type=2)
          

            optimizer.step()
            bias =  torch.sum((gt_t60 -output_pts))/output_pts.shape[0]#/sum(valid_len)
            total_mean_loss = total_mean_loss +  loss.item()
            total_mean_bias = total_mean_bias+bias.item()

        mean_loss = total_mean_loss/len(train_loader)
        mean_bias = total_mean_bias/len(train_loader)
        writer.add_scalar('train/mean_loss', mean_loss, epoch)
        writer.add_scalar('train/mean_bias',mean_bias,epoch)
        writer.add_scalar('lr/lr',lr.get_last_lr()[-1],epoch)
        print("In training, epoch {},mse is {},bias is {}".format(epoch,mean_loss,mean_bias))
        lr.step()
        if epoch%1 == 0:
            print("eval")
            val_net(net,epoch,val_loader,writer)
            net.train()

        if True:

            #'./Checkpoints/rir_timit_noise_new_0509_alldata/'
            model_name = 't60_predict_model_%d_fullT60_rir_timit_noise.pt' % (epoch)

            # after training, save your model parameters in the dir 'saved_models'
            state = {"model":net.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,
                     "lr":lr.state_dict()}
            torch.save(state, os.path.join(model_dir, model_name))
            print('Finished Training')


    writer.close()

class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.warmup = False
        self.warmup_epochs = 10
        self.min_lr = min(lr)
        self.max_lr = max(lr)
        #self.lr = lr
        #self.lr_epochs = lr_epochs

        #self.lr_warmup = lambda epoch_in : min(self.lr)+0.5*(max(self.lr) - min(self.lr))*(1+np.cos((epoch_in-self.warmup_epochs)*PI/(2*self.warmup_epochs)))
        self.lr_warmup = lambda epoch_in: self.min_lr + 0.5*(self.max_lr - self.min_lr)*(1+np.cos((epoch_in - self.warmup_epochs)*PI/(2*self.warmup_epochs)))
        if self.warmup == True:
            self.lr_epochs = [self.warmup_epochs] + [i+self.warmup_epochs for i in lr_epochs]
            self.lr = [self.lr_warmup] + lr
        else:
            self.lr_epochs = lr_epochs
            self.lr = lr

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        if self.warmup == True and idx ==0:
            return  self.lr[idx](epoch)
        else:
            return self.lr[idx]


if __name__ == "__main__":


    args = parse_args()

    DEBUG = 0
    LOAD_PRETRAIN = args.load_pretrain

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path =  args.model_path

    net = ACE_Net()

    ### 这个判断的作用，用来计时
    if LOAD_PRETRAIN == True:
        start_time = time.time()
        net,trained_epoch = load_checkpoint(model_path,99, net, device)
        print('Successfully Loaded model: {}'.format(model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time() - start_time))
    
    else:
        trained_epoch = 0
    net.to(device)

    print(net)



    print("args.gt",args.gt)

    print("args.image", args.image)



    val_batch_size = 500
    criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':0.0001}], lr=0.0001,weight_decay=0.0001)
    optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':1e-3}], lr=0.0001,weight_decay=0.0001)
    n_epochs = 300
    data_transform = transforms.Compose([ToTensor()])
    # 网络参数数量的作用
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num,"Trainable":trainable_num}
    print(get_parameter_number(net))

    #  数据集地址
    ### 数据集地址为啥要用if语句，分两个分支 而且批次也不一样 
    if DEBUG == 1:
        # train_dict_root ="/Users/queenie/Documents/ace_dict_data/ACE"
        # val_dict_root = "/Users/queenie/Documents/ace_dict_data/ACE"
        # train_dict_root ="/data1/Train_Data_SameSpeaker/exp2"
        # val_dict_root = "/data1/Train_Data_SameSpeaker/Eval/ACE_eval"
        batch_size = 2
    else:

        train_dict_root = "D:/TAE/TAE_Dataset/Train/TAE_Wav_Ouput_Pt"
        val_dict_root = "D:/TAE/TAE_Dataset/Val"
        batch_size = 600
        # batch_size = 1

    print("train_dir:",train_dict_root)
    ### 调用new_data_load.py文件
    ### start_freq=args.start_freq 8和nd_freq = args.end_freq 28的作用

    train_transformed_dataset = Dataset_dict(root_dir=train_dict_root,transform=None,start_freq=args.start_freq,end_freq = args.end_freq)
    


    print("len of train dataset:",len(train_transformed_dataset))
    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root,transform=None,start_freq=args.start_freq,end_freq = args.end_freq)
    
    print("len of val dataset:",len(val_transformed_dataset))
    if DEBUG == 1:
        train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                                   shuffle=False, num_workers=0,
                                                   batch_size=batch_size, drop_last=True,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,shuffle=False,num_workers=0,
                                                   batch_size=val_batch_size, drop_last=True,
                                                   collate_fn=collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                  shuffle=True,num_workers=6,
                                  batch_size=batch_size,drop_last=True,#prefetch_factor=batch_size,
                                                  collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                   shuffle=True, num_workers=4,
                                                   batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                   collate_fn=collate_fn)
    print("after train loader init")
    trained_epoch = args.trained_epoch
    train_net(trained_epoch, n_epochs, train_loader, val_loader, batch_size,args)


    #TODO 训练时要改dict，权重名字，验证时保存的文件名和路径
