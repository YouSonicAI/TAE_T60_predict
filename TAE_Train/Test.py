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
import csv
import os
import glob
from new_data_load import Dataset_dict,collate_fn
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#device_ids = [0, 1]
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from mymodel_crnn_modify import Net
#from mymodel import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60
import argparse

torch.backends.cudnn.benchmark = True

def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--trained_epoch',type=int,default=0)

    parser.add_argument('--save_dir', type=str, default='save_model/CNN/train_acewithtimit_05262_freq_8_28')

    parser.add_argument('--model_path', type=str, default="/data2/wzd/t60_detection_cnn_2018-queenie_crnn_clean/t60_detection/save_model/CRNN/CRNN/compare_30_updata/t60_predict_model_18_fullT60_rir_timit_noise.pt")
    parser.add_argument('--test_path',type =str,default = "/data1/Train_Data_SameSpeaker/Test/EchoThief_timit_test")
    parser.add_argument('--load_pretrain', type=bool, default=True)

    parser.add_argument('--start_freq',type=int,default=8)
    parser.add_argument('--end_freq',type=int,default=28)
    #parser.add_argument('--epoch',type = int,default = 170)
    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path=None,trained_epoch=None,model=None,device=None):
    save_model = torch.load(checkpoint_path,map_location=device)
    #model_dict = model.state_dict()
    #state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

    #model_dict.update(state_dict)

    #model.load_state_dict(model_dict)
    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']
    #trained_epoch = state['epoch']
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

            h_n = torch.randn((1, 98, 20), device=device, dtype=torch.float32)
            h_c = torch.randn((1, 98, 20), device=device, dtype=torch.float32)

            #batch_size = len(valid_len)
            max_image_num = images.shape[1]
            images_reshape = torch.unsqueeze(images,dim=1)#torch.reshape(images, (images.shape[0] * images.shape[1], 1, images.shape[2], images.shape[3]))
            output_pts = net(images_reshape,h_n,h_c,valid_len,val_batch_size,max_image_num)

            # output_pts_extract = []
            # for oc in range(batch_size):
            #     output_split = output_pts[oc, 0:valid_len[oc], :]
            #     output_pts_extract.append(torch.reshape(output_split, (valid_len[oc], -1)))
            # output_pts_extract = torch.cat(output_pts_extract, dim=0)
            gt_t60_reshape = gt_t60

            loss = criterion(output_pts, gt_t60_reshape)
            bias = torch.sum((gt_t60_reshape - output_pts)) / output_pts.shape[0]
            #bias = torch.sum((gt_t60 - output_pts)) / output_pts.shape[0] / sum(valid_len)

            total_mean_loss += loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss/len(val_loader)
        mean_bias = float(total_mean_bias)/len(val_loader)
        if not writer is None:
            writer.add_scalar('val/mean_loss',mean_loss,epoch)
            writer.add_scalar('val/mean_bias',mean_bias,epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))

# def output_result_analysis(result_dict,csv_file):
#     margin = 0.2
#     larget_than_margin_csv = csv_file.split(".")[0]+"larger_than_0.2"+".csv"
# =======
def output_result_analysis(result_dict,output_dir,test_path):
    test_dataset = test_path.split("/")[-1]
    csv_file = os.path.join(output_dir,test_dataset+".csv")
    margin = 0.2
    larget_than_margin_csv = csv_file.split(".")[0] + "larger_than_%f" %(margin) + ".csv"

    f = open(csv_file,"w")
    csv_writer_normal = csv.writer(f)
    f_larger = open(larget_than_margin_csv,"w")
    larger_csv_writer = csv.writer(f_larger)



    for key,value in result_dict.items():
        #csv_writer.writerow([160,200,250,315,400,500,630,800,"1k","1.25k","1.6k","2k","2.5k","3.15k","4k","5k"\
                           #  "6.3k","8k","10k","12.5k","16k"])
        #csv_writer.writerow([str(key)])
        if result_dict[key]["gt"][5] < result_dict[key]["gt"][11] - 0.2:
            csv_writer = larger_csv_writer
        else:
            csv_writer = csv_writer_normal
        csv_writer.writerow([160,200,250,315,400,500,630,800,"1k","1.25k","1.6k","2k","2.5k","3.15k","4k","5k",\
                             "6.3k","8k","10k","12.5k","16k"])
        csv_writer.writerow([str(key)])

        for k in range(len(result_dict[key]["output_list"])):
            csv_writer.writerow(["output%d" %(k)]+result_dict[key]["output_list"][k].cpu().numpy().tolist())
        csv_writer.writerow(["mean_output"]+result_dict[key]["mean_output"].cpu().numpy().tolist())
        csv_writer.writerow(["gt"]+result_dict[key]["gt"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_bias"]+result_dict[key]["mean_bias"].cpu().numpy().tolist())

        csv_writer.writerow(["mean_mse"]+(result_dict[key]["mean_bias"]**2).cpu().numpy().tolist())

        csv_writer.writerow([])




def test_net(net,epoch,val_loader,writer):
    result_dict = dict()
    with torch.no_grad():
        net.eval()
        total_mean_loss = torch.zeros( (1,20))
        total_mean_bias = torch.zeros((1,20))
        progress_bar = tqdm(val_loader)
        useless_count = 0
        for j,datas in enumerate(progress_bar):

            meanT60 = datas['MeanT60'].to(torch.float32).to(device)
            images = datas['image'].to(torch.float32).to(device)
            gt_t60 = datas['t60'].to(torch.float32).to(device)
            valid_len = datas['validlen']
            names = datas['name']

            h_n = torch.randn((1, 98, 20), device=device, dtype=torch.float32)
            h_c = torch.randn((1, 98, 20), device=device, dtype=torch.float32)

            #batch_size = len(valid_len)
            max_image_num = images.shape[1]
            images_reshape = torch.unsqueeze(images,dim=1)#torch.reshape(images, (images.shape[0] * images.shape[1], 1, images.shape[2], images.shape[3]))
            output_pts = net(images_reshape,h_n,h_c,valid_len,val_batch_size,max_image_num)

            # output_pts_extract = []
            # for oc in range(batch_size):
            #     output_split = output_pts[oc, 0:valid_len[oc], :]
            #     output_pts_extract.append(torch.reshape(output_split, (valid_len[oc], -1)))
            # output_pts_extract = torch.cat(output_pts_extract, dim=0)
            gt_t60_reshape = gt_t60
            bias = gt_t60_reshape - output_pts
            rsquare_error = torch.sqrt(bias**2)
            abs_bias = torch.abs(gt_t60_reshape - output_pts)
            mean_output_list = []

            if not torch.isnan(rsquare_error).all():
                total_mean_loss += torch.mean(rsquare_error,dim=0).cpu().detach()
                total_mean_bias += torch.mean(bias,dim=0).cpu().detach()
            else:
                useless_count+=1



            for i in range(len(valid_len)):
                start_num = 0
                if i>0:
                    start_num = sum(valid_len[0:i])

                output_list = [output_pts[k]  for k in range(start_num,start_num+valid_len[i])]
                mean_output = torch.mean(output_pts[start_num:start_num+valid_len[i]],dim=0)
                mean_abs_bias = torch.mean(abs_bias[start_num:start_num+valid_len[i]],dim=0)
                mean_bias = torch.mean(bias[start_num:start_num+valid_len[i]],dim=0)
                mean_rsquare_error = torch.mean(rsquare_error[start_num:start_num+valid_len[i]],dim=0)
                mean_gt = torch.mean(gt_t60_reshape[start_num:start_num+valid_len[i]],dim=0)
                result_dict[names[i][0]] = {"mean_output":mean_output,"mean_bias":mean_bias,"gt":mean_gt,"square_error":mean_rsquare_error,"output_list":output_list}


        mean_loss = total_mean_loss/ (len(val_loader)-batch_size*useless_count)
        mean_bias = total_mean_bias/(len(val_loader)-batch_size*useless_count)
        if not writer is None:
            writer.add_scalar('val/mean_loss',mean_loss,epoch)
            writer.add_scalar('val/mean_bias',mean_bias,epoch)

        print("Mean loss:",mean_loss)
        print("Mean bias:",mean_bias)
        #print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))

        return result_dict





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
                                         step_size=20,
                                         gamma=0.1,last_epoch=start_epoch)

    print("lr at beginning:",lr.get_last_lr())
    net.train()

    lr_list = list()
    nn = 0
    #加载预训练权重
    # checkpoint = torch.load("/data2/cql/code/cnnLstmPredictT60/Checkpoints/cnnLstm/t60_predict_model_99_meanT60.pt", map_location=device)
    # net.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr.load_state_dict(checkpoint['lr'])
    # start_epoch = checkpoint['epoch'] + 1
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
            gt_t60 = datas['t60'].to(torch.float32).to(device)
            #print("gt_t60")
            valid_len = datas['validlen']

            h_n = torch.randn((1, 98, 20), device=device, dtype=torch.float32)
            h_c = torch.randn((1, 98, 20), device=device, dtype=torch.float32)
            #images = torch.unsqueeze(images,2)
            batch_size = images.shape[0]
            max_image_num = images.shape[1]
            images_reshape = torch.unsqueeze(images, dim=1)
            #images_reshape = #torch.reshape(images,(images.shape[0]*images.shape[1],1,images.shape[2],images.shape[3]))
            output_pts = net(images_reshape,h_n,h_c,valid_len,batch_size,max_image_num) #需要确认维度变化后对应数值是否是对的
            # output_pts_extract = []
            # for oc in range(batch_size):
            #     output_split = output_pts[oc,0:valid_len[oc],:]
            #     output_pts_extract.append(torch.reshape(output_split,(valid_len[oc],-1)))
            # output_pts_extract = torch.cat(output_pts_extract,dim=0)

            gt_t60_reshape = gt_t60#.reshape(images_reshape.shape[0],-1)
            loss = criterion(output_pts, gt_t60_reshape)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=0.1, norm_type=2)
            # for name, parms in net.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #           ' -->grad_value:', torch.mean(parms.grad))

            optimizer.step()
            bias =  torch.sum((gt_t60_reshape -output_pts))/output_pts.shape[0]#/sum(valid_len)
            total_mean_loss = total_mean_loss +  loss.item()
            total_mean_bias = total_mean_bias+bias.item()

        mean_loss = total_mean_loss/len(train_loader)
        mean_bias = total_mean_bias/len(train_loader)
        if not writer is None:
            writer.add_scalar('train/mean_loss', mean_loss, epoch)
            writer.add_scalar('train/mean_bias',mean_bias,epoch)
            writer.add_scalar('lr/lr',lr.get_last_lr()[-1],epoch)
        print("In training, epoch {},mse is {},bias is {}".format(epoch,mean_loss,mean_bias))
        lr.step()
        if epoch%1 == 0:
            print("eval")
            val_net(net,epoch,val_loader,writer)
            net.train()
            #val = Val_meanT60()
            #val_loss,val_mse = val(writer,epoch, net, val_loader, lr.get_last_lr(),device)
            #print("in val,val_loss is {},val_mse is {}".format(val_loss,val_mse))

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
    LOAD_PRETRAIN = args.load_pretrain   #False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path =  args.model_path
    #"/data2/queenie/download/code/t60_detection_cnn_2018/t60_detection/save_model/CRNN/2022-05-17-21-09-30/t60_predict_model_7_fullT60_rir_timit_noise.pt"#"/data2/queenie/download/code/t60_detection_cnn_2018/t60_detection/save_model/CRNN/2022-05-17-16-24-28/t60_predict_model_6_fullT60_rir_timit_noise.pt"#"/data2/queenie/download/code/t60_detection_cnn_2018/t60_detection/save_model/CRNN/2022-05-17-13-36-43/t60_predict_model_0_fullT60_rir_timit_noise.pt"#"/Users/queenie/Documents/t60_detection_cnn_2018/t60_detection/save_model/CRNN/t60_predict_model_49_fullT60_rir_timit_noise.pt"
    #"/Users/queenie/Documents/t60_detection_cnn_2018/t60_detection/save_model/CRNN/2022-05-11-18-33-19/2022-05-11-18-34-27t60_predict_model_299_fullT60_rir_timit_noise.pt"#"Checkpoint/t60_predict_model_99_fullT60_rir_timit_noise.pt"#"/data2/queenie/download/code/t60_detection_cnn_2018/t60_detection/Checkpoints/rir_timit_noise_new/t60_predict_model_99_fullT60_rir_timit_noise.pt"#"/Users/queenie/Documents/t60_detection_cnn_2018/t60_detection/model_0505_ql/t60_predict_model_0_fullT60_rir_timit_noise.pt"

    net = Net(ln_out=args.end_freq - args.start_freq)
    if LOAD_PRETRAIN == True:
        start_time = time.time()
        net,trained_epoch = load_checkpoint(model_path,99, net, device)
        print('Successfully Loaded model: {}'.format(model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time() - start_time))
    else:
        trained_epoch = 0
    net.to(device)

    #net.to(device)
    print(net)
    print(net)
#    print(next(net.parameters()).device)

    val_batch_size = 200
    # val_batch_size = 1
    criterion = torch.nn.MSELoss()
    #optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':0.0001}], lr=0.0001,weight_decay=0.0001)
    n_epochs = 300
    data_transform = transforms.Compose([ToTensor()])
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num,"Trainable":trainable_num}
    print(get_parameter_number(net))

    if DEBUG == 1:
        val_dict_root = "/Users/queenie/Documents/ace_dict_data/ACE"
        batch_size = 2
    else:

        val_dict_root = args.test_path #"/data4_ssd/Ace_dict_data/ACE_eval"  # "/data4_ssd/dict_eval"#"/data2/cql/code/augu_data/Dataset
        batch_size = 350


    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root,transform=data_transform,start_freq=args.start_freq,end_freq = args.end_freq)
    print("len of val dataset:",len(val_transformed_dataset))
    #print('Number of images: ', len(transformed_dataset))
    if DEBUG == 1:

        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,shuffle=False,num_workers=0,
                                                   batch_size=val_batch_size, drop_last=True,
                                                   collate_fn=collate_fn)
    else:

        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                   shuffle=True, num_workers=4,
                                                   batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                   collate_fn=collate_fn)
    print("after train loader init")
    trained_epoch = args.trained_epoch
    #train_net(trained_epoch,n_epochs, train_loader,val_loader,batch_size,args)
    test_output_result = test_net(net,trained_epoch,val_loader,None)
    outputresult_dir = "output_analysis_crnn_onlyour_0605_mse"
    if not os.path.exists(outputresult_dir):
        os.makedirs(outputresult_dir)
    #test_output_csv = os.path.join(outputresult_dir,"test.csv")
    output_result_analysis(test_output_result,outputresult_dir,args.test_path)


    #TODO 训练时要改dict，权重名字，验证时保存的文件名和路径
