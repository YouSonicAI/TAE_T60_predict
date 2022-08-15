import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torchvision import transforms


class Dataset_dict(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=None,transform = None,start_freq = 0,end_freq = 29):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.freq_start = start_freq
        self.freq_end = end_freq
        self.dict_root = root_dir
        self.transform = transform
        self.dict_data = []
        #self.flag = "Val"
        #print("len of root dir:",len(os.walk(root_dir)))
        print("root dir:",root_dir)


        for (root, dirs, files) in os.walk(root_dir):
            print("root:",root,"dirs:",dirs,"len of files:",len(files))    

            self.dict_data =self.dict_data +  [os.path.join(root,filen) for filen in files]
            # self.root = root


    def __len__(self):
        return len(self.dict_data)

    def __getitem__(self, idx):
        images_list = []
        ddr_list = []
        t60_list = []
        MeanT60_list = []
        name_list = []
        data_dict = dict()
        dict_data_name = self.dict_data[idx]
        audio_image_dict = dict()
        try:
            #print(dict_data_root)
            audio_image_dict = torch.load(dict_data_name)
        except:
            print("failed read file name:",dict_data_name)
           

            audio_image_dict = torch.load("D:/TAE/TAE_Dataset/Train/TAE_Wav_Ouput_Pt/creswell-crags/creswell-crags_1_s_mainlevel_r_mainlevel_0_1_creswell-crags_东北话男声_1_TIMIT_a014_180_190_0dB-0.pt")
       
        dict_keys = list(audio_image_dict.keys())[0]
        dict_data = audio_image_dict[dict_keys]
        valid_info_count = 0
        for list_c in range(len(dict_data)):
            if dict_data[list_c] ==0 :
                continue
            else:
                valid_info_count+=1

                if "image" in dict_data[list_c].keys():
                    image_single = dict_data[list_c]['image']
                    tran = transforms.ToTensor()


                    images_125_totensor = torch.cat([tran(image_single[i].unsqueeze(0).cpu().numpy()) for i in range(6)],dim=0)
                    images_125_totensor = images_125_totensor - torch.mean(images_125_totensor,dim=-1).unsqueeze(2).repeat(1,1,160)
                    image = torch.div(images_125_totensor , torch.max(torch.abs(images_125_totensor),dim=-1).values.unsqueeze(2).repeat(1,1,160))
                    # image = torch.div(images_125_totensor,
                                    #   torch.std(images_125_totensor, dim=-1).unsqueeze(2).repeat(1, 1, 160))
                    images_list.append(image.transpose(0,1))


                    #images_list.append(dict_data[list_c]['image'])
                    t60_list.append(torch.unsqueeze(dict_data[list_c]['t60'][self.freq_start:self.freq_end],0))
                    ddr_list.append(torch.unsqueeze(dict_data[list_c]['ddr'][self.freq_start:self.freq_end],0))
                    MeanT60_list.append(torch.unsqueeze(dict_data[list_c]['MeanT60'],0))
                    name_list.append(dict_data_name)

        random_choose_num = np.random.randint(low=0,high = valid_info_count,size=1)[0]

        """
            2022.8.4
            images_list

        """
        images_list = [x.unsqueeze(0) for x in images_list]   # torch.Size[30,160]
      
        images = torch.cat(images_list,dim=0)
        ddr = torch.cat(ddr_list,dim=0)
        t60 = torch.cat(t60_list,dim=0)
        MeanT60 = torch.cat(MeanT60_list,dim=0)
       

        sample = {'image': images, 'ddr': ddr, 't60':t60, "MeanT60":MeanT60,"validlen":valid_info_count,'name':name_list}
       

        return sample

padding_keys = []
stacking_keys = ['MeanT60','ddr','image','t60']

def collate_fn(batch):
        keys = batch[0].keys()

        out = {k: [] for k in keys}

        for data in batch:
            for k,v in data.items():
                out[k].append(v)

        for k in stacking_keys:
            #print("key:",k)
            out[k] = torch.cat(out[k],dim=0)#torch.stack(out[k],dim=0)

        for k in padding_keys:
            out[k] = pad_sequence(out[k],batch_first = True)



        return out
       
       









