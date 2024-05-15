import random
import numpy as np
import torch
import rasterio as rs
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
import glob
from torchvision.transforms import GaussianBlur
import re

class CogDataset_v46(Dataset):
    def __init__(self, max_num_frames = 10, verbose=False, mode="test", image_size=256):
        self.dataset_path = Path("/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2/spatio_temporal_v46")
        self.verbose = verbose
        self.mode = mode
        self.max_num_frames = max_num_frames

        self.test_roi_indices = [0, 1, 14, 21, 29, 31, 34, 35, 37, 38, 50, 51, 52, 56, 65, 71, 80, 83]
        self.train_roi_indices = [i for i in range(187) if i not in self.test_roi_indices]
        self.test_rois = [f"roi{i}" for i in self.test_roi_indices]
        self.train_rois = [f"roi{i}" for i in self.train_roi_indices]

        self.load_spatio_temporal_info()

    def __len__(self):
        return 5000
    
    def transforms(self, msi):

        # Horizontal flip
        if torch.rand(1) > 0.5:
            msi = msi.flip(2)

        # Vertical flip
        if torch.rand(1) > 0.5:
            msi = msi.flip(3)

        # Rotation 90*n degrees
        if torch.rand(1) < 0.75:
            n = random.randint(1, 3)
            msi = msi.rot90(n, [2, 3])

        return msi
    
    def __getitem__(self, idx):

        # randomly select a row in self.roi_spatio_temporal_info
        row = self.roi_spatio_temporal_info.iloc[random.randint(0, len(self.roi_spatio_temporal_info)-1)]
        roi = row["roi_id"]
        patch_id = row["patch_id"]
        day_counts = row["day_count"]
        dates = row["dates"]

        # sample num_frames from 3 to self.max_num_frames
        if self.mode == "train":
            self.num_frames = random.randint(5, self.max_num_frames)
        elif self.mode == "test":
            self.num_frames = self.max_num_frames
        
        day_random_idx = random.randint(0, len(day_counts)-self.num_frames)        
        FILE_PATH = os.path.join(self.dataset_path, f"{roi}_patch{patch_id}.cog")
        WINDOW = rs.windows.Window(0, day_random_idx * 256, 256, 256 * self.num_frames)

        if self.verbose:
            print(f"""{roi} | patch {patch_id} | day {day_random_idx} | latitude {row["latitude"]:.3f} | longtitude {row["longtitude"]:.3f}""")
            print(f"start date: {day_counts[day_random_idx]} | end date: {day_counts[day_random_idx+self.num_frames]} ")
            print(f"start date: {dates[day_random_idx]} | end date: {dates[day_random_idx+self.num_frames]} ")
        
        with rs.open(FILE_PATH) as src:
            msi = torch.from_numpy(src.read(list(range(1, 19)), window=WINDOW)).float()
            
        # Scale back MSI to raw values. 
        # MSI ranges [0, 10K], SAR ranges [0, 32.5*1K], Cloud[0] ranges [0,100], Cloud[1:5] ranges [0,1]
        msi[:13] *= 1 / 10000
        msi[13:15] *= 1 / 1000
        msi[13] *= 1 / 25
        msi[14] *= 1 / 32.5
        msi[15] *= 1 / 100
        
        # print(msi.shape, self.num_frames)
        assert msi.shape == (18, 256 * self.num_frames, 256)
        msi = msi.reshape(18, self.num_frames, 256, 256)

        cld_count = msi[15].mean(keepdim=True, dim=[1,2])
        least_cld_day = cld_count.min(0).indices[0,0]
        cloudy_days = [i for i in range(self.num_frames) if i != least_cld_day]

        least_cloudy_image = msi[(3,2,1,7),least_cld_day]
        cloudy_image = msi[[3,2,1,7]]
        other_images = [cloudy_image[:, day_index] for day_index in cloudy_days[:3]]

        return other_images, least_cloudy_image, torch.zeros([1])

    def load_spatio_temporal_info(self):
        csv_list = glob.glob("/share/hariharan/cloud_removal/MultiSensor/dataset_temp_preprocessed_v2/spatio_temporal_v46/roi*.csv")
        self.roi_spatio_temporal_info = []

        # check if csv'roi in self.test_rois or self.train_rois
        if self.mode == "test":
            csv_list = [csv for csv in csv_list if csv.split("/")[-1].split(".")[0] in self.test_rois]
        elif self.mode == "train":
            csv_list = [csv for csv in csv_list if csv.split("/")[-1].split(".")[0] in self.train_rois]

        for csv_file in csv_list:
            df = pd.read_csv(csv_file)
            if len(self.roi_spatio_temporal_info) == 0:
                df["day_count"] = df['day_count'].apply(lambda x: torch.Tensor([int(num) for num in re.findall(r'\d+', x)]))
                df["dates"] = df['dates'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace("\n", "").split())
                self.roi_spatio_temporal_info = df
            else:
                df["day_count"] = df['day_count'].apply(lambda x: [int(num) for num in re.findall(r'\d+', x)])
                df["dates"] = df['dates'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace("\n", "").split())
                self.roi_spatio_temporal_info = pd.concat([self.roi_spatio_temporal_info, df], ignore_index=True, axis=0)

batch_size = 4
dataset = CogDataset_v46(max_num_frames=8, verbose=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)