import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class TypologyDataset(Dataset):
    def __init__(self, root_dataset_folder, df, transform):
        
        self.root_dataset_folder = root_dataset_folder
        self.transform = transform

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_prompt_pair = self.df.iloc[idx]
        img_filename = img_prompt_pair['Plan_Image_Filename'].replace('Plan', 'Seg')
        
        # Floor_Enc Sofa_Enc Room_Enc Hallway_Enc Type_Enc
        ############################################
        floor_ids = img_prompt_pair['Floor_Enc']
        sofa_ids = img_prompt_pair['Sofa_Enc']
        room_ids = img_prompt_pair['Room_Enc']
        hallway_ids = img_prompt_pair['Hallway_Enc']
        type_ids = img_prompt_pair['Type_Enc']
        ############################################
        
        img_raw = Image.open(os.path.join(self.root_dataset_folder, img_filename)).convert('RGB')
        image = self.transform(img_raw)

        feature_ids = torch.tensor([floor_ids, sofa_ids, room_ids, hallway_ids, type_ids], dtype=torch.float32)
        
        return image, feature_ids