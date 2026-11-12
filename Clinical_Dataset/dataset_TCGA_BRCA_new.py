import numpy as np
import torch
import os
import pandas as pd


def add_suffix(element):
    element = str(element)
    if 'normal' in element:
        return element + '_neg'
    elif 'tumor' in element:
        return element + '_pos'
    else:
        return element  # 如果既不包含 normal 也不包含 tumor，保持原样


class TCGA_BRCA_feat(torch.utils.data.Dataset):
    # @profile
    def __init__(self, data_csv, train=True, transform=None, downsample=1.0, drop_threshold=0.0, preload=True, return_bag=False):
        self.train = train
        self.transform = transform
        self.downsample = downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        self.return_bag = return_bag
        
        slide_data = pd.read_csv(data_csv)

            #self.root_dir = os.path.join(self.root_dir, "testing")

        # all_slides = glob.glob(self.root_dir + "/*")
        # 1.filter the pos slides which have 0 pos patch
        # all_pos_slides = glob.glob(self.root_dir + "/*_pos")
        

        # 1.1 down sample the slides

        # 2. load all pre-trained patch features (by SimCLR in DSMIL)

        self.feat_dir = "/data/minghongduan/project2spring/data/patches_tcga_brca_224x224_x10_fast_conch_new/"
        
        if train:
            train_data = slide_data['train'].dropna()
            all_slides_feat_file = (train_data
                .apply(add_suffix)                
                .apply(lambda x: self.feat_dir + x) 
            ).tolist()
        else:
            val_data = slide_data['val'].dropna()
            all_slides_feat_file = (val_data
                .apply(add_suffix)                
                .apply(lambda x: self.feat_dir + x) 
            ).tolist()

        self.patch_feat_all = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        
        cnt_slide = 0
        pointer = 0
        for i in all_slides_feat_file:
            slide_name_i = i.split('/')[-1]
            slide_i_feat = np.load(os.path.join(i, "patches.npy"))
            slide_i_label = np.load(os.path.join(i, "patches_corresponding_slide_label.npy"))
            num_patches_i = slide_i_feat.shape[0]
            self.patch_feat_all.append(np.squeeze(slide_i_feat))
            self.patch_corresponding_slide_label.append(slide_i_label)
            self.patch_corresponding_slide_index.append(np.ones(slide_i_feat.shape[0]) * cnt_slide)
            self.patch_corresponding_slide_name.append(np.array(slide_name_i).repeat(num_patches_i))
            
            if num_patches_i == 0:
                print(slide_name_i)

            pointer = pointer + num_patches_i
            cnt_slide = cnt_slide + 1

        self.patch_feat_all = np.concatenate(self.patch_feat_all, axis=0).astype(np.float32)
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label).astype(np.int64)
        self.patch_corresponding_slide_index =np.concatenate(self.patch_corresponding_slide_index).astype(np.int64)
        self.patch_corresponding_slide_name = np.concatenate(self.patch_corresponding_slide_name)

        self.num_patches = self.patch_feat_all.shape[0]

        print("")

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}".format(
            len(all_slides_feat_file), self.num_patches))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            bag = self.patch_feat_all[idx_patch_from_slide_i, :]

            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [slide_label, slide_index, slide_name], index
        else:
            patch_image = self.patch_feat_all[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class TCGA_BRCA_feat_test(torch.utils.data.Dataset):
    # @profile
    def __init__(self, data_csv, transform=None, downsample=1.0, drop_threshold=0.0, preload=True, return_bag=False):
        self.transform = transform
        self.downsample = downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        self.return_bag = return_bag

        slide_data = pd.read_csv(data_csv)

        # 1.filter the pos slides which have 0 pos patch

        # 1.1 down sample the slides

        self.feat_dir = "/data/minghongduan/project2spring/data/patches_tcga_brca_224x224_x10_fast_conch_new/"
        
        test_data = slide_data['test'].dropna()
        all_slides_feat_file = (test_data                
            .apply(lambda x: self.feat_dir + x) 
        ).tolist()
            
        self.patch_feat_all = []
        self.patch_corresponding_slide_label = []
        self.patch_corresponding_slide_index = []
        self.patch_corresponding_slide_name = []
        
        cnt_slide = 0
        pointer = 0
        for i in all_slides_feat_file:
            slide_name_i = i.split('/')[-1]
            slide_i_feat = np.load(os.path.join(i, "patches.npy"))
            slide_i_label = np.load(os.path.join(i, "patches_corresponding_slide_label.npy"))
            num_patches_i = slide_i_feat.shape[0]
            self.patch_feat_all.append(np.squeeze(slide_i_feat))
            self.patch_corresponding_slide_label.append(slide_i_label)
            self.patch_corresponding_slide_index.append(np.ones(slide_i_feat.shape[0]) * cnt_slide)
            self.patch_corresponding_slide_name.append(np.array(slide_name_i).repeat(num_patches_i))

            pointer = pointer + num_patches_i
            cnt_slide = cnt_slide + 1

        self.patch_feat_all = np.concatenate(self.patch_feat_all, axis=0).astype(np.float32)
        self.patch_corresponding_slide_label = np.concatenate(self.patch_corresponding_slide_label).astype(np.int64)
        self.patch_corresponding_slide_index =np.concatenate(self.patch_corresponding_slide_index).astype(np.int64)
        self.patch_corresponding_slide_name = np.concatenate(self.patch_corresponding_slide_name)

        self.num_patches = self.patch_feat_all.shape[0]

        print("")

        # 3.do some statistics
        print("[DATA INFO] num_slide is {};".format(
            len(all_slides_feat_file), self.num_patches))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            bag = self.patch_feat_all[idx_patch_from_slide_i, :]

            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            return bag, [slide_label, slide_index, slide_name], index
        else:
            patch_image = self.patch_feat_all[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches
