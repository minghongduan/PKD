import numpy as np
import pandas as pd
import torch
import os
import glob


def statistics_slide(slide_path_list):
    num_patch_allSlide = 0

    for i in slide_path_list:
        num_patch = len(glob.glob(i + "/*.jpg"))
        num_patch_allSlide = num_patch_allSlide + num_patch

    return num_patch_allSlide


class PANDA_feat(torch.utils.data.Dataset):
    # @profile
    def __init__(self, data_csv, root_dir='/data/minghongduan/project2spring/data/PANDA_patches_224x224_scale1/train/',
                 train=True, transform=None, downsample=1.0, drop_threshold=0.0, preload=True, return_bag=False):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.downsample = downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        self.return_bag = return_bag
        
        
        slide_data = pd.read_csv(data_csv)

        if train:
            train_data = slide_data['train'].dropna()
            all_slides = (train_data            
                .apply(lambda x: self.root_dir + x) 
            ).tolist()
            #self.root_dir = os.path.join(self.root_dir, "training")
        else:
            val_data = slide_data['val'].dropna()
            all_slides = (val_data      
                .apply(lambda x: self.root_dir + x) 
            ).tolist()
            #self.root_dir = os.path.join(self.root_dir, "testing")
        
        # 1.1 down sample the slides
        print("================ Down sample ================")
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)
        self.num_patches = statistics_slide(all_slides)

        # 2. load all pre-trained patch features (by SimCLR in DSMIL)
        all_slides_name = [i.split('/')[-1] for i in all_slides]
        self.feat_dir = "/data/minghongduan/project2spring/data/patches_Panda_224x224_10x_fast_conch_new/"
        
        if train:
            train_data = slide_data['train'].dropna()
            all_slides_feat_file = (train_data               
                .apply(lambda x: self.feat_dir + x) 
            ).tolist()
        else:
            val_data = slide_data['val'].dropna()
            all_slides_feat_file = (val_data                
                .apply(lambda x: self.feat_dir + x) 
            ).tolist()
            
        self.slide_feat_all = np.zeros([self.num_patches, 512], dtype=np.float32)
        self.slide_patch_label_all = np.zeros([self.num_patches, 3], dtype=np.int_)
        self.patch_corresponding_slide_label = np.zeros([self.num_patches, 3], dtype=np.int_)
        self.patch_corresponding_slide_index = np.zeros([self.num_patches], dtype=np.int_)
        self.patch_corresponding_slide_name = np.zeros([self.num_patches], dtype='<U13')
        cnt_slide = 0
        pointer = 0
        for i in all_slides_feat_file:
            '''
            slide_name_i = i.split('/')[-1].split('.')[0].replace('_patchfeat', '')
            if slide_name_i not in all_slides_name:
                continue
            slide_i_label_feat = np.load(i)
            slide_i_patch_label = slide_i_label_feat[:, 0]
            slide_i_feat = slide_i_label_feat[:, 1:]
            '''
            slide_name_i = i.split('/')[-1]
            slide_i_feat = np.load(os.path.join(i, "patches.npy"))
            slide_i_patch_label = np.load(os.path.join(i, "patches_label.npy"))
            slide_i_label = np.load(os.path.join(i, "patches_corresponding_slide_label.npy"))
            
            num_patches_i = slide_i_feat.shape[0]

            self.slide_feat_all[pointer:pointer+num_patches_i, :] = np.squeeze(slide_i_feat)
            self.slide_patch_label_all[pointer:pointer+num_patches_i] = slide_i_patch_label
            self.patch_corresponding_slide_label[pointer:pointer+num_patches_i] = slide_i_label * np.ones([num_patches_i, 3], dtype=np.int_)
            self.patch_corresponding_slide_index[pointer:pointer+num_patches_i] = cnt_slide * np.ones([num_patches_i], dtype=np.int_)
            self.patch_corresponding_slide_name[pointer:pointer+num_patches_i] = np.array(slide_name_i).repeat(num_patches_i)
            pointer = pointer + num_patches_i
            cnt_slide = cnt_slide + 1

        self.all_patches = self.slide_feat_all
        self.patch_label = self.slide_patch_label_all
        print("")

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
            self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            bag = self.all_patches[idx_patch_from_slide_i, :]
            patch_labels = self.slide_patch_label_all[idx_patch_from_slide_i]
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            """
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            """
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            patch_image = self.all_patches[index]
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches


class PANDA_feat_test(torch.utils.data.Dataset):
    # @profile
    def __init__(self, data_csv, root_dir='/data/minghongduan/project2spring/data/PANDA_patches_224x224_scale1/train/',
                 transform=None, downsample=1.0, drop_threshold=0.0, preload=True, return_bag=False):
        self.root_dir = root_dir
        self.transform = transform
        self.downsample = downsample
        self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
        self.preload = preload
        self.return_bag = return_bag
        
        
        slide_data = pd.read_csv(data_csv)

        test_data = slide_data['test'].dropna()
        all_slides = (test_data      
            .apply(lambda x: self.root_dir + x) 
        ).tolist()
        #self.root_dir = os.path.join(self.root_dir, "testing")
    
        # 1.1 down sample the slides
        print("================ Down sample ================")
        np.random.shuffle(all_slides)
        all_slides = all_slides[:int(len(all_slides)*self.downsample)]
        self.num_slides = len(all_slides)
        self.num_patches = statistics_slide(all_slides)

        # 2. load all pre-trained patch features (by SimCLR in DSMIL)
        all_slides_name = [i.split('/')[-1] for i in all_slides]
        self.feat_dir = "/data/minghongduan/project2spring/data/patches_Panda_224x224_10x_fast_conch_new/"
        
        test_data = slide_data['test'].dropna()
        all_slides_feat_file = (test_data                
            .apply(lambda x: self.feat_dir + x) 
        ).tolist()
            
        self.slide_feat_all = np.zeros([self.num_patches, 512], dtype=np.float32)
        self.slide_patch_label_all = np.zeros([self.num_patches, 3], dtype=np.int_)
        self.patch_corresponding_slide_label = np.zeros([self.num_patches, 3], dtype=np.int_)
        self.patch_corresponding_slide_index = np.zeros([self.num_patches], dtype=np.int_)
        self.patch_corresponding_slide_name = np.zeros([self.num_patches], dtype='<U13')
        cnt_slide = 0
        pointer = 0
        for i in all_slides_feat_file:
            '''
            slide_name_i = i.split('/')[-1].split('.')[0].replace('_patchfeat', '')
            if slide_name_i not in all_slides_name:
                continue
            slide_i_label_feat = np.load(i)
            slide_i_patch_label = slide_i_label_feat[:, 0]
            slide_i_feat = slide_i_label_feat[:, 1:]
            '''
            slide_name_i = i.split('/')[-1]
            slide_i_feat = np.load(os.path.join(i, "patches.npy"))
            slide_i_patch_label = np.load(os.path.join(i, "patches_label.npy"))
            slide_i_label = np.load(os.path.join(i, "patches_corresponding_slide_label.npy"))
            num_patches_i = slide_i_feat.shape[0]

            self.slide_feat_all[pointer:pointer+num_patches_i, :] = np.squeeze(slide_i_feat)
            self.slide_patch_label_all[pointer:pointer+num_patches_i] = slide_i_patch_label
            self.patch_corresponding_slide_label[pointer:pointer+num_patches_i] = slide_i_label * np.ones([num_patches_i, 3], dtype=np.int_)
            self.patch_corresponding_slide_index[pointer:pointer+num_patches_i] = cnt_slide * np.ones([num_patches_i], dtype=np.int_)
            self.patch_corresponding_slide_name[pointer:pointer+num_patches_i] = np.array(slide_name_i).repeat(num_patches_i)
            pointer = pointer + num_patches_i
            cnt_slide = cnt_slide + 1

        self.all_patches = self.slide_feat_all
        self.patch_label = self.slide_patch_label_all
        print("")

        # 3.do some statistics
        print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
            self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))
        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index==index)[0]
            bag = self.all_patches[idx_patch_from_slide_i, :]
            patch_labels = self.slide_patch_label_all[idx_patch_from_slide_i]
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

            # check data
            """
            if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
                raise
            if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
                raise
            """
            return bag, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            patch_image = self.all_patches[index]
            patch_label = self.patch_label[index]
            patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
            patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
            patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

            # patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))
            return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
                                 patch_corresponding_slide_name], index

    def __len__(self):
        if self.return_bag:
            return self.patch_corresponding_slide_index.max() + 1
        else:
            return self.num_patches
