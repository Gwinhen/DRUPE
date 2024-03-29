
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random
import cv2
import copy



class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']
        print("self.target_image_list.shape",self.target_image_list.shape)

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            #backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)
            #print("target_image.shape",target_image.shape)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    def __len__(self):
        return len(self.indices)


class BadEncoderDataset_shiwei(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]
        self.target_image_list = self.target_input_array['x']
        print("self.target_image_list.shape",self.target_image_list.shape)

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        #if self.bd_transform is not None:
        tg_mask  = self.bd_transform(
                    Image.fromarray(np.uint8(self.trigger_mask * 255)).convert('RGB') )
        tg_patch = self.bd_transform(
                    Image.fromarray(np.uint8(self.trigger_patch)).convert('RGB') )
        trans_img  = self.bd_transform(Image.fromarray(np.uint8(img_copy)).convert('RGB') )
        #print(img_copy.shape)
        #print(self.trigger_mask.shape)
        #print(self.trigger_patch.shape)
        #print(self.bd_transform(Image.fromarray(np.uint8(img_copy)).convert('RGB')).shape)
        #print(tg_mask.shape)
        #print(tg_patch.shape)
        #print(trans_img.shape)
        #print(self.trigger_mask.shape)
        #print(self.trigger_patch.shape)

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            #backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            #trans_img  = self.bd_transform(Image.fromarray(np.uint8(img_copy)).convert('RGB') )
            backdoored_image = trans_img * tg_mask + tg_patch
            img_backdoor = backdoored_image
            #img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)
            #print("target_image.shape",target_image.shape)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    def __len__(self):
        return len(self.indices)


class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None, trigger_size=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']


        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

        self.trigger_size = trigger_size

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        if self.trigger_size:
            ori_size = img.shape
            img = cv2.resize(img,(self.trigger_size,self.trigger_size))

        img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        if self.trigger_size:
            img = cv2.resize(img,(ori_size[1],ori_size[1]))
        img_backdoor =self.test_transform(Image.fromarray(img))
        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]



class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None,indices=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
        self.indices = indices
    def __len__(self):
        #return self.data.shape[0]
        if self.indices is not None:
            return len(self.indices)
        else:
            return self.data.shape[0]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        if self.indices is not None:
            img = self.data[self.indices[index]]
        else:
            img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
