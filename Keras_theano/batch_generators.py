import numpy as np
import re
import cv2
import os
from sklearn import preprocessing
from skimage import exposure
import random

random.seed(1769)
DataValidationFolder =['val_kitchen','val_office']
second_DataValidationFolder = ['01', '09', '15']
third_DataValidationFolder = ['TEST']
DataTestFolder =['test1']

def makeGaussian(size_x, size_y, sigma = 11, center=None):
    x = np.arange(0, size_x, 1, float)
    y = np.arange(0, size_y, 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size_x // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def load_names(val_seq = -1, augm=0, dataset=0):

    # choose different dataset
    gt_dir = ''
    gt_list = []
    if dataset == 0:
        gt_dir = 'C:\Users\Diego\Desktop\Dataset\Dataset_elaborato\\'
    elif dataset == 1:
        gt_dir = 'Z:\POSEidon\Pandora Dataset\Dataset Pandora'
    elif dataset == 2:
        gt_dir = 'C:\Users\Amministratore\Desktop\MotorMark Dataset\\'

    if val_seq < 0:
        # load all and remove validation sequence
        gt_list = os.listdir(gt_dir)
        deleting = []
        to_remove = []
        if dataset == 0:
            deleting = DataValidationFolder
        elif dataset == 1:
            deleting = second_DataValidationFolder
        elif dataset == 2:
            deleting = third_DataValidationFolder

        for el in deleting:
            to_remove = os.path.join(el)
            gt_list.remove(to_remove)

        if dataset == 0:
            gt_list.remove('test')
            gt_list.remove('test1')
            gt_list.remove('test2')
            gt_list.remove('test3')
            gt_list.remove('testpandora')
        elif dataset == 2:
            gt_list.remove('test1')
    else:
        if val_seq == 2:
            gt_list = DataTestFolder
        else:
            if dataset==0:
                gt_list = DataValidationFolder
            elif dataset==1:
                gt_list = second_DataValidationFolder
            elif dataset==2:
                gt_list = third_DataValidationFolder
    data = []
    if dataset == 0:
        skelfile = 'data.txt'
    elif dataset == 1:
        skelfile = 'data_fix2.txt'
    elif dataset == 2:
        skelfile = 'gt_depth.txt'

    for gt_folder in gt_list:
        for gt_trial_folder in os.listdir(os.path.join(gt_dir,gt_folder)):
            with open(os.path.join(gt_dir,gt_folder,gt_trial_folder, skelfile)) as f:
                lines = f.readlines()
            for i, gt_file in enumerate(sorted(os.listdir(os.path.join(gt_dir,gt_folder,gt_trial_folder,'depth')), key=lambda x: (int(re.sub('\D','',x)),x))):
                if dataset == 2:
                    img_name = os.path.join(gt_dir,gt_folder,gt_trial_folder,'depth',gt_file)
                else:
                    img_name = os.path.join(gt_dir,gt_folder,gt_trial_folder,'DEPTH',gt_file)

                if dataset == 2:
                    skel = np.array(lines[i].split('\t')[1:-1])
                else:
                    skel = np.fromstring(lines[i], sep='\t')
                faces = []
                if dataset == 0:
                    face = int(round(skel[2]))
                    if skel[2] != 0:
                        for i in range(int(skel[2])):
                            x = int(round(skel[9 + (i * 200)]))
                            y = int(round(skel[10 + (i * 200)]))
                            faces.append((x, y))
                    else:
                        x = 0
                        y = 0
                elif dataset == 1:
                    face = 1
                    x = int(round(skel[22]))
                    y = int(round(skel[23]))
                    faces.append((x, y))

                elif dataset == 2:
                    x = y = 0
                    num = (len(skel)-1)/2
                    for index in range(0,len(skel),2):
                        if int(skel[index]) != 9999:
                            x = x + int(skel[index])
                            y = y + int(skel[index + 1])
                        else:
                            num = num - 1
                    x = x/num
                    y = y/num
                    faces.append((x, y))
                    face = 1
                data.append({'image': img_name,'face':face,'augm': int(augm),'facecord':faces})
    #shuffle
    if val_seq!=2:
        random.shuffle(data)
    return data


def load_names_val(dataset=0):
    return load_names(val_seq=1,augm=0,dataset=dataset)


def identity(img):
    return img


def Flip(img):
    return cv2.flip(img, 1)


def Traleft(img):
    M = np.float32([[1, 0, -(img.shape[0]/4)], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traright(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traup(img):
    M = np.float32([[1, 0, 0], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Tradown(img):
    M = np.float32([[1, 0, 0], [0, 1, (img.shape[0]/4 )]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traleftup(img):
    M = np.float32([[1, 0, -(img.shape[0] / 4)], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Trarightup(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, -(img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Traleftdown(img):
    M = np.float32([[1, 0, -(img.shape[0] / 4)], [0, 1, (img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Trarightdown(img):
    M = np.float32([[1, 0, (img.shape[0] / 4)], [0, 1, (img.shape[0] / 4)]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def Rot45Left(img):
    image_center = tuple(np.array(img.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 45.0, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape, flags=cv2.INTER_LINEAR)


def Rot45Right(img):
    image_center = tuple(np.array(img.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -45.0, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape, flags=cv2.INTER_LINEAR)

# map the inputs to the function blocks
Augmentation = {
    0: identity,
    1 : Tradown,
    2 : Traup,
    3 : Traleft,
    4 : Traright,
    5 : Flip,
    6 : Traleftdown,
    7 : Traleftup,
    8 : Trarightdown,
    9 : Trarightup,
    10 : Rot45Left,
    11 : Rot45Right,
}


def load_images(train_data_names, crop, scale, rescale, normcv2, b_debug,fulldepth, rows, cols,equalize,removeBackground,division=8):

    # channel
    ch = 1

    # input structure
    img_batch = np.zeros(shape=(len(train_data_names), ch, rows, cols), dtype=np.float32)

    # Gt structure
    y_batch = np.zeros(shape=(len(train_data_names),  rows/division, cols/division,2), dtype=np.float32)

    for i, line in enumerate(train_data_names):
        # image name
        img_name = line['image']
        img = cv2.imread(img_name, cv2.IMREAD_ANYDEPTH)

        # data augmentation
        img = Augmentation[line['augm']](img)
        deb = img.copy()

        # Rescale
        if rescale:
            img = exposure.rescale_intensity(img.astype('float'), in_range=(np.min(img), np.max(img)), out_range=(0, 1))

        # Scale
        if scale:
            img = preprocessing.scale(img.astype('float'))

        # Normalize (openCV)
        if normcv2:
            img = cv2.normalize(img.astype('float'), alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


        if b_debug:
            cv2.imshow("caricanda augm:{}".format(line['augm']), deb)
            cv2.waitKey()

        # DEBUG
        if line['face']>0:
            c = line['facecord'][0]
            gt = makeGaussian(size_x=img.shape[1], size_y=img.shape[0], center=c,)
        else:
            gt = np.zeros(img.shape)

        # resize
        img = cv2.resize(img, (cols, rows))
        gt = cv2.resize(gt, (cols, rows))
        gt = Augmentation[line['augm']](gt)
        gt = cv2.resize(gt, (cols/division, rows/division))
        # add channel dimension
        img = np.expand_dims(img, 2)
        img = img.astype(np.float32)
        # batch loading
        img_batch[i] = img.transpose(2, 0, 1)

        gt = np.expand_dims(gt, 2)
        gt = np.concatenate((gt, gt), axis=2)
        gt[:, :, 1] = 1 - gt[:, :, 1]
        if b_debug:
            imgdeb = gt[:, :, 0].copy()
            imgdeb = cv2.resize(imgdeb, (512, 424))
            imgdeb = imgdeb* 255
            imgdeb = cv2.applyColorMap(imgdeb.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite('result/gt_{}.jpg'.format(i),imgdeb)
            cv2.imshow('gt',imgdeb)
            cv2.waitKey()
        y_batch[i] = gt

    return img_batch, y_batch





