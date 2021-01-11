import cv2
import pathlib
import skimage
import matplotlib.pyplot as plt
import time
import config
import numpy as np
from math import *
import math
import os
from tqdm import tqdm
from progressbar import ProgressBar

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # print(label_to_index)
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]
    return all_image_path, all_image_label

def data_aug_flip(img,horizontal=False,vertical=False,HorizVerti=False):
    if np.random.randint(1,4)==1:
        # 水平翻转
        if horizontal==True:
            flip_horizontal = cv2.flip(img, 1)
            return flip_horizontal
        # 垂直翻转
        if vertical==True:
            flip_vertical = cv2.flip(img, 0)
            return flip_vertical
        # 水平加垂直翻转
        if HorizVerti==True:
            flip_hv = cv2.flip(img, -1)
            return flip_hv
    else:
        return img

def data_aug_rotate(image,random=False,ang=5,angstart=-90,angend=90):
    #以中心为基准无缩放旋转
    height, width = image.shape[:2]
    if random:
        degree = np.random.randint(angstart, angend)  # 随机设置图片旋转的角度
    else:
        degree=ang
    #获取宽高
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    #计算新图片的宽高
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    #对图片以中心进行无缩放旋转
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    #设置平移量
    imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation

def data_aug_gs(img,var_l=0.007,var_h=0.035):
    #高斯噪声
    var=np.random.uniform(var_l,var_h)
    noisy = skimage.util.random_noise(img, mode='gaussian', var=var)
    return noisy

def data_aug_mask(img):
    #遮挡
    # x, y, w, h, l = [50, 50, 100, 100, 80]
    x=np.random.randint(10,50)
    y=np.random.randint(10,50)
    w=np.random.randint(50,100)
    h=np.random.randint(50,100)
    l=np.random.randint(10,50)
    img_h, img_w = img[:, :, 0].shape
    temp = x
    while y < img_h:
        x = temp
        while x < img_w:
            if x + l >= img_w or y + l > img_h: break
            for j in range(x, x + l):
                for i in range(y, y + l):
                    # for k in range(3):
                    #     img[i, j, k] = 0
                    img[i, j, 0] = np.random.randint(0,255)
                    img[i, j, 1] = np.random.randint(0,255)
                    img[i, j, 2] = np.random.randint(0,255)

            x = x + l + w
        y = y + l + h
    return img

def data_aug_twise(img,random=False,ang=10,angstart=5,angend=25):
    #扭曲
    if random:
        angle = np.random.randint(angstart, angend)  # 随机设置扭曲的基准角度
    else:
        angle=ang
    rows, cols = img.shape[:2]
    img_output = np.zeros(img.shape, dtype=img.dtype)
    for i in range(rows):
        for j in range(cols):

            offset_x = int(angle * math.sin(2 * 3.14 * i / cols))
            offset_y = int(angle * math.cos(2 * 3.14 * j / cols))
            if i + offset_y < rows and j + offset_x < cols:
                img_output[i, j] = img[i + offset_y, j + offset_x]
            else:
                img_output[i, j] = 0
    return img_output

def data_aug_shine(img,random=False,inten=10,intenstart=10,intenend=90):
    #光照增强
    def make_pic_lights(img):
        # 读取原始图像
        # 获取图像行和列
        rows, cols = img.shape[:2]

        '''可修改'''
        centerX = np.random.randint(0, rows)  # 设置中心点
        centerY = np.random.randint(0, cols)

        print(centerX, centerY)
        radius = min(rows, cols) / 4
        print(radius)

        '''可修改'''
        if random:
            strength = np.random.randint(intenstart, intenend)  # 设置光照强度
        else:
            strength=inten

        # 图像光照特效
        for i in range(rows):
            for j in range(cols):
                # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
                distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
                # 获取原始图像
                B = img[i, j][0]
                G = img[i, j][1]
                R = img[i, j][2]

                result = (int)(strength * (1.0 - math.sqrt(distance) / radius))

                B = img[i, j][0] + result
                G = img[i, j][1] + result
                R = img[i, j][2] + result
                # 判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                img[i, j] = np.uint8((B, G, R))
        print("finished")
        return img
    return make_pic_lights(img)

def data_aug_expan_crop(img,random=False,expanpix=50):
    #扩增，不改变原图大小基础上进行裁剪
    w, h, _ = img.shape
    if w<250 and h<250:
        expanpix=15
    img = cv2.copyMakeBorder(img, expanpix, expanpix, expanpix, expanpix, cv2.BORDER_CONSTANT,
                             value=[125, 122, 113])  # 顶部 底部 左 右 各扩充4个像素 32*32=》40*40
    if random:
        y=np.random.randint(1,w/5)
        x=np.random.randint(1,w/5)
    else:
        x=y=5
    img = img[x:w+x, y:h+y]
    return img
def rgb2gray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.299 * r + 0.578 * g + 0.114 * b
    img[:,:,0]=out
    img[:, :, 1] = out
    img[:, :, 2] = out
    return img
def data_Aug_Contrast_and_Brightness(img,alpha=None, beta=None,random=False):
    if beta==0 and random:
        alpha=np.random.uniform(0.7,1.3)
    if alpha==0 and random:
        beta=np.random.uniform(-60,60)
    if beta!=0 and alpha !=0 and random:
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.uniform(-60, 60)
    if random==False:
        alpha=alpha
        beta=beta
    # beta=0只改变对比度 alpha=1 只改变亮度
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst
if __name__=='__main__':
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=config.train_dir)
    for path in tqdm(all_image_path):
        # pathS=path.split('\\')
        # if pathS[3]=='cat.5.jpg':
        #     print(pathS[3])
        img_or = cv2.imread(path)
        #
        # cv2.namedWindow('t', cv2.WINDOW_NORMAL)
        # cv2.imshow('t', img_or)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB)
        #--调试
        # img=data_aug_mask(img_or)
        # plt.imshow(img)
        # plt.show()
        # time.sleep(500)
        #-------------
        #难度一数据集
        # img=data_aug_flip(img_or,vertical=True)
        # img_1=data_aug_rotate(img,random=True,angstart=-90,angend=90)

        #
        # img = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
        # cv2.namedWindow('t', cv2.WINDOW_NORMAL)
        # cv2.imshow('t', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # time.sleep(500)
        # #难度二数据集
        # img=data_aug_flip(img_or,vertical=True)
        # img=data_aug_rotate(img,random=True)
        # img=data_Aug_Contrast_and_Brightness(img,random=True)
        # img_2=data_aug_twise(img,random=True)
        # #
        # img = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
        # cv2.namedWindow('t', cv2.WINDOW_NORMAL)
        # cv2.imshow('t', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # #难度三数据集
        img=data_aug_flip(img_or,vertical=True)
        img=data_aug_rotate(img,random=True)
        img=data_Aug_Contrast_and_Brightness(img,random=True)
        img=data_aug_twise(img,random=True)
        img_3=data_aug_expan_crop(img,random=True)
        # #
        # img = cv2.cvtColor(img_3, cv2.COLOR_RGB2BGR)
        # img = data_aug_gs(img)
        # cv2.namedWindow('t', cv2.WINDOW_NORMAL)
        # cv2.imshow('t', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # #难度四数据集
        # img=data_aug_flip(img_or,vertical=True)
        # img=data_aug_rotate(img,random=True)
        # img=data_Aug_Contrast_and_Brightness(img,random=True)
        # img=data_aug_twise(img,random=True)
        # img=data_aug_expan_crop(img,random=True)
        # img=data_aug_mask(img)
        # plt.imshow(img_4)
        # plt.show()
        # #
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img_4 = data_aug_gs(img)
        # cv2.namedWindow('t', cv2.WINDOW_NORMAL)
        # cv2.imshow('t', img_4)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pathS=path.split('\\')
        dirs='dataset3\\'+pathS[1]+'\\'+pathS[2]
        path_save='dataset3\\'+pathS[1]+'\\'+pathS[2]+'\\'+pathS[3]
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        img=cv2.cvtColor(img_3,cv2.COLOR_RGB2BGR)
        img = data_aug_gs(img)
        # time.sleep(500)
        cv2.imwrite(path_save,img*255)






