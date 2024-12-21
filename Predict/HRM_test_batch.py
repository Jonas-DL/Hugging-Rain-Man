import glob
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import Sigmoid
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
# For FMAE, FMAE-IAT, MAE-FACE, please comment out if not running this algorithm | 用于FMAE,FMAE-IAT,MAE-FACE，如果不运行该算法，请注释
import models_vit
from torch.utils.data import Dataset
# For EMOFAN, please comment out if not running this algorithm | 用于EMOFAN，如果不运行该算法，请注释
# from models.emonet_split import EmoNet

# For ME-GRAPH, please comment out if not running this algorithm | 用于ME-GRAPH，如果不运行该算法，请注释
# from model.ANFL import MEFARG


class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        """
        初始化CustomImageDataset类

        参数:
            img_dirs (list): 包含所有.jpg文件路径的列表
            transform (callable, optional): 可选的数据变换操作
        """
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.img_dirs)

    def __getitem__(self, idx):
        """
        根据索引获取图像和标签

        参数:
            idx (int): 图像的索引

        返回:
            image: 获取的图像
        """
        img_path = self.img_dirs[idx]

        # frame num
        if img_path.lower().endswith('.bmp'): # for Openface img
            frame = img_path.split('/')[-1].split('.')[0].split('_')[-1] # frame_det_00_000021->000021
        else:
            frame = img_path.split('/')[-1].split('.')[0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image,int(frame)

def get_jpg_files(directory):
    jpg_files = []
    for filename in os.listdir(directory):
        # 判断文件是否是jpg,png,bmp文件
        if filename.lower().endswith('.png') or filename.lower().endswith('.bmp') or filename.lower().endswith('.jpg'):
            jpg_files.append(os.path.join(directory, filename))

    return jpg_files
def test_AU(image, model, device, threshold=0.5, algo='FMAE'):
    sigmoid = Sigmoid()
    model.eval()  # switch to evaluation mode

    image = image.to(device)
    image = image.unsqueeze(0)

    # predcit
    with torch.no_grad():
        output = model(image)  # 2D tensor (1, 17)
        print(output)

    if algo in ['FMAE','MAE-FACE','EMOFAN']:
        probs = sigmoid(output).squeeze().cpu().numpy()  # for FMAE,MAE-FACE
        y_pred = (probs >= threshold).astype(int)
    elif algo == 'FMAE-IAT':
        probs = sigmoid(output[0]).squeeze().cpu().numpy()  # for FAME-IAT
        y_pred = (probs >= threshold).astype(int)
    elif algo == 'ME-Graph':
        probs = output.squeeze().cpu().numpy()
        y_pred = probs >= threshold
    else:
        raise NotImplementedError(f"{algo} is not implemented.")

    # Determine which AU is activated or not | 判断AU是否激活
    for i in range(y_pred.shape[0]):
        print(f"AU {aus[i]}: {'Activated' if y_pred[i] == 1 else 'Not activated'}")

@torch.no_grad()
def eval_batch(model, device, threshold=0.5, algo='FMAE',predict_save_path='./'):
    # binary classification | 二分类
    print(algo)
    sigmoid = Sigmoid()
    model.eval()
    df = pd.DataFrame(columns=['frame'] + aus)

    for images, frames in tqdm(test_data_loader):
        images = images.to(device,non_blocking=True)
        outputs = model(images)

        if algo in ['FMAE', 'MAE-FACE', 'EMOFAN']:
            probs = sigmoid(outputs).cpu().numpy()  # for FMAE,MAE-FACE
            y_pred = (probs >= threshold).astype(int)
        elif algo == 'FMAE-IAT':
            probs = sigmoid(outputs[0]).cpu().numpy()  # for FAME-IAT
            y_pred = (probs >= threshold).astype(int)
        elif algo == 'ME-Graph':
            probs = outputs.cpu().numpy()
            y_pred = probs >= threshold
        else:
            raise NotImplementedError(f"{algorithm} is not implemented.")

        # frame_list.append(frames)
        # pred_list.append(y_pred)
        # 将frames和y_pred组合为一个dataframe，和df拼接  y_pred[8,22] frames[8]

        pred_list = []
        # 判断AU是否激活
        for i in range(y_pred.shape[0]): # batch
            # print(f"Frame {frames[i]}:")
            for au_idx in range(y_pred.shape[1]):
                pred_list.append(y_pred[i, au_idx]) # single au predict
                # if y_pred[i, au_idx] == 1:
                #     print(f"AU {aus[au_idx]}: 1", end=' ')

        # 与df拼接，frames为一列，pred_list中每个AU的结果各自为一列，AU的列名保存在aus变量中
        df = pd.concat([df, pd.DataFrame({'frame': frames, **dict(zip(aus, pred_list))})], ignore_index=True)

    # 按照frame列的值升序排序
    df.sort_values(by='frame',ascending=True,inplace=True)
    df.to_csv(predict_save_path,index=False)
    print(f'Predict results save success!')



if __name__ == '__main__':
    """
    基于MAE-FACE的批量图像AU预测demo
    """
    torch.manual_seed(0)
    np.random.seed(0)
    cudnn.benchmark = True

    algorithm = 'MAE-FACE' # select the algorithm you want to test:FMAE, FMAE-IAT, MAE-FACE, ME-Graph, EMOFAN | 选择算法
    pretrained_weights = './ckpt/Tr23-t1-checkpoint-23.pth'
    # pretrained_weights = '/media/jiyanfeng/纪延峰-陈靓影/Datasets/HuggingRainMan/Pretrained-Models/22AU-new/MAE-FACE/TR-1_2-checkpoint-24.pth'
    # pretrained_weights = '/media/jiyanfeng/纪延峰-陈靓影/Datasets/HuggingRainMan/Pretrained-Models/22AU/FMAE/TR-1_3-checkpoint-29.pth'
    # pretrained_weights = '/media/jiyanfeng/纪延峰-陈靓影/Datasets/HuggingRainMan/Pretrained-Models/22AU/FMAE-IAT/TR-1_3-checkpoint-19.pth'
    # pretrained_weights = '/media/jiyanfeng/纪延峰-陈靓影/Datasets/HuggingRainMan/Pretrained-Models/17AU/MAE-FACE/Fold3-16-55.29.pth'

    batch_size = 8
    num_works = 4

    # "27" means "2X", include 26 and 27 | 27相当于2X标签
    # aus = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 16, 17, 20, 23, 24, 25, 27] # 17 aus
    aus = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18,19,20, 23, 24, 25, 27,28,32,43] # 22 aus [1, 2, 4, 6, 7,9, 10, 12, 14, 15, 16,17,18,19,20, 23, 24,25,26,28,32,43]
    mean = (0.4434122, 0.36354306, 0.35404983)
    std = (0.19467306, 0.19811313, 0.1991408)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # for FMAE,FMAE-IAT,MAE-FACE,ME-Graph
        # transforms.Resize((256, 256)),  # for EMOFAN
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # The images to be predicted are pre-processed using RetinaFace for facial alignment. | 需要预测的图像预先使用retina-face进行人脸对齐
    image_paths = get_jpg_files(r'./imgs/')
    my_data = CustomImageDataset(image_paths, transform)
    # 创建dataloader,如果显存不足请降低batch_size
    test_data_loader = torch.utils.data.DataLoader(my_data, batch_size=batch_size, shuffle=False,num_workers=num_works,pin_memory=True)


    print(f"Init {algorithm}…………")
    if algorithm == 'MAE-FACE':
        # for MAE-FACE
        model_name = 'vit_base_patch16'
        model = models_vit.__dict__[model_name](
            num_classes=22,
            drop_path_rate=0.1,
            global_pool=False,
        )
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint = checkpoint['model']

    elif algorithm == 'FMAE':
        # for FMAE
        model_name = 'vit_large_patch16'
        # for MAE-FACE
        # model_name = 'vit_base_patch16'
        model = models_vit.__dict__[model_name](
            num_classes=22,
            drop_path_rate=0.1,
            global_pool=False,
        )
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint = checkpoint['model']
    elif algorithm == 'FMAE-IAT':
        # for FAME-IAT
        model_name = 'vit_large_patch16'
        model = models_vit.__dict__[model_name](
            num_classes=22,
            drop_path_rate=0.1,
            global_pool=False,
            grad_reverse=1.0,  # for grad_reverse | 梯度逆转
        )
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint = checkpoint['model']
    elif algorithm == 'EMOFAN':
        model = EmoNet(n_classes=22)
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
    elif algorithm == 'ME-Graph':
        # Here, only the implementation of stage 1 is provided. | 这里，我们仅提供了stage 1的实现
        model = MEFARG(num_classes=22, backbone='resnet50', neighbor_num=4)

        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'module.' in k:
                k = k[7:]  # remove `module.`
            new_state_dict[k] = v
        checkpoint = new_state_dict
    else:
        raise NotImplementedError(f"{algorithm} is not implemented.")

    # init model
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    predict_results_csv_path = './results/predict.csv'
    eval_batch(model, device,algo=algorithm,predict_save_path=predict_results_csv_path)



