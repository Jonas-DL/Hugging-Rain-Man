from collections import OrderedDict
import torch
from torch.nn import Sigmoid
from torchvision import transforms
from PIL import Image
# For FMAE, FMAE-IAT, MAE-FACE, please comment out if not running this algorithm | 用于FMAE,FMAE-IAT,MAE-FACE，如果不运行该算法，请注释
# import models_vit

# For EMOFAN, please comment out if not running this algorithm | 用于EMOFAN，如果不运行该算法，请注释
# from models.emonet_split import EmoNet

# For ME-GRAPH, please comment out if not running this algorithm | 用于ME-GRAPH，如果不运行该算法，请注释
from model.ANFL import MEFARG


def test_MAE_AU_evaluate(image, model, device, threshold=0.5,algo='FMAE'):
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

if __name__ == '__main__':
    """
    We provide pre-trained models and test scripts for the top 5 algorithms.
    You can first download the entire FMAE, FMAE-IAT, ME-Graph or EMOFAN4AU-detection project, and then import this script in the root directory for AU prediction. For MAE-FACE, you only need to download the provided pre-trained model and place it in the FMAE root directory.
    Remember to pre-install the required libraries! (FMAE, FMAE-IAT and MAE-FACE Test pass versions:torch==1.11.0, torchvision==0.12.0; ME-Graph and EMOFAN4AU-detection Test pass versions:torch==2.3.0, torchvision==0.18.0)
    For FMAE and FMAE-IAT, use vit_large_patch16, and for MAE-FACE, use vit_base_patch16.
    
    我们提供了排名前5的算法的预训练模型和测试脚本，
    你可以首先下载FMAE, FMAE-IAT, ME-Graph 或 EMOFAN4AU-detection整个工程，然后在根目录导入该脚本进行AU预测。对于MAE-FACE，仅需要下载它提供的预训练模型，然后放置在FMAE根目录即可。
    记得预先安装所需的库（FMAE, FMAE-IAT和MAE-FACE测试版本:torch==1.11.0, torchvision==0.12.0； ME-Graph和EMOFAN4AU-detection测试版本:torch==2.3.0, torchvision==0.18.0）！
    对于FMAE和FMAE-IAT，使用vit_large_patch16，对于MAE-FACE，使用vit_base_patch16。
    """
    algorithm = 'ME-Graph' # select the algorithm you want to test:FMAE, FMAE-IAT, MAE-FACE, ME-Graph, EMOFAN | 选择算法

    # pretrained_weights = './exp_HRM_finetune_vit_base_maeface/Fold3-16-55.29.pth'
    # pretrained_weights = './exp_HRM_finetune_vit_L_12_3/Fold1-checkpoint-18.pth'
    pretrained_weights = 'results/ASD-resnet50_first_stage_fold3/bs_64_seed_0_lr_0.0001/epoch_best_model_fold3.pth'
    # "27" means "2X", include 26 and 27 | 27相当于2X标签
    aus = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 16, 17, 20, 23, 24, 25, 27]
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
    image_path = './figures/5956.jpg'
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    print(f"Init {algorithm}…………")
    if algorithm in ['FMAE', 'MAE-FACE']:
        # for FMAE
        model_name = 'vit_large_patch16'
        # for MAE-FACE
        # model_name = 'vit_base_patch16'
        model = models_vit.__dict__[model_name](
            num_classes=17,
            drop_path_rate=0.1,
            global_pool=True,
        )
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint = checkpoint['model']
    elif algorithm == 'FMAE-IAT':
        # for FAME-IAT
        model_name = 'vit_large_patch16'
        model = models_vit.__dict__[model_name](
            num_classes=17,
            drop_path_rate=0.1,
            global_pool=True,
            grad_reverse=1.0,  # for grad_reverse | 梯度逆转
        )
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint = checkpoint['model']
    elif algorithm == 'EMOFAN':
        model = EmoNet(n_classes=17)
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
    elif algorithm == 'ME-Graph':
        # Here, only the implementation of stage 1 is provided. | 这里，我们仅提供了stage 1的实现
        model = MEFARG(num_classes=17, backbone='resnet50', neighbor_num=4)

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


    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    # binary classification | 二分类
    sigmoid = Sigmoid()
    test_MAE_AU_evaluate(image_tensor, model, device,algo=algorithm)




