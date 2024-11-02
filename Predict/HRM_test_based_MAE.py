import numpy as np
import torch
from torch.nn import Sigmoid
from torchvision import transforms
from PIL import Image
import models_vit


def test_AU_evaluate(image, model, device, threshold=0.5):
    model.eval()  # switch to evaluation mode

    # 将图像转换为模型输入格式
    image = image.to(device)
    image = image.unsqueeze(0)  # 增加一个批次维度

    # 进行预测
    with torch.no_grad():
        output = model(image)  # 2D tensor (1, 12)


    # probs = sigmoid(output).squeeze().cpu().numpy()  # 移除批次维度并转换为numpy数组,用于FMAE,MAE-FACE
    probs = sigmoid(output[0]).squeeze().cpu().numpy()  # 用于FAME-IAT

    # 根据阈值判断每个AU是否激活
    y_pred = (probs >= threshold).astype(int)

    # 打印每个AU是否激活
    for i in range(y_pred.shape[0]):
        print(f"AU {aus[i]}: {'ACTIVATED' if y_pred[i] == 1 else 'NOT ACTIVATED'}")

if __name__ == '__main__':
    """
    You can first download the entire FMAE or FMAE-IAT project, and then import this script in the root directory for AU prediction.
    Remember to pre-install the required libraries! (Test pass versions:torch==1.11.0, torchvision==0.12.0)
    For FMAE and FMAE-IAT, use vit_large_patch16, and for MAE-FACE, use vit_base_patch16.
    
    你可以首先下载FMAE或FMAE-IAT整个工程，然后在根目录导入该脚本进行AU预测
    记得预先安装所需的库（测试版本:torch==1.11.0, torchvision==0.12.0）
    对于FMAE和FMAE-IAT，使用vit_large_patch16，对于MAE-FACE，使用vit_base_patch16
    """
    model_name = 'vit_large_patch16'
    # pretrained_weights = './exp_HRM_finetune_vit_base_maeface/Fold3-16-55.29.pth'
    pretrained_weights = './exp_HRM_finetune_vit_L_12_3/Fold1-checkpoint-18.pth'
    # 示例调用 注意，27=(26 and 27)
    aus = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 16, 17, 20, 23, 24, 25, 27]
    mean = (0.4434122, 0.36354306, 0.35404983)
    std = (0.19467306, 0.19811313, 0.1991408)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像 resize 到 224x224
        transforms.ToTensor(),  # 将图像转换为 Tensor
        transforms.Normalize(mean=mean, std=std)  # 归一化
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 需要预测的图像预先使用retina-face进行人脸对齐
    image_path = './figures/5956.jpg'
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    # 用于FMAE,MAE-FACE
    # model = models_vit.__dict__[model_name](
    #     num_classes=17,
    #     drop_path_rate=0.1,
    #     global_pool=True,
    # )

    # 用于FMAE-IAT
    model = models_vit.__dict__[model_name](
        num_classes=17,
        drop_path_rate=0.1,
        global_pool=True,
        grad_reverse=1.0, # 用于grad_reverse
    )
    checkpoint = torch.load(pretrained_weights, map_location='cpu')
    checkpoint_model = checkpoint['model']
    model.load_state_dict(checkpoint_model, strict=False)
    model = model.to(device)
    # 应用Sigmoid函数
    sigmoid = Sigmoid()
    test_AU_evaluate(image_tensor, model, device)




