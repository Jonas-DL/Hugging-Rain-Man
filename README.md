## <div align="center"> Hugging Rain Man: A Novel Dataset for Analyzing Facial Action Units in Children with Autism Spectrum Disorder </div>

<div align="center">
<a href="https://arxiv.org/abs/2411.13797"><img src="https://img.shields.io/static/v1?label=ArXiv&message=2411.13797&color=B31B1B&logo=arxiv"></a> 
    <div>
        <img src="./Pic/logo.png" width=26%>
    </div>
</div>

## Introduction

This repository contains the annotated Action Unit (AU) and Action Descriptor (AD) labels for the HRM dataset, 
along with pre-trained models for facial action detection and atypical expression regression. The dataset consists of 131,758 frames, 
organized into 1,535 segments. The images themselves are not publicly available temporarily due to privacy and 
ethical considerations. However, the AU labels,  and pre-trained models are provided to facilitate 
research and development in the field of children facial expression analysis, particularly for Autism Spectrum Disorder (ASD).

## Dataset Description

### Overview
- **Total Frames**: 131,758
- **Segments**: 1,535
- **Action Units and Action Descriptors**: 22AUs + 10ADs
- **Atypical Rating**: Annotated by 5 people
- **Facial Expression**: It is obtained by soft voting using 3 algorithms

### Labels, Machine-extracted Features, and Pre-trained Models

- **Model**: ResNet-50, EmoFAN, ME-GraphAU, MAE-Face, FMAE
- **Training Data**: HRM dataset
- **Selected 22 AUs/ADs for detection**: AU1, AU2, AU4, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU16, AU17, AU18, AD19, AU20, AU23, AU24, AU25, AU2X (AU26/27), AU28, AD32, and AU43E.
- **Selected 17 AUs/ADs for detection**: AU1, AU2, AU4, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU16, AU17, AU20, AU23, AU24, AU25 and AU2X (AU26/27).
- **Performance Metrics**: Accuracy, F1-Score
- **Machine-extracted Features**: InsightFace and OpenFace features (5 key points, head pose and bounding box, etc)

### AU Detection Baseline

#### 22 AU
<img src="./Pic/22au.png">


#### 17 AU
<img src="./Pic/17au.png">

### Download Links
- **Baidu Cloud**: [Download Link](https://pan.baidu.com/s/1hMCuq4L892kl092uiDjrvw), pwd:CCNU
- **Mega Cloud (Only Pre-trained Models)**: [Download Link](https://mega.nz/folder/GVYRmbKa#5vfygvAm0mYl_h-6YbFzAQ
)

### Data Format
- **AU Labels**: CSV file with columns for frame number, AU activations, basic facial expression category and atypical rating .
- **Pre-trained Models**: PyTorch `.pth` files.
- **Machine-extracted Features**: `.csv` files.

### Pre-trained Models Usage
You can access the original repositories of each project through the provided links. 
To use the pre-trained models, you can slightly modify the "evaluation" method of each model to 
change the output dimension to 17 or 22. Alternatively, you can use the code snippets we provide in the [Predict](https://github.com/Jonas-DL/Hugging-Rain-Man/tree/main/Predict) directory to 
perform single image AU testing. Any missing libraries can be found in the original repositories.



## AU/AD Annotation Tool 
We provide an additional AU annotation tool that you need to install the PySimpleGUI library in advance.

### Buttons
- **Open data path**: Path where the annotated data (.csv) will be saved.
- **Confirm**: Enter the subject you are currently annotating, and clicking this button will generate S-X.csv in the specified data path.
- **Clear Checkbox**: Clear all checkboxes.
- **Submit**: Submit the final AU/AD annotations.

### LRTB Input Box
Enter the direction of the AU. For example, if AU2 is activated on the right side, enter 2 in the R input box.

## Acknowledgment
We would like to express our gratitude to the following excellent open-source projects: [JAA-Net](https://github.com/ZhiwenShao/PyTorch-JAANet),[EmoFAN](https://github.com/face-analysis/emonet), [EmoFAN4AU-Detection](https://github.com/jingyang2017/aunet_train), 
[ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU), [MAE-Face](https://github.com/FuxiVirtualHuman/MAE-Face), 
[FMAE](https://github.com/forever208/FMAE-IAT), [EAC](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency), 
[Poster++](https://github.com/talented-q/poster_v2), and [DDAMFN++](https://github.com/SainingZhang/DDAMFN).
## Citation
if the data or method help you in the research, please cite the following paper:
```
@article{ji2024hugging,
  title={Hugging Rain Man: A Novel Facial Action Units Dataset for Analyzing Atypical Facial Expressions in Children with Autism Spectrum Disorder},
  author={Yanfeng Ji, Shutong Wang, Ruyi Xu, Jingying Chen, Xinzhou Jiang, Zhengyu Deng, Yuxuan Quan, Junpeng Liu},
  journal={arXiv preprint arXiv:2411.13797},
  year={2024}
}
```
