# Multi-scale Domain-adversarial Multiple Instance Learning CNN (CVPR2020)

# Overview
PyTorch implementation of the paper 
- Noriaki H. and Daisuke F. et al., Multi-scale Domain-adversarial Multiple-instance CNN for Cancer Subtype Classification with Unannotated Histopathological Images, CVPR2020 Proceeding [[link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hashimoto_Multi-scale_Domain-adversarial_Multiple-instance_CNN_for_Cancer_Subtype_Classification_with_Unannotated_CVPR_2020_paper.pdf)]

# Requirments
I confirmed that the source code was running with the following environment.
- Python3.6
- pytorch 1.4.0
- CUDA 10.0
- NVIDIA Quadro RTX 5000
- and python library in requirements.txt
# How to use
There is no image data here. 
Therefore, you need to edit the `model/dataset.py` to fit your data.

Here, I'll explain the case of using two magnifications. ('scale1' and 'scale2')
## single scale learning
First, run single scale learning(DA-MIL) for each magnification.
```
$ python single_scale_learning.py scale=scale1
```

```
$ python single_scale_learning.py scale=scale2
```
After run, parameter-files `DAMIL_params_scale1.pth` and `DAMIL_params_scale2.pth` are generated in `tmp_storage/`.

## multi scale learning
After each single scale learning, run multi scale learning (MS-DA-MIL).
```
$ python multi_scale_learning.py
```