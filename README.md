# DRUPE

This repository is the source code for ["Distribution Preserving Backdoor Attack in Self-supervised Learning"](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a029/1RjEa5rjsHK) (S&P 2024).

# Environment
See requirements.txt

# Prepare Pretrained Encoders
Downloading the clean pretrained encoders in the link provided by [Jia et al.](https://github.com/jinyuan-jia/BadEncoder):
https://drive.google.com/drive/folders/1D8YxcBS4Lo5Da62IbPZrXMP_CA4aWHqL

For example download the pth file "output/cifar10/clean_encoder/model_1000.pth"

# Running

For example, running our method on CIFAR-10 pretrained dataset and GTSRB downstream dataset:

```bash
python3 -u main.py \
--mode drupe \
--batch_size 256 \
--shadow_dataset cifar10 \
--pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
--encoder_usage_info cifar10 \
--downstream_dataset gtsrb \
--target_label 12 \
--gpu 0 \
--trigger_file ./trigger/trigger_pt_white_21_10_ap_replace.npz \
--lr 0.05 --epochs 120 \
--reference_file ./reference/gtsrb_l12_n3.npz 
```

Running baseline method:

```bash
python3 -u main.py \
--mode badencoder \
--batch_size 256 \
--shadow_dataset cifar10 \
--pretrained_encoder ./output/cifar10/clean_encoder/model_1000.pth \
--encoder_usage_info cifar10 \
--downstream_dataset gtsrb \
--target_label 12 \
--gpu 0 \
--trigger_file ./trigger/trigger_pt_white_21_10_ap_replace.npz \
--lr 0.05 --epochs 120 \
--reference_file ./reference/gtsrb_l12_n3.npz 
```

## Acknowledgements
Part of the codes are modifed based on https://github.com/jinyuan-jia/BadEncoder.

## Cite this work
You are encouraged to cite the following paper if you use the repo for academic research.

```
@inproceedings{tao2023distribution,
  title={Distribution preserving backdoor attack in self-supervised learning},
  author={Tao, Guanhong and Wang, Zhenting and Feng, Shiwei and Shen, Guangyu and Ma, Shiqing and Zhang, Xiangyu},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  pages={29--29},
  year={2023},
  organization={IEEE Computer Society}
}
```
