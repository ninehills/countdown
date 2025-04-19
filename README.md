# CountDown Game Distill SFT and RL

## 1. 环境搭建

```bash
conda create -n countdown python=3.10
conda activate countdown
```

## 2. 准备数据

使用 `1_prepare_data.ipynb` 准备数据集，并保存到 data/ 目录下。

## 3. 基座模型评测

使用 `2_eval_basic.ipynb` 评测选择的基座模型在数据集上的表现。