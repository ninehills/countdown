# CountDown Game Distill SFT and RL

## 0. 环境搭建

```bash
conda create -n countdown python=3.10
conda activate countdown

cp env.template .env
```

## 1. 准备数据

使用 `1_prepare_data.ipynb` 准备数据集，并保存到 data/ 目录下。

## 2. 基座模型评测

使用 `2_eval_basic.ipynb` 评测选择的基座模型在数据集上的表现。

## 3. 合成简单的 SFT 数据

使用 `3_sft_data_simple.ipynb` 合成简单 SFT 数据。

合成最简化的正确答案。

## 4. 简单SFT 训练

使用 `4_sft_train_simple.ipynb` 简单 SFT 训练。

## 5. 蒸馏 DeepSeek R1 模型

使用 `5_distill_deepseek_r1.ipynb` 蒸馏 DeepSeek R1 模型。

## 6. 使用蒸馏数据进行 SFT 训练

使用 `6_sft_train_distill.ipynb` 使用蒸馏数据进行 SFT 训练。
