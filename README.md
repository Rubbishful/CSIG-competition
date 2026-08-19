# HYPIR 本地图像恢复项目

基于 [HYPIR](https://arxiv.org/abs/2507.20590)（Harnessing Diffusion-Yielded Score Priors for Image Restoration，HYPIR-SD2）的本地图像恢复项目，用于**华为手机摄影比赛（赛题一）**的同尺寸 4K 图像恢复增强：在 HYPIR-SD2 预训练权重基础上继续 LoRA 微调，聚焦钟表、花朵等纹理细节丰富的场景。

## 目录

- [项目结构](#项目结构)
- [数据](#数据)
- [数据生成与评估工作流](#数据生成与评估工作流)
- [安装](#安装)
- [推理](#推理)
- [训练](#训练)
- [相关文档](#相关文档)

## 项目结构

```text
HYPIR/
├── HYPIR/                     # 核心代码
│   ├── dataset/               # 数据加载与在线退化合成（GT→LQ，无需预存数据对）
│   │   ├── realesrgan.py      #   RealESRGANDataset：二阶退化数据集
│   │   ├── batch_transform.py #   RealESRGANBatchTransform：退化变换流水线
│   │   ├── diffjpeg.py        #   可微 JPEG 压缩
│   │   └── file_backend.py    #   文件读取后端
│   ├── enhancer/              # 推理增强器 SD2Enhancer（tiled 大图处理）
│   ├── model/                 # 判别器 D 与骨干网络
│   ├── trainer/               # 训练器 SD2Trainer（LoRA + GAN/LPIPS/L2 + EMA）
│   └── utils/                 # degradation.py 退化核心、captioner、ema 等
├── configs/                   # 训练/推理配置（sd2_train.yaml、sd2_gradio.yaml）
├── Data/                      # 数据（详见下文）
├── doc/                       # 分析文档（初步分析.md）
├── examples/                  # 推理示例（lq/、prompt/）
├── train.py                   # 训练入口
├── test.py                    # 批量推理入口（tiled patch）
├── app.py / app_openxlab.py   # Gradio / OpenXLab 演示
├── predict.py                 # Cog 推理接口
└── save_model.py              # 权重保存
```

## 数据

```text
Data/
├── 102flowers/          8189 张（花朵）
├── CUB_200_2011/        11788 张（鸟类）
├── WIDER_train/         12880 张（场景/人脸）
├── 赛题一/              比赛数据
│   ├── 测试集/          100 张 LQ（无 GT，4K）
│   ├── 验证集/          10 对 GT/LQ（4K 同尺寸）
│   └── Result/          推理结果输出
└── check_images.py      图片质量检查脚本（数量/分辨率/损坏检测）
```

- 训练集共约 3.3 万张，均为清晰图，分辨率不一（非 512x512），训练时随机裁剪处理。
- 赛题一为**同尺寸 4K 恢复**（非超分），LQ 视为手机光学 + ISP 矫正后的图像（数码变焦/超分算法噪声、抖动等），退化类型多样。

## 数据生成与评估工作流

### 数据生成：在线合成退化（无需预存 LQ 对）

本项目训练**不需要预先准备"清晰图-模糊图"数据对**。`RealESRGANDataset`（[HYPIR/dataset/realesrgan.py](HYPIR/dataset/realesrgan.py)）+ `RealESRGANBatchTransform`（[HYPIR/dataset/batch_transform.py](HYPIR/dataset/batch_transform.py)）在训练时**即时从 GT 合成 LQ**，采用 Real-ESRGAN 风格的**二阶退化**：

1. 第一阶：随机模糊核（iso/aniso/generalized_iso/generalized_aniso/plateau_iso/plateau_aniso 六种）+ sinc 振铃 + resize + 高斯/泊松噪声 + JPEG 压缩
2. 第二阶：对退化图再叠加一轮退化，并按下采样倍率缩放（`stage2_scale`）
3. 全部退化参数在 `configs/sd2_train.yaml` 的 `data_config` 中配置

> 针对本赛题手机摄影场景（数码变焦/ISP 矫正/抖动：模糊 + 噪声 + 偏色 + 重采样伪影的混合退化）的合成管线定制分析见 [doc/初步分析.md](doc/初步分析.md)。

### 图像评估

| 工具 | 作用 |
|---|---|
| `Data/check_images.py` | 批量检查图片数量、分辨率分布、损坏文件（数据质量把关） |
| 退化指纹分析（方法见 [doc/初步分析.md](doc/初步分析.md)） | 对赛题验证集 10 对 GT/LQ 做残差结构分析（残差 RMS / 高频占比 / 边缘-平坦残差 / 偏色量），量化退化构成，用于**校准合成退化参数** |
| 测试集评估 | 赛题测试集 100 张无 GT → NR-IQA（MANIQA/MUSIQ/NIQE）相对提升；验证集 10 对 → PSNR/SSIM/LPIPS/DISTS 客观指标 |

## 安装

```shell
pip install -r requirements.txt
```

模型权重位于 `HYPIR_model/`：

- `HYPIR_sd2.pth`：HYPIR-SD2 的 LoRA 权重（续训/推理使用）
- `HYPIR_sd2_D.safetensors`：判别器权重
- `sd2-1-base/`：Stable Diffusion 2.1 base 模型（本地缓存，避免重复下载）

## 推理

### 批量推理（test.py）

```shell
python test.py \
--base_model_type sd2 \
--base_model_path stabilityai/stable-diffusion-2-1-base \
--model_t 200 --coeff_t 200 \
--lora_rank 256 \
--lora_modules to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj \
--weight_path HYPIR_model/HYPIR_sd2.pth \
--patch_size 512 --stride 256 \
--lq_dir Data/赛题一/测试集 \
--scale_by factor --upscale 1 \
--output_dir Data/赛题一/Result \
--seed 231 --device cuda
```

- 赛题一为同尺寸恢复，使用 `--upscale 1`（如需超分可调整）。
- 4K 大图由 `patch_size`/`stride` 的 tiled 方式自动分块处理。

### Gradio 演示（app.py）

```shell
python app.py --config configs/sd2_gradio.yaml --local --device cuda
```

## 训练

在 HYPIR-SD2 预训练 LoRA 权重（`HYPIR_model/HYPIR_sd2.pth`，`lora_rank=256`）基础上继续微调。

1. 生成训练数据索引文件（parquet，含图片路径与 prompt）：

    ```python
    import polars as pl
    image_paths = [...]  # 递归收集 Data/ 下所有图片路径
    df = pl.from_dict({"image_path": image_paths, "prompt": [""] * len(image_paths)})
    df.write_parquet("path/to/train_meta.parquet")
    ```

2. 填写 [configs/sd2_train.yaml](configs/sd2_train.yaml) 中的 TODO 项：`output_dir`、`file_list`、`image_path_prefix` 等（退化合成参数见上文"数据生成"）。

3. 启动训练：

    ```shell
    accelerate launch train.py --config configs/sd2_train.yaml
    ```

> 详细方案（合成管线定制、训练/评估流程）见 [doc/初步分析.md](doc/初步分析.md)。

## 相关文档

- [doc/初步分析.md](doc/初步分析.md)：数据统计、退化指纹实验、合成数据/训练/评估方案与决策记录。
