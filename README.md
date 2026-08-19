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
├── evaluate_tool/             # 批量评估工具链（paired 成对 / nr 无参考）
├── evaluate_output/           # 评估结果输出（CSV/JSON，gitignore）
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
├── competition/              比赛数据
│   ├── test/          100 张 LQ（无 GT，4K）
│   ├── evaluate/          5 对 GT/LQ（case1-5，4K 同尺寸）
│   └── Result/          推理结果输出（每轮独立文件夹）
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

评估工具链位于 [`evaluate_tool/`](evaluate_tool/)，提供 **paired（成对，有 GT）** 与 **nr（无参考，无 GT）** 两种批量模式，输出 CSV 明细 + JSON 汇总 + 控制台表格。

```shell
# 成对评估：GT vs 恢复结果（PSNR/SSIM/LPIPS + 残差指纹 + 退化构成画像）
# --lq-dir 提供时额外输出 LQ→结果 的退化构成变化（delta_rms / delta_cast）
python -m evaluate_tool.analyze paired \
    --gt-dir Data/competition/evaluate --pred-dir Data/competition/Result/<run> \
    --lq-dir Data/competition/evaluate --out-dir evaluate_output/run_vs_gt

# 无参考评估：NIQE（本地参数）/ MANIQA / MUSIQ（本地权重）+ 单图退化指纹
# --dir2 提供时按序号配对输出指标相对提升（delta_*）
python -m evaluate_tool.analyze nr \
    --dir Data/competition/test --dir2 Data/competition/Result/<run> \
    --metric niqe,maniqa,musiq
```

- **paired**：全参考指标 PSNR/SSIM/LPIPS（LPIPS 自动缩放最长边到 1024 控制 4K 开销）+ 残差结构指纹（残差 RMS / 高频占比 / 边缘-平坦残差 / 通道偏色量，方法见 [doc/初步分析.md](doc/初步分析.md) 实验 4），并给出**退化构成画像**：严重度（轻/中/重度，按残差 RMS）+ 退化成分组合（模糊/噪声/偏色/边缘伪影独立判定，支持混合，如"重度·模糊+偏色"）+ 结构主导成分（模糊/噪声/边缘伪影中分数最高者——对恢复任务结构退化比偏色更本质）。
- **nr**：NR-IQA 指标默认 `NIQE`（内置实现，无需下载）；`MANIQA`/`MUSIQ` 使用本地权重 `HYPIR_model/MANIQA.pt`、`HYPIR_model/MUSIQ.pth`，NIQE 参数文件 `HYPIR_model/niqe_modelparameters.mat`（均需提前放置，不会运行时联网下载）。
- 文件名按数字序号自动配对（`case1_gt.jpg` ↔ `case-1.png` ↔ `case1_lq.jpg`），目录混放 GT/LQ 时用 `--gt-suffix _gt` / `--lq-suffix _lq` 区分。
- `Data/check_images.py` 负责数据质量把关（数量/分辨率/损坏检测）。

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

模型参数默认从 `configs/sd2_gradio.yaml` 读取（命令行显式指定时覆盖）；默认输入赛题测试集、默认输出到 Result 下自动编号的独立文件夹：

```shell
# 赛题默认链路：输入 Data/competition/test，输出 Data/competition/Result/<YYYY-MM-DD-N>，
# upscale=1（同尺寸恢复）、captioner=empty（无提示词）、patch 512/stride 256
python test.py

# 自定义：指定输出文件夹名 / 输入目录 / 覆盖模型参数
python test.py --run_name my-run --lq_dir /path/to/images --upscale 1 \
    --lora_rank 256 --patch_size 512 --stride 256 --device cuda
```

- 赛题一为同尺寸恢复，`--upscale` 默认 1（如需超分可调整）。
- 模型参数（`--base_model_path`/`--model_t`/`--coeff_t`/`--lora_rank`/`--lora_modules`/`--weight_path`）默认取 `configs/sd2_gradio.yaml`，命令行传参即覆盖。
- 4K 大图由 `patch_size`/`stride` 的 tiled 方式自动分块处理；结果保存到 `<output>/result/`，prompt 保存到 `<output>/prompt/`。

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
