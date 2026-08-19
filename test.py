import argparse
import os
from datetime import datetime
from pathlib import Path
from time import time

from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from HYPIR.enhancer.sd2 import SD2Enhancer
from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sd2_gradio.yaml"
DEFAULT_LQ_DIR = PROJECT_ROOT / "Data" / "competition" / "test"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "Data" / "competition" / "Result"


def default_output_dir(run_name: str | None = None) -> Path:
    """默认输出到 Data/competition/Result/<YYYY-MM-DD-N>(自动找下一个序号),可用 --run_name 覆盖。"""
    result_root = DEFAULT_RESULT_ROOT
    if run_name:
        name = run_name
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        existing = {d.name for d in result_root.glob(f"{today}-*")} if result_root.exists() else set()
        n = 1
        while f"{today}-{n}" in existing:
            n += 1
        name = f"{today}-{n}"
    return result_root / name


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "批量推理入口(赛题默认链路)。模型参数默认从 configs/sd2_gradio.yaml 读取,"
            "命令行显式指定时覆盖 yaml 值。默认输入 Data/competition/test,"
            "默认输出到 Data/competition/Result 下按日期自动编号的独立文件夹。"
        )
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="模型参数配置文件(yaml),默认 configs/sd2_gradio.yaml")
    parser.add_argument("--base_model_type", type=str, default=None, choices=["sd2"],
                        help="Type of the base model. Currently only 'sd2' is supported. 默认取 yaml。")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to the base model directory. 默认取 yaml。")
    parser.add_argument("--model_t", type=int, default=None,
                        help="Model input timestep. 默认取 yaml。")
    parser.add_argument("--coeff_t", type=int, default=None,
                        help="Timestep used to calculate the conversion coefficients from noise to data. 默认取 yaml。")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="Rank of the LoRA modules. 默认取 yaml。")
    parser.add_argument("--lora_modules", type=str, default=None,
                        help="Comma-separated list of LoRA module names. 默认取 yaml。")
    parser.add_argument("--weight_path", type=str, default=None,
                        help="Path to the LoRA weight file. 默认取 yaml。")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Size of the patches to process.")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for the patches.")
    parser.add_argument("--lq_dir", type=str, default=str(DEFAULT_LQ_DIR),
                        help="Directory containing low-quality images. Support nested directories. "
                             "默认 Data/competition/test(赛题测试集)。")
    parser.add_argument("--scale_by", type=str, default="factor", choices=["factor", "longest_side"],
                        help=(
                            "Method to scale the input images. "
                            "'factor' scales by a fixed factor, 'longest_side' scales by the longest side (to a fixed size)."
                        ))
    parser.add_argument("--upscale", type=int, default=1,
                        help="Upscaling factor. 默认 1(赛题同尺寸恢复)。")
    parser.add_argument("--target_longest_side", type=int, default=None,
                        help="Target longest side for scaling if 'scale_by' is set to 'longest_side'.")
    parser.add_argument("--txt_dir", type=str, default=None,
                        help=(
                            "Directory containing text prompts for images. "
                            "The structure of the directory should match the structure of 'lq_dir'. "
                            "e.g. if image path is 'lq_dir/a/b/c.png', then the prompt should be in 'txt_dir/a/b/c.txt'. "
                            "If txt_dir is None, will use captioner."
                        ))
    parser.add_argument("--captioner", type=str, choices=["empty", "fixed"], default="empty",
                        help="Captioner to use. 'empty' for no captions, 'fixed' for a fixed caption. 默认 empty。")
    parser.add_argument("--fixed_caption", type=str, default=None,
                        help="Fixed caption to use if 'captioner' is set to 'fixed'.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=(
                            "Directory to save the results. 默认 Data/competition/Result/<YYYY-MM-DD-N>,"
                            "N 为当日自动递增序号;可用 --run_name 自定义文件夹名。"
                        ))
    parser.add_argument("--run_name", type=str, default=None,
                        help="输出文件夹名(覆盖自动日期编号,仅当未指定 --output_dir 时生效)。")
    parser.add_argument("--seed", type=int, default=231,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., 'cuda', 'cpu').")
    args = parser.parse_args()
    return args


def resolve_model_args(args):
    """从 yaml 读取模型参数默认值,命令行显式指定时覆盖。"""
    config = OmegaConf.load(args.config)
    model_keys = [
        "base_model_type", "base_model_path", "model_t", "coeff_t",
        "lora_rank", "lora_modules", "weight_path",
    ]
    resolved = {}
    for key in model_keys:
        cli_val = getattr(args, key)
        if cli_val is not None:
            resolved[key] = cli_val
        else:
            if key not in config:
                raise ValueError(
                    f"参数 --{key} 未指定,且配置文件 {args.config} 中缺少 {key}。"
                )
            resolved[key] = config[key]
    # lora_modules 支持 yaml 列表或命令行逗号分隔字符串
    if isinstance(resolved["lora_modules"], str):
        resolved["lora_modules"] = resolved["lora_modules"].split(",")
    else:
        resolved["lora_modules"] = list(resolved["lora_modules"])
    return resolved


if __name__ == "__main__":
    args = parse_args()
    model_args = resolve_model_args(args)
    set_seed(args.seed)

    print(f"Model config (from {args.config}, CLI overrides applied):")
    for k, v in model_args.items():
        display = v if k != "lora_modules" else f"[{len(v)} modules]"
        print(f"  {k}: {display}")

    model = SD2Enhancer(
        base_model_path=model_args["base_model_path"],
        weight_path=model_args["weight_path"],
        lora_modules=model_args["lora_modules"],
        lora_rank=model_args["lora_rank"],
        model_t=model_args["model_t"],
        coeff_t=model_args["coeff_t"],
        device=args.device,
    )
    print("Start loading models")
    load_start = time()
    model.init_models()
    print(f"Models loaded in {time() - load_start:.2f} seconds.")

    input_dir = Path(args.lq_dir)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = default_output_dir(run_name=args.run_name)
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = Path(root) / file
                images.append(full_path)
    images.sort(key=lambda x: str(x.relative_to(input_dir)))
    print(f"Found {len(images)} images in {input_dir}.")

    if args.txt_dir is None:
        if args.captioner == "empty":
            captioner = EmptyCaptioner(args.device)
        elif args.captioner == "fixed":
            if args.fixed_caption is None:
                raise ValueError("Fixed caption must be provided when 'captioner' is set to 'fixed'.")
            captioner = FixedCaptioner(args.device, args.fixed_caption)
        else:
            raise ValueError(f"Unknown captioner: {args.captioner}")

    to_tensor = transforms.ToTensor()

    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"
    for file_path in images:
        print(f"Process file: \033[92m{os.path.basename(file_path)}\033[0m")

        relative_path = file_path.relative_to(input_dir)
        result_path = result_dir / relative_path.with_suffix(".png")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / relative_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        lq_pil = Image.open(file_path).convert("RGB")
        lq_tensor = to_tensor(lq_pil).unsqueeze(0)

        if args.txt_dir is not None:
            with open(args.txt_dir / relative_path.with_suffix(".txt"), "r") as fp:
                prompt = fp.read().strip()
        else:
            prompt = captioner(lq_pil)
        with open(prompt_path, "w") as fp:
            fp.write(prompt)
        print(f"Prompt: \033[94m{prompt}\033[0m")

        result = model.enhance(
            lq=lq_tensor,
            prompt=prompt,
            scale_by=args.scale_by,
            upscale=args.upscale,
            target_longest_side=args.target_longest_side,
            patch_size=args.patch_size,
            stride=args.stride,
            return_type="pil",
        )[0]
        result.save(result_path)
    print(f"Done. \033[92mEnjoy your results in {result_dir}.\033[0m")
