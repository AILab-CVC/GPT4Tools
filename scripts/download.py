import os
import wget
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ['CURL_CA_BUNDLE'] = ''

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Making {path}")

def download_sam(cache_dir):
    model_checkpoint_path = os.path.join(cache_dir, "sam")
    check_path(model_checkpoint_path)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    wget.download(url,out=model_checkpoint_path)

def download_groundingdino(cache_dir):
    path = os.path.join(cache_dir, "groundingdino")
    check_path(path)
    model_checkpoint_path = os.path.join(path, "groundingdino.pth")
    model_config_path = os.path.join(path, "grounding_config.py")
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    wget.download(url, out=model_checkpoint_path)
    config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    wget.download(config_url,out=model_config_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="GPT4Toos download")
    parser.add_argument("--model-names", type=str, nargs='+', required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-dir", type=str, default=None)
    args = parser.parse_args()
    print_args(args)
    ignore_patterns = ["*.safetensors", "*.msgpack", "*.h5", "*.fp16.safetensors", "*.ckpt", "*.fp16.bin", "*.pt", "open_clip*.bin"]
    if args.local_dir is not None:
        check_path(args.local_dir)
        for model_name in args.model_names:
            if 'sam' in model_name:
                download_sam(args.local_dir)
                continue
            elif 'groundingdino' in model_name:
                download_groundingdino(args.local_dir)
                continue
            else:
                snapshot_download(
                    model_name, 
                    local_dir=args.local_dir, 
                    local_dir_use_symlinks=False, 
                    ignore_patterns=ignore_patterns)
    elif args.cache_dir is not None:
        check_path(args.cache_dir)
        for model_name in args.model_names:
            if 'sam' in model_name:
                download_sam(args.cache_dir)
                continue
            elif 'groundingdino' in model_name:
                download_groundingdino(args.cache_dir)
                continue
            snapshot_download(
                model_name,
                cache_dir=args.cache_dir,
                ignore_patterns=ignore_patterns)
            
    print("Done!")