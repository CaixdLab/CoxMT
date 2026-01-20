import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# import openslide  # Uncomment when integrating real WSI reading.

class WSIPreprocessor:
    def __init__(self, patch_size=224, target_dim=1024):
        self.patch_size = patch_size
        self.target_dim = target_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading DINOv2 model...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_tumor_patches(self, slide_path):
        """
        Extracts tumor patches.
        This is a placeholder; replace with ROI selection + real WSI patching.
        """
        # slide = openslide.OpenSlide(slide_path)
        print(f"Extracting patches from {slide_path}...")
        patches = [
            Image.new('RGB', (224, 224), color=(np.random.randint(255), 100, 100))
            for _ in range(5)
        ]
        return patches

    def extract_features(self, patches):
        """
        Extracts features using DINOv2.
        Returns:
            avg_feature: 1x1024 vector (Student Input) 
            all_features: Nx1024 vectors (For Multi-modal Tokenization) 
        """
        batch_tensors = torch.stack([self.transform(p) for p in patches]).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch_tensors)  # Shape: (N, 1024)
            
        # Average features for Single-Modal Cox-MT student
        avg_feature = torch.mean(features, dim=0).cpu().numpy()
        
        return avg_feature, features.cpu().numpy()

def list_wsi_files(wsi_dir):
    exts = {".svs", ".ndpi", ".tif", ".tiff", ".mrxs"}
    wsi_dir = Path(wsi_dir)
    return [p for p in wsi_dir.iterdir() if p.suffix.lower() in exts]


def process_dataset(wsi_dir, output_dir):
    processor = WSIPreprocessor()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_paths = list_wsi_files(wsi_dir)
    if not slide_paths:
        print(f"No WSI files found in {wsi_dir}.")
        return

    for slide_path in slide_paths:
        patches = processor.get_tumor_patches(slide_path)
        if not patches:
            continue

        avg_feat, all_feats = processor.extract_features(patches)
        sample_id = slide_path.stem
        np.save(output_dir / f"{sample_id}_avg.npy", avg_feat)
        np.save(output_dir / f"{sample_id}_seq.npy", all_feats)
        print(f"Saved features for {sample_id}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract WSI features with DINOv2.")
    parser.add_argument("--wsi-dir", required=True, help="Directory containing WSI files.")
    parser.add_argument("--output-dir", default="data/processed_wsi", help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset(wsi_dir=args.wsi_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
