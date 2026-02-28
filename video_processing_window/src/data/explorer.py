import os
import pandas as pd
from pathlib import Path
import json

class DatasetExplorer:
    DEFECT_TYPES = {
        "00": "Good Weld",
        "01": "Excessive Penetration",
        "02": "Burn Through",
        "06": "Overlap",
        "07": "Lack of Fusion",
        "08": "Excessive Convexity",
        "11": "Crater Cracks"
    }

    def __init__(self, root_path=None):
        self.root_path = Path(root_path) if root_path else None
        self.df = pd.DataFrame()

    def scan(self, root_path=None):
        """Scans the directory structure and returns a manifest DataFrame."""
        if root_path:
            self.root_path = Path(root_path)
        
        if not self.root_path or not self.root_path.exists():
            raise ValueError(f"Invalid root path: {self.root_path}")

        data = []
        
        # Process good_weld
        self._process_split(self.root_path / "good_weld", "good_weld", "00", data)
        
        # Process defect-weld (handle both variants)
        defect_dir = self.root_path / "defect-weld"
        if not defect_dir.exists():
            defect_dir = self.root_path / "defect_data_weld"
        
        self._process_split(defect_dir, "defect_weld", None, data)
        
        self.df = pd.DataFrame(data)
        return self.df

    def _process_split(self, split_dir, split_name, label_override, data_list):
        if not split_dir.exists():
            return
        
        for config_path in split_dir.iterdir():
            if not config_path.is_dir():
                continue
            
            config_name = config_path.name
            
            for run_path in config_path.iterdir():
                if not run_path.is_dir() or run_path.name == "images":
                    continue
                
                run_id = run_path.name
                
                if label_override:
                    label_code = label_override
                else:
                    parts = run_id.split("-")
                    label_code = parts[-1] if len(parts) > 1 else "Unknown"
                
                csv_file = run_path / f"{run_id}.csv"
                audio_file = run_path / f"{run_id}.flac"
                video_file = run_path / f"{run_id}.avi"
                images_dir = run_path / "images"
                
                num_images = 0
                if images_dir.exists():
                    num_images = len(list(images_dir.glob("*.jpg")))
                
                data_list.append({
                    "run_id": run_id,
                    "config": config_name,
                    "label_code": label_code,
                    "label_name": self.DEFECT_TYPES.get(label_code, f"Unknown ({label_code})"),
                    "rel_path": str(run_path.relative_to(self.root_path)),
                    "has_csv": csv_file.exists(),
                    "has_audio": audio_file.exists(),
                    "has_video": video_file.exists(),
                    "num_images": num_images,
                    "split": split_name
                })

    def save_manifest(self, output_path):
        if self.df.empty:
            raise ValueError("No data to save. Run scan() first.")
        self.df.to_csv(output_path, index=False)
        print(f"Manifest saved to {output_path}")

    def load_manifest(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        return self.df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan welding dataset and generate manifest.")
    parser.add_argument("--root", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--output", type=str, default="dataset_manifest.csv", help="Output CSV path")
    
    args = parser.parse_args()
    explorer = DatasetExplorer(args.root)
    explorer.scan()
    explorer.save_manifest(args.output)
