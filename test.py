import os
import sys
import argparse

# --- DOCSTRING WITH RAW STRING TO FIX SYNTAX WARNING ---
r"""
For Random Forest (PKL):
python test.py --test_dir test_data/ --model_path supervised_multiclass_video_processing/weights/multimodal_rf_model_full.pkl --output_csv rf_submission.csv

For Deep Learning (PTH):
python test.py --test_dir test_data/ --model_path checkpoints/video_classifier.pth --output_csv nn_submission.csv
"""

def main():
    parser = argparse.ArgumentParser(description="Generate submission CSV from test data")
    parser.add_argument("--test_dir", type=str, default="test_data",
                        help="Path to test_data/ directory containing sample folders")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.pkl for Random Forest or .pth for PyTorch Neural Network)")
    parser.add_argument("--output_csv", type=str, default="submission.csv",
                        help="Output path for submission CSV")
    
    args = parser.parse_args()
    
    # Ensure test directory exists
    if not os.path.exists(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        print("Please check the path to your test data.")
        return
        
    # Ensure model checkpoint exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model checkpoint not found: {args.model_path}")
        print("Please train the model first or provide the correct path.")
        return
        
    print(f"Generating submission using model: {args.model_path}")
    print(f"Test data directory: {args.test_dir}")
    print(f"Output file: {args.output_csv}\n")
    
    # Add the project root and multiclass folder to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    multiclass_dir = os.path.join(project_root, "supervised_multiclass_video_processing")
    
    if args.model_path.endswith('.pkl'):
        print("Detected PKL model. Running Random Forest multimodal inference...")
        if multiclass_dir not in sys.path:
            sys.path.insert(0, multiclass_dir)
        
        # Delayed import to avoid ModuleNotFoundError at startup
        from supervised_multiclass_video_processing.inference_rf import generate_multimodal_submission
        
        generate_multimodal_submission(
            test_dir=args.test_dir,
            model_pkl_path=args.model_path,
            output_csv=args.output_csv
        )
    elif args.model_path.endswith('.pth'):
        print("Detected PTH model. Running Deep Learning video network inference...")
        if multiclass_dir not in sys.path:
            sys.path.insert(0, multiclass_dir)
            
        # Delayed import to avoid ModuleNotFoundError at startup
        from supervised_multiclass_video_processing.inference import generate_submission_csv
        
        generate_submission_csv(
            test_dir=args.test_dir,
            video_model_pth=args.model_path,
            output_csv=args.output_csv
        )
    else:
        print("ERROR: Unknown model format. Please provide a .pkl or .pth file.")
        
if __name__ == "__main__":
    main()
