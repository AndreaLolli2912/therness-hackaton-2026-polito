import os
import argparse
from inference import generate_submission_csv

"""
python test.py --test_dir ../test_data/ --video_model /checkpoints/video/... --output_csv final_submission.csv
"""

def main():
    parser = argparse.ArgumentParser(description="Generate submission CSV from test data")
    parser.add_argument("--test_dir", type=str, default="../test_data",
                        help="Path to test_data/ directory containing sample folders")
    parser.add_argument("--video_model", type=str, default="checkpoints/video_classifier.pth",
                        help="Path to trained video model checkpoint")
    parser.add_argument("--output_csv", type=str, default="submission.csv",
                        help="Output path for submission CSV")
    
    args = parser.parse_args()
    
    # Ensure test directory exists
    if not os.path.exists(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        print("Please check the path to your test data.")
        return
        
    # Ensure model checkpoint exists
    if not os.path.exists(args.video_model):
        print(f"ERROR: Model checkpoint not found: {args.video_model}")
        print("Please train the model first or provide the correct path.")
        return
        
    print(f"Generating submission using model: {args.video_model}")
    print(f"Test data directory: {args.test_dir}")
    print(f"Output file: {args.output_csv}\n")
    
    # Run the batch inference from inference.py
    generate_submission_csv(
        test_dir=args.test_dir,
        video_model_pth=args.video_model,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()
