# train_cv.py
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run 5-fold cross validation for a given model")
    parser.add_argument("--model", type=str, default="densenet3d121", help="simple3dcnn | densenet3d121")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_ch", type=int, default=16, help="Stem channels for Simple3DCNN")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    args = parser.parse_args()

    python_bin = sys.executable
    print(f"Using Python: {python_bin}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")

    # Create runs directory
    os.makedirs("runs", exist_ok=True)

    for fold in range(5):
        print(f"\n======================================")
        print(f"       STARTING CV FOLD {fold}/5       ")
        print(f"======================================")

        train_csv = f"folds/fold{fold}_train.csv"
        val_csv = f"folds/fold{fold}_val.csv"
        out_dir = f"runs/cv_fold{fold}"
        ckpt_path = f"{out_dir}/best.pt"
        pred_csv = f"{out_dir}/val_subject_predictions.csv"

        # 1. Train
        train_cmd = [
            python_bin, "train.py",
            "--model", args.model,
            "--train_csv", train_csv,
            "--val_csv", val_csv,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--num_workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--base_ch", str(args.base_ch),
            "--dropout", str(args.dropout),
            "--out_dir", out_dir
        ]
        
        print(f"Training command: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)

        # 2. Evaluate subject predictions
        eval_cmd = [
            python_bin, "eval_subject.py",
            "--model", args.model,
            "--csv", val_csv,
            "--ckpt", ckpt_path,
            "--out_csv", pred_csv,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers)
        ]
        print(f"Evaluation command: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)

    # 3. Aggregate results
    print(f"\n======================================")
    print(f"       AGGREGATING CV RESULTS         ")
    print(f"======================================")
    agg_cmd = [python_bin, "aggregate_cv.py"]
    subprocess.run(agg_cmd, check=True)

if __name__ == "__main__":
    main()
