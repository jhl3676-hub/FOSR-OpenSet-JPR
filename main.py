import argparse
import os
import torch
from src.trainers import run_ae_training, run_fosr_training
from src.testers import run_basic_test, run_da_test


def main():
    parser = argparse.ArgumentParser(description="JPR Open Set Recognition Base Model")

    # Required: Source data path
    parser.add_argument('--source_data', type=str, required=True,
                        help='Full path to source .mat file')

    # Optional: Target data path (Required only for DA mode)
    parser.add_argument('--target_data', type=str, default=None,
                        help='Full path to target .mat file')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save models')

    parser.add_argument('--mode', type=str, choices=['basic', 'da'], default='basic',
                        help='Execution mode')

    args = parser.parse_args()

    # Logic: Handle Target Path defaults
    if args.mode == 'basic':
        if args.target_data is None:
            print(f"[*] Basic Mode: Target data not provided. Defaulting to Source data.")
            args.target_data = args.source_data

    elif args.mode == 'da':
        if args.target_data is None:
            raise ValueError("Error: --target_data is REQUIRED for DA mode.")

    # Validation
    if not os.path.exists(args.source_data):
        raise FileNotFoundError(f"Source file not found: {args.source_data}")
    if not os.path.exists(args.target_data):
        raise FileNotFoundError(f"Target file not found: {args.target_data}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Source: {args.source_data}")
    print(f"Target: {args.target_data}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Train AE
    ae_model_path = run_ae_training(
        source_path=args.source_data,
        target_path=args.target_data if args.mode == 'da' else None,
        output_dir=args.output_dir,
        device=device,
        use_da=(args.mode == 'da')
    )

    # 2. Train FOSR
    fosr_model_path = run_fosr_training(
        source_path=args.source_data,
        ae_path=ae_model_path,
        output_dir=args.output_dir,
        device=device
    )

    # 3. Test
    if args.mode == 'basic':
        run_basic_test(args.source_data, args.target_data, ae_model_path, fosr_model_path, device)
    elif args.mode == 'da':
        run_da_test(args.source_data, args.target_data, ae_model_path, fosr_model_path, device)


if __name__ == '__main__':
    main()