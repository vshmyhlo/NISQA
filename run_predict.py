import argparse

from nisqa.inference import NISQAPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str, help="either predict_file, predict_dir, or predict_csv")
    parser.add_argument(
        "--pretrained_model",
        required=True,
        type=str,
        help="path to pretrained model; supports local files and packaged weights",
    )
    parser.add_argument("--deg", type=str, help="path to speech file")
    parser.add_argument("--data_dir", type=str, help="folder with speech files")
    parser.add_argument("--output_dir", type=str, help="folder to output results.csv")
    parser.add_argument("--csv_file", type=str, help="file name of csv")
    parser.add_argument("--csv_deg", type=str, help="column in csv with file names or paths")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for PyTorch dataloader")
    parser.add_argument("--bs", type=int, default=1, help="batch size for predicting")
    parser.add_argument("--ms_channel", type=int, help="audio channel in case of stereo file")
    args = parser.parse_args()

    predictor = NISQAPredictor(
        pretrained_model=args.pretrained_model,
        num_workers=args.num_workers,
        batch_size=args.bs,
        ms_channel=args.ms_channel,
        output_dir=args.output_dir,
    )

    if args.mode == "predict_file":
        predictor.predict_file(args.deg)
    elif args.mode == "predict_dir":
        predictor.predict_dir(args.data_dir)
    elif args.mode == "predict_csv":
        predictor.predict_csv(args.csv_file, args.csv_deg, data_dir=args.data_dir or "")
    else:
        raise NotImplementedError("--mode given not available")





























