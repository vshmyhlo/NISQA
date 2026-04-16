import argparse

from nisqa.NISQA_model import nisqaModel
from nisqa._resources import resolve_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        default="weights/nisqa.tar",
        type=str,
        help="path to pretrained model; supports local files and packaged weights",
    )
    parser.add_argument("--data_dir", required=True, type=str, help="main input dir with dataset samples and csv files")
    parser.add_argument("--output_dir", required=True, type=str, help="output dir for predictions and evaluation")
    parser.add_argument("--csv_file", required=True, type=str, help="csv file with file-level labels")
    parser.add_argument("--csv_deg", default="filepath_deg", type=str, help="csv column with degraded file paths")
    parser.add_argument("--csv_mos_val", default="mos", type=str, help="csv column with target MOS values")
    parser.add_argument("--csv_con", type=str, help="optional csv file with condition-level labels")
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers for PyTorch dataloader")
    parser.add_argument("--bs", type=int, default=40, help="batch size for evaluation")
    parser.add_argument("--ms_channel", type=int, help="audio channel in case of stereo file")
    parser.add_argument("--mapping", default="first_order", type=str, help="evaluation mapping to apply")
    parser.add_argument("--do_plot", action="store_true", help="plot evaluation figures")
    parser.add_argument("--no_print", action="store_true", help="disable evaluation result printing")
    namespace = parser.parse_args()

    args = {
        "mode": "predict_csv",
        "pretrained_model": resolve_path(namespace.pretrained_model, "weights"),
        "data_dir": namespace.data_dir,
        "output_dir": namespace.output_dir,
        "csv_file": namespace.csv_file,
        "csv_con": namespace.csv_con,
        "csv_deg": namespace.csv_deg,
        "csv_mos_val": namespace.csv_mos_val,
        "tr_num_workers": namespace.num_workers,
        "tr_bs_val": namespace.bs,
        "ms_channel": namespace.ms_channel,
    }

    nisqa = nisqaModel(args)
    nisqa.predict()
    nisqa.evaluate(
        mapping=namespace.mapping,
        do_print=not namespace.no_print,
        do_plot=namespace.do_plot,
    )






























