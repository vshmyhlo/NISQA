from nisqa._resources import resolve_path


def _predict_args(mode, pretrained_model, num_workers, batch_size, ms_channel, output_dir, **kwargs):
    args = {
        "mode": mode,
        "pretrained_model": resolve_path(pretrained_model, "weights"),
        "output_dir": output_dir,
        "tr_bs_val": batch_size,
        "tr_num_workers": num_workers,
        "ms_channel": ms_channel,
        **kwargs,
    }

    if mode == "predict_file":
        if args.get("deg") is None:
            raise ValueError("--deg argument with path to input file needed")
    elif mode == "predict_dir":
        if args.get("data_dir") is None:
            raise ValueError("--data_dir argument with folder with input files needed")
    elif mode == "predict_csv":
        if args.get("csv_file") is None:
            raise ValueError("--csv_file argument with csv file name needed")
        if args.get("csv_deg") is None:
            raise ValueError("--csv_deg argument with csv column name of the filenames needed")
        if args.get("data_dir") is None:
            args["data_dir"] = ""
    else:
        raise NotImplementedError("--mode given not available")

    return args


class NISQAPredictor:
    def __init__(
        self,
        pretrained_model,
        num_workers=0,
        batch_size=1,
        ms_channel=None,
        output_dir=None,
    ):
        self.pretrained_model = pretrained_model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.ms_channel = ms_channel
        self.output_dir = output_dir

    def _predict(self, mode, **kwargs):
        from nisqa.NISQA_model import nisqaModel

        args = _predict_args(
            mode=mode,
            pretrained_model=self.pretrained_model,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            ms_channel=self.ms_channel,
            output_dir=self.output_dir,
            **kwargs,
        )
        model = nisqaModel(args)
        return model.predict()

    def predict_file(self, deg):
        return self._predict("predict_file", deg=deg)

    def predict_dir(self, data_dir):
        return self._predict("predict_dir", data_dir=data_dir)

    def predict_csv(self, csv_file, csv_deg, data_dir=""):
        return self._predict(
            "predict_csv",
            csv_file=csv_file,
            csv_deg=csv_deg,
            data_dir=data_dir,
        )
