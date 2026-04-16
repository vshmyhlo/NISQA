from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn as nn

from nisqa import NISQA_lib as NL
from nisqa._resources import resolve_path

ArgsDict = dict[str, Any]
PredictionInput = str | Path | bytes


class MosPrediction(TypedDict):
    mos_pred: float


class DimPrediction(TypedDict):
    mos_pred: float
    noi_pred: float
    dis_pred: float
    col_pred: float
    loud_pred: float

class NISQAPredictor:
    """Single-audio inference wrapper that keeps one NISQA checkpoint in memory."""

    pretrained_model: str
    dev: torch.device
    ms_channel: int | None
    args: ArgsDict
    model: nn.Module

    def __init__(
        self,
        pretrained_model: str | Path,
        device: str | torch.device | None = None,
        ms_channel: int | None = None,
    ) -> None:
        """Load a pretrained checkpoint for repeated single-audio inference."""
        self.pretrained_model = resolve_path(pretrained_model, "weights")
        self.dev = self._get_device(device)
        self.ms_channel = ms_channel
        self.args, self.model = self._load_model()

    def _get_device(self, device: str | torch.device | None) -> torch.device:
        """Resolve the target device for inference."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self) -> tuple[ArgsDict, nn.Module]:
        """Construct the model from the checkpoint metadata and weights."""
        checkpoint: dict[str, Any] = torch.load(self.pretrained_model, map_location=self.dev)
        args: ArgsDict = checkpoint["args"].copy()
        args["pretrained_model"] = self.pretrained_model

        if self.ms_channel is not None:
            args["ms_channel"] = self.ms_channel

        if args["model"] == "NISQA_DIM":
            args["dim"] = True
            args["csv_mos_train"] = None
            args["csv_mos_val"] = None
        else:
            args["dim"] = False

        if args["model"] == "NISQA_DE":
            raise NotImplementedError("NISQAPredictor only supports single-ended models for per-file prediction.")

        args["double_ended"] = False
        args["csv_ref"] = None

        model_args: ArgsDict = {
            "ms_seg_length": args["ms_seg_length"],
            "ms_n_mels": args["ms_n_mels"],
            "cnn_model": args["cnn_model"],
            "cnn_c_out_1": args["cnn_c_out_1"],
            "cnn_c_out_2": args["cnn_c_out_2"],
            "cnn_c_out_3": args["cnn_c_out_3"],
            "cnn_kernel_size": args["cnn_kernel_size"],
            "cnn_dropout": args["cnn_dropout"],
            "cnn_pool_1": args["cnn_pool_1"],
            "cnn_pool_2": args["cnn_pool_2"],
            "cnn_pool_3": args["cnn_pool_3"],
            "cnn_fc_out_h": args["cnn_fc_out_h"],
            "td": args["td"],
            "td_sa_d_model": args["td_sa_d_model"],
            "td_sa_nhead": args["td_sa_nhead"],
            "td_sa_pos_enc": args["td_sa_pos_enc"],
            "td_sa_num_layers": args["td_sa_num_layers"],
            "td_sa_h": args["td_sa_h"],
            "td_sa_dropout": args["td_sa_dropout"],
            "td_lstm_h": args["td_lstm_h"],
            "td_lstm_num_layers": args["td_lstm_num_layers"],
            "td_lstm_dropout": args["td_lstm_dropout"],
            "td_lstm_bidirectional": args["td_lstm_bidirectional"],
            "td_2": args["td_2"],
            "td_2_sa_d_model": args["td_2_sa_d_model"],
            "td_2_sa_nhead": args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": args["td_2_sa_num_layers"],
            "td_2_sa_h": args["td_2_sa_h"],
            "td_2_sa_dropout": args["td_2_sa_dropout"],
            "td_2_lstm_h": args["td_2_lstm_h"],
            "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
            "pool": args["pool"],
            "pool_att_h": args["pool_att_h"],
            "pool_att_dropout": args["pool_att_dropout"],
        }

        if args["model"] == "NISQA":
            model = NL.NISQA(**model_args)
        elif args["model"] == "NISQA_DIM":
            model = NL.NISQA_DIM(**model_args)
        else:
            raise NotImplementedError("Model not available")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        if args.get("tr_parallel") and self.dev.type != "cpu":
            model = nn.DataParallel(model)

        return args, model

    def _prepare_audio_input(self, audio: PredictionInput) -> PredictionInput:
        """Validate and normalize a path-or-bytes audio input."""
        if isinstance(audio, bytes):
            if not audio:
                raise ValueError("Audio bytes must not be empty")
            return audio

        prepared_path = Path(audio).expanduser().resolve()
        if not prepared_path.is_file():
            raise FileNotFoundError("Audio file not found: {}".format(prepared_path))
        return str(prepared_path)

    def _prediction_kwargs(self) -> ArgsDict:
        """Return shared preprocessing arguments for single-audio inference."""
        return {
            "seg_length": self.args["ms_seg_length"],
            "max_length": self.args["ms_max_segments"],
            "seg_hop_length": self.args["ms_seg_hop_length"],
            "ms_n_fft": self.args["ms_n_fft"],
            "ms_hop_length": self.args["ms_hop_length"],
            "ms_win_length": self.args["ms_win_length"],
            "ms_n_mels": self.args["ms_n_mels"],
            "ms_sr": self.args["ms_sr"],
            "ms_fmax": self.args["ms_fmax"],
            "ms_channel": self.args["ms_channel"],
        }

    def _require_mos_model(self) -> None:
        """Ensure the loaded checkpoint predicts MOS only."""
        if self.args["dim"]:
            raise RuntimeError(
                "Loaded model '{}' predicts dimensions; use predict_dim() instead.".format(
                    self.args["model"]
                )
            )

    def _require_dim_model(self) -> None:
        """Ensure the loaded checkpoint predicts the full NISQA dimension set."""
        if not self.args["dim"]:
            raise RuntimeError(
                "Loaded model '{}' predicts MOS only; use predict_mos() instead.".format(
                    self.args["model"]
                )
            )

    def predict_mos(self, audio: PredictionInput) -> MosPrediction:
        """Predict a single MOS-style score for one audio path or audio byte string."""
        self._require_mos_model()
        audio_input = self._prepare_audio_input(audio)
        y_hat = NL.predict_mos_file(self.model, audio_input, self.dev, **self._prediction_kwargs())
        return {
            "mos_pred": float(y_hat[0, 0]),
        }

    def predict_dim(self, audio: PredictionInput) -> DimPrediction:
        """Predict MOS plus noisiness, discontinuity, coloration, and loudness."""
        self._require_dim_model()
        audio_input = self._prepare_audio_input(audio)
        y_hat = NL.predict_dim_file(self.model, audio_input, self.dev, **self._prediction_kwargs())
        return {
            "mos_pred": float(y_hat[0, 0]),
            "noi_pred": float(y_hat[0, 1]),
            "dis_pred": float(y_hat[0, 2]),
            "col_pred": float(y_hat[0, 3]),
            "loud_pred": float(y_hat[0, 4]),
        }
