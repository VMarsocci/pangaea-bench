import importlib.util
import urllib.request
from logging import Logger
from pathlib import Path

import torch

from pangaea.encoders.base import Encoder


class Galileo_Encoder(Encoder):
    _GALILEO_SINGLE_FILE_URL = (
        "https://raw.githubusercontent.com/nasaharvest/galileo/main/single_file_galileo.py"
    )

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str | None,
        embed_dim: int = 128,
        patch_size: int = 8,
        default_month: int = 6,
        input_resolution_m: int = 10,
        model_subdir: str = "models/nano",
    ) -> None:
        super().__init__(
            model_name="galileo_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.patch_size = patch_size
        self.default_month = default_month
        self.input_resolution_m = input_resolution_m
        self.model_subdir = model_subdir.strip("/")

        self.galileo = self._load_galileo_module()
        self._ensure_galileo_weights_available()
        self.model = self.galileo.Encoder.load_from_folder(
            Path(self.encoder_weights),
            device=torch.device("cpu"),
        )

        self.s2_idx_map = {
            band: self.galileo.SPACE_TIME_BANDS.index(band)
            for band in self.galileo.S2_BANDS
            if band in self.galileo.SPACE_TIME_BANDS
        }
        self.s1_idx_map = {
            band: self.galileo.SPACE_TIME_BANDS.index(band)
            for band in self.galileo.S1_BANDS
            if band in self.galileo.SPACE_TIME_BANDS
        }

        self.s2_group_idx = [
            idx for idx, key in enumerate(self.galileo.SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
        ]
        self.s1_group_idx = [
            idx for idx, key in enumerate(self.galileo.SPACE_TIME_BANDS_GROUPS_IDX) if "S1" in key
        ]

    def download_model(self) -> None:
        # Galileo needs a folder of files (config.json + encoder.pt), so we handle
        # downloads ourselves instead of the base single-file downloader.
        return

    def _load_galileo_module(self):
        cache_dir = Path("pretrained_models")
        cache_dir.mkdir(parents=True, exist_ok=True)
        module_path = cache_dir / "single_file_galileo.py"
        if not module_path.exists():
            urllib.request.urlretrieve(self._GALILEO_SINGLE_FILE_URL, module_path)

        spec = importlib.util.spec_from_file_location("single_file_galileo", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load single_file_galileo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_encoder_weights(self, logger: Logger) -> None:
        if not self.encoder_weights:
            logger.warning("No Galileo model folder specified.")
            return
        if not Path(self.encoder_weights).is_dir():
            raise ValueError(
                f"Galileo encoder_weights must point to a folder containing config.json and encoder.pt, got: {self.encoder_weights}"
            )

    def _candidate_urls(self, filename: str) -> list[str]:
        if not self.download_url:
            return []
        base = str(self.download_url).rstrip("/")
        urls = []
        if "/resolve/" in base:
            urls.append(f"{base}/{self.model_subdir}/{filename}")
            urls.append(f"{base}/{filename}")
        else:
            urls.append(f"{base}/resolve/main/{self.model_subdir}/{filename}")
            urls.append(f"{base}/resolve/main/{filename}")
        return urls

    def _download_with_fallback(self, filename: str, destination: Path) -> None:
        last_error = None
        for url in self._candidate_urls(filename):
            try:
                urllib.request.urlretrieve(url, destination)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"Could not download Galileo file '{filename}' from {self.download_url}"
        ) from last_error

    def _ensure_galileo_weights_available(self) -> None:
        model_dir = Path(self.encoder_weights)
        if model_dir.exists() and model_dir.is_file():
            model_dir.unlink()
        model_dir.mkdir(parents=True, exist_ok=True)

        config_path = model_dir / "config.json"
        encoder_path = model_dir / "encoder.pt"

        if not config_path.exists():
            self._download_with_fallback("config.json", config_path)
        if not encoder_path.exists():
            self._download_with_fallback("encoder.pt", encoder_path)

    @staticmethod
    def _to_bhwtc(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(f"Expected input with 4 or 5 dims, got {tuple(x.shape)}")
        return x.permute(0, 3, 4, 2, 1).contiguous()

    def _build_inputs(self, images: dict[str, torch.Tensor]):
        optical = images.get("optical")
        sar = images.get("sar")
        if optical is None and sar is None:
            raise ValueError("Galileo_Encoder expects at least one of optical or sar inputs.")

        ref = optical if optical is not None else sar
        ref_bhwtc = self._to_bhwtc(ref)
        b, h, w, t, _ = ref_bhwtc.shape
        device = ref_bhwtc.device
        dtype = ref_bhwtc.dtype

        s_t_x = torch.zeros((b, h, w, t, len(self.galileo.SPACE_TIME_BANDS)), device=device, dtype=dtype)
        s_t_m = torch.ones(
            (b, h, w, t, len(self.galileo.SPACE_TIME_BANDS_GROUPS_IDX)),
            device=device,
            dtype=dtype,
        )

        if optical is not None and "optical" in self.input_bands:
            opt = self._to_bhwtc(optical)
            for i, band in enumerate(self.input_bands["optical"]):
                if band in self.s2_idx_map:
                    s_t_x[..., self.s2_idx_map[band]] = opt[..., i]
            if self.s2_group_idx:
                s_t_m[..., self.s2_group_idx] = 0

        if sar is not None and "sar" in self.input_bands:
            s1 = self._to_bhwtc(sar)
            for i, band in enumerate(self.input_bands["sar"]):
                if band in self.s1_idx_map:
                    s_t_x[..., self.s1_idx_map[band]] = s1[..., i]
            if self.s1_group_idx:
                s_t_m[..., self.s1_group_idx] = 0

        sp_x = torch.zeros((b, h, w, len(self.galileo.SPACE_BANDS)), device=device, dtype=dtype)
        t_x = torch.zeros((b, t, len(self.galileo.TIME_BANDS)), device=device, dtype=dtype)
        st_x = torch.zeros((b, len(self.galileo.STATIC_BANDS)), device=device, dtype=dtype)

        sp_m = torch.ones((b, h, w, len(self.galileo.SPACE_BAND_GROUPS_IDX)), device=device, dtype=dtype)
        t_m = torch.ones((b, t, len(self.galileo.TIME_BAND_GROUPS_IDX)), device=device, dtype=dtype)
        st_m = torch.ones((b, len(self.galileo.STATIC_BAND_GROUPS_IDX)), device=device, dtype=dtype)

        months = torch.full(
            (b, t),
            fill_value=self.default_month,
            device=device,
            dtype=torch.long,
        )

        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    def forward(self, images: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = self._build_inputs(images)

        out = self.model(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
            patch_size=self.patch_size,
            input_resolution_m=self.input_resolution_m,
            add_layernorm_on_exit=True,
        )

        patch_tokens = self.model.apply_mask_and_average_tokens_per_patch(
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
        )
        g = int(patch_tokens.shape[1] ** 0.5)
        features = patch_tokens.permute(0, 2, 1).contiguous().view(patch_tokens.shape[0], -1, g, g)

        return [features for _ in self.output_layers]
