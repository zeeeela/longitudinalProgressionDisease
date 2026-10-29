"""DICOM -> uint8 square image, with an optional on-disk cache (decode each DICOM once)."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def load_dicom_array(path):
    """Rescale slope/intercept, padding mask, 1-99 percentile scaling, MONOCHROME1 inversion -> float [0,1]."""
    import pydicom
    ds = pydicom.dcmread(str(path), force=True)
    raw = ds.pixel_array
    if raw.ndim == 3:
        raw = raw[..., 0] if raw.shape[-1] in (3, 4) else raw[0]
    arr = raw.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    valid = np.ones(arr.shape, dtype=bool)
    pad = getattr(ds, "PixelPaddingValue", None)
    if pad is not None:
        try:
            valid = raw != pad
        except Exception:
            pass
    if not valid.any():
        valid = np.ones(arr.shape, dtype=bool)
    lo, hi = np.percentile(arr[valid], [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    arr[~valid] = 0.0
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 1.0 - arr
        arr[~valid] = 0.0
    return arr


def _load_any(path):
    p = Path(path)
    if p.suffix.lower() in (".dcm", ".dicom", ""):
        return load_dicom_array(p)
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def _to_square_uint8(arr, size, keep_aspect):
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if keep_aspect:
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("L", (s, s), 0)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        img = canvas
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)


class ImageStore:
    """get(path) -> uint8 [size, size]. Cached as .npy keyed by path+size+aspect."""

    def __init__(self, size=224, cache_dir=None, keep_aspect=False):
        self.size, self.keep_aspect = int(size), bool(keep_aspect)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem = {}

    def get(self, path):
        key = f"{path}|{self.size}|{int(self.keep_aspect)}"
        if key in self._mem:
            return self._mem[key]
        f = None
        if self.cache_dir:
            f = self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".npy")
            if f.exists():
                a = np.load(f)
                self._mem[key] = a
                return a
        a = _to_square_uint8(_load_any(path), self.size, self.keep_aspect)
        if f is not None:
            np.save(f, a)
        self._mem[key] = a
        return a
