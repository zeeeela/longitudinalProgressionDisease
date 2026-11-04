"""Optional local dataset extraction from outer or stored nested ZIP archives."""

import io
import re
import shutil
import struct
import zipfile
from pathlib import Path
import pandas as pd

MEMBER_RX = re.compile(r"(?:^|/)p(\d+)/s(\d+)/([^/]+\.dcm)$", re.I)

class WindowedFile(io.RawIOBase):
    """Seekable read-only view of a byte range of a file (used for stored nested zips)."""

    def __init__(self, path, start, size):
        self.f, self.start, self.size, self.pos = open(path, "rb"), start, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        new = off if whence == io.SEEK_SET else self.pos + off if whence == io.SEEK_CUR else self.size + off
        self.pos = max(0, min(new, self.size))
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        self.f.seek(self.start + self.pos)
        data = self.f.read(n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

    def close(self):
        if not self.closed:
            self.f.close()
        super().close()

def _open_nested(outer_path, info):
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    with open(outer_path, "rb") as raw:
        raw.seek(info.header_offset)
        h = raw.read(30)
    n, e = struct.unpack("<HH", h[26:30])
    win = WindowedFile(outer_path, info.header_offset + 30 + n + e, info.file_size)
    return zipfile.ZipFile(win), win

def _iter_archives(path):
    """Yield (label, ZipFile) for the zip and for each stored nested zip inside it."""
    outer = zipfile.ZipFile(path)
    yield str(path), outer
    for info in outer.infolist():
        if info.filename.lower().endswith(".zip"):
            opened = _open_nested(path, info)
            if opened is None:
                print(f"  skip (compressed nested zip, cannot read in place): {info.filename}")
                continue
            z, win = opened
            yield f"{path}::{info.filename}", z
            z.close(); win.close()
    outer.close()

def wanted_from_pairs(pairs_csv):
    d = pd.read_csv(pairs_csv)
    key = lambda p: re.search(r"p(\d+)[\\/]s(\d+)[\\/]([^\\/]+\.dcm)\s*$", str(p), re.I).groups()
    per_pair = [(key(a), key(b)) for a, b in zip(d["prior_dicom_path"], d["current_dicom_path"])]
    return per_pair, {k for pair in per_pair for k in pair}

def wanted_from_manifest(manifest_csv):
    rx = re.compile(r"p(\d+)[\\/]s(\d+)[\\/]([^\\/]+\.dcm)\s*$", re.I)
    return {rx.search(str(p)).groups() for p in pd.read_csv(manifest_csv)["dicom_path"]}

def extract_needed_dicoms(pairs_csv, src_zips, dest_root, dry_run=True, extra_manifest=None):
    per_pair, wanted = wanted_from_pairs(pairs_csv)
    if extra_manifest is not None and Path(extra_manifest).exists():
        extra = wanted_from_manifest(extra_manifest)
        print(f"backbone manifest DICOMs: {len(extra)} ({len(extra - wanted)} not in pairs)")
        wanted = wanted | extra
    print(f"pairs: {len(per_pair)} | unique DICOMs needed: {len(wanted)}")
    dest_root = Path(dest_root)
    found, sample = {}, []
    for zpath in src_zips:
        zpath = Path(zpath)
        if not zpath.exists():
            print("missing zip:", zpath); continue
        for label, z in _iter_archives(zpath):
            names = z.namelist()
            hits = 0
            for info in z.infolist():
                m = MEMBER_RX.search(info.filename)
                if not m:
                    continue
                k = m.groups()
                if len(sample) < 3:
                    sample.append(info.filename)
                if k in wanted and k not in found:
                    hits += 1
                    found[k] = None
                    if not dry_run:
                        out = dest_root / f"p{k[0]}" / f"s{k[1]}" / k[2]
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(info) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst)
            print(f"{label}: {len(names)} members, {hits} needed DICOMs {'found' if dry_run else 'extracted'}")
    complete = sum(a in found and b in found for a, b in per_pair)
    print(f"\nDICOM coverage: {len(found)}/{len(wanted)} files | pairs with BOTH images: {complete}/{len(per_pair)}")
    if sample:
        print("example member names:", sample)
    if not found:
        print("No needed DICOM found. Check src_zips (print z.namelist()[:10]) and the member layout.")
    return found

def extract_mimic_iv(zip_path, dest, tables, dry_run=True, chunk=16 * 1024 * 1024):
    import io, re, shutil, struct, zipfile
    from collections import Counter

    class _Win(io.RawIOBase):                                # seekable view of a stored nested zip, read in place
        def __init__(self, path, start, size): self.f, self.s, self.n, self.p = open(path, "rb"), start, size, 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.p
        def seek(self, o, w=io.SEEK_SET):
            self.p = max(0, min(o if w == io.SEEK_SET else self.p + o if w == io.SEEK_CUR else self.n + o, self.n)); return self.p
        def readinto(self, b):
            k = min(len(b), self.n - self.p)
            if k <= 0: return 0
            self.f.seek(self.s + self.p); d = self.f.read(k); b[:len(d)] = d; self.p += len(d); return len(d)
        def close(self):
            if not self.closed: self.f.close()
            super().close()

    def archives(path):                                      # the zip itself + any stored nested zip (one level)
        outer = zipfile.ZipFile(path); yield str(path), outer
        for info in outer.infolist():
            if info.filename.lower().endswith(".zip"):
                if info.compress_type != zipfile.ZIP_STORED:
                    print(f"  cannot read compressed nested zip in place: {info.filename} ({info.file_size/1e9:.2f} GB)"); continue
                with open(path, "rb") as r:
                    r.seek(info.header_offset); h = r.read(30)
                n, e = struct.unpack("<HH", h[26:30])
                w = _Win(path, info.header_offset + 30 + n + e, info.file_size); z = zipfile.ZipFile(w)
                yield f"{path.name}::{info.filename}", z; z.close(); w.close()
        outer.close()

    module_of = {t: m for m, ts in tables.items() for t in ts}
    other_modules = {"ed", "note", "ecg", "cxr", "hosp", "icu"}
    rx = re.compile(r"(?:^|/)(?:([^/]+)/)?([a-z0-9_]+)\.csv(?:\.gz)?$", re.I)
    found, seen = {}, []                                     # table -> (label, info)
    zip_path = Path(zip_path)
    for label, z in archives(zip_path):
        names = z.infolist(); seen.append((label, names))
        for info in names:
            m = rx.search(info.filename)
            if not m: continue
            parent, tab = (m.group(1) or "").lower(), m.group(2).lower()
            if tab in module_of and tab not in found and (parent == module_of[tab] or parent not in other_modules):
                found[tab] = (label, info)
    total = 0
    with zipfile.ZipFile(zip_path) as outer:
        for tab, (label, info) in sorted(found.items(), key=lambda kv: (module_of[kv[0]], kv[0])):
            mod = module_of[tab]
            out = Path(dest) / mod / Path(info.filename).name
            state = "skip (already there)" if out.exists() and out.stat().st_size == info.file_size else ("would extract" if dry_run else "extracting")
            print(f"{mod}/{tab:<16} {info.file_size/1e9:6.2f} GB  {state}   <- {label} :: {info.filename}")
            total += info.file_size
            if not dry_run and state == "extracting":
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(out.suffix + ".part")
                for lab2, z2 in archives(zip_path):
                    if lab2 == label:
                        with z2.open(info) as src, open(tmp, "wb") as dst: shutil.copyfileobj(src, dst, chunk)
                        break
                tmp.rename(out)
    missing = sorted(set(module_of) - set(found))
    print(f"\ntables found: {len(found)}/{len(module_of)} | total {total/1e9:.1f} GB | missing: {missing or 'none'}")
    if not found:
        print("\nNo table matched. What is inside the zip (send this output):")
        for label, names in seen:
            print(f"- {label}: {len(names)} members; extensions: {dict(Counter(Path(i.filename).suffix.lower() or '(none)' for i in names).most_common(6))}")
            for i in names[:25]: print("    ", i.filename, f"({i.file_size/1e6:.1f} MB)")
    return found
