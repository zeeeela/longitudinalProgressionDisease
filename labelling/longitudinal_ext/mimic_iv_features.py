"""Time-aligned MIMIC-IV features for each (prior, current) CXR pair.

LEAKAGE RULES (enforced in code):
  * only events with time <= the study time are used (labs: 48 h look-back; vitals/fluids/new drug starts: 24 h);
  * NEVER used: diagnoses_icd / procedures_icd, discharge notes, discharge time, death date, length of stay.
MIMIC-CXR and MIMIC-IV share the same per-patient date shift, so StudyDate+StudyTime (DICOM) can be matched to
event times. Check the coverage report before trusting the features.

Feature names:  <name>_cur (at current study) | <name>_prev (at prior study) | <name>_delta (cur - prev)
                static_*   (age at current study, sex, log gap)     interval_* (drug starts between the two studies)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

H = 3600 * 10**9  # ns per hour

LAB_GROUPS = {  # label regex (matched on d_labitems.label, fluid == Blood)
    "ntprobnp": r"^NT-proBNP$", "wbc": r"^White Blood Cells$", "creatinine": r"^Creatinine$",
    "bun": r"^Urea Nitrogen$", "crp": r"^C-Reactive Protein$", "troponin_t": r"^Troponin T$",
    "lactate": r"^Lactate$", "hemoglobin": r"^Hemoglobin$", "albumin": r"^Albumin$", "sodium": r"^Sodium$",
}
VITAL_GROUPS = {  # label regex on icu/d_items.label (chartevents)
    "hr": r"^Heart Rate$", "rr": r"^Respiratory Rate$", "spo2": r"^O2 saturation pulseoxymetry$",
    "sbp": r"^(Arterial Blood Pressure systolic|Non Invasive Blood Pressure systolic)$",
    "temp_c": r"^Temperature Celsius$", "fio2": r"^Inspired O2 Fraction$",
    "weight": r"^(Daily Weight|Admission Weight \(Kg\))$",
}
PLAUSIBLE = {"hr": (20, 250), "rr": (4, 80), "spo2": (50, 100), "sbp": (40, 300), "temp_c": (30, 43),
             "fio2": (21, 100), "weight": (20, 400), "wbc": (0, 500), "creatinine": (0, 30), "bun": (0, 300),
             "hemoglobin": (2, 25), "albumin": (0.5, 6), "sodium": (100, 180), "lactate": (0, 30),
             "crp": (0, 600), "ntprobnp": (0, 200000), "troponin_t": (0, 50)}
DRUG_GROUPS = {
    "diuretic": r"furosemide|bumetanide|torsemide|metolazone|chlorothiazide|ethacryn",
    "antibiotic": r"vancomycin|piperacillin|cefepime|ceftriaxone|ceftazidime|meropenem|imipenem|azithromycin|"
                  r"levofloxacin|ciprofloxacin|ampicillin|amoxicillin|doxycycline|linezolid|clindamycin|"
                  r"metronidazole|cefazolin|aztreonam|gentamicin",
}
LAB_LOOKBACK_H, WINDOW_H = 48, 24


# ----------------------------------------------------------------------------- file helpers
def _find(root, *rel):
    base = Path(root).joinpath(*rel)
    for ext in (".csv", ".csv.gz"):
        p = Path(str(base) + ext)
        if p.exists():
            return p
    return None


def _read_filtered(path, usecols, subjects, itemids=None, chunksize=4_000_000, extra_filter=None):
    parts = []
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        m = ch["subject_id"].isin(subjects)
        if itemids is not None:
            m &= ch["itemid"].isin(itemids)
        ch = ch[m]
        if extra_filter is not None and len(ch):
            ch = extra_filter(ch)
        if len(ch):
            parts.append(ch)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def _ns(series):
    return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]").astype("int64")


def _cache(cache_dir, name, fn):
    f = Path(cache_dir) / f"{name}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    df = fn()
    f.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(f)
    return df


# ----------------------------------------------------------------------------- study times from DICOM
def read_study_times(manifest, input_col, target_col, cache_csv=None):
    """Return DataFrame[subject_id, study_id, study_time(ns int64)] from DICOM StudyDate/StudyTime."""
    import pydicom
    if cache_csv and Path(cache_csv).exists():
        return pd.read_csv(cache_csv, dtype={"subject_id": str, "study_id": str})
    rows = []
    for sid_col, p_col in (("input_study_id", input_col), ("target_study_id", target_col)):
        sub = manifest[["subject_id", sid_col, p_col]].drop_duplicates(subset=["subject_id", sid_col])
        for subj, study, path in sub.itertuples(index=False):
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            d, t = str(getattr(ds, "StudyDate", "")), str(getattr(ds, "StudyTime", "000000")).split(".")[0].ljust(6, "0")
            ts = pd.to_datetime(d + t[:6], format="%Y%m%d%H%M%S", errors="coerce")
            rows.append((str(subj), str(study), np.int64(ts.as_unit("ns").value) if pd.notna(ts) else np.nan))
    out = pd.DataFrame(rows, columns=["subject_id", "study_id", "study_time"]).drop_duplicates(["subject_id", "study_id"])
    if cache_csv:
        out.to_csv(cache_csv, index=False)
    return out


# ----------------------------------------------------------------------------- event stores
class Series:
    """(subject, group) -> sorted times (ns) and values."""

    def __init__(self, df, group="group", time="time", value="value"):
        self.d = {}
        if df is None or df.empty:
            return
        df = df.dropna(subset=[time, value])
        df = df[df[time] > 0].sort_values(time)  # NaT parses to a negative int
        for (s, g), part in df.groupby(["subject_id", group], sort=False):
            self.d[(str(s), g)] = (part[time].to_numpy("int64"), part[value].to_numpy(float))

    def last(self, s, g, t, lookback_h):
        x = self.d.get((s, g))
        if x is None:
            return np.nan
        i = np.searchsorted(x[0], t, side="right") - 1
        return x[1][i] if i >= 0 and t - x[0][i] <= lookback_h * H else np.nan

    def window(self, s, g, t, hours, how="median"):
        x = self.d.get((s, g))
        if x is None:
            return np.nan
        lo, hi = np.searchsorted(x[0], t - hours * H, side="right"), np.searchsorted(x[0], t, side="right")
        v = x[1][lo:hi]
        if not len(v):
            return np.nan
        return float(np.median(v) if how == "median" else v.sum())


class Intervals:
    """(subject, group) -> start/stop arrays for drugs, admissions, ICU stays."""

    def __init__(self, df, group="group", start="start", stop="stop"):
        self.d = {}
        if df is None or df.empty:
            return
        df = df.dropna(subset=[start])
        df = df[df[start] > 0]
        for (s, g), part in df.groupby(["subject_id", group], sort=False):
            st = part[start].to_numpy("int64")
            sp = part[stop].to_numpy(dtype="float64")
            sp = np.where(np.isnan(sp) | (sp <= 0), np.inf, sp)  # open-ended / missing stop
            o = np.argsort(st)
            self.d[(str(s), g)] = (st[o], sp[o])

    def active(self, s, g, t):
        x = self.d.get((s, g))
        return float(np.any((x[0] <= t) & (x[1] > t))) if x else 0.0

    def starts_between(self, s, g, ta, tb):
        x = self.d.get((s, g))
        return float(((x[0] > ta) & (x[0] <= tb)).sum()) if x else 0.0


# ----------------------------------------------------------------------------- loaders per source
def _load_labs(root, subjects, cache):
    dl, ev = _find(root, "hosp", "d_labitems"), _find(root, "hosp", "labevents")
    if dl is None or ev is None:
        return None
    items = pd.read_csv(dl)
    items = items[items["fluid"].astype(str).str.lower() == "blood"]
    id2g = {}
    for g, rx in LAB_GROUPS.items():
        for i in items.loc[items["label"].astype(str).str.match(rx, flags=re.I), "itemid"]:
            id2g[int(i)] = g
    print("lab itemids:", {g: sorted(i for i, gg in id2g.items() if gg == g) for g in LAB_GROUPS})

    def go():
        df = _read_filtered(ev, ["subject_id", "itemid", "charttime", "valuenum"], subjects, list(id2g))
        df["group"] = df["itemid"].map(id2g)
        df["time"], df["value"] = _ns(df["charttime"]), df["valuenum"].astype(float)
        return df[["subject_id", "group", "time", "value"]]
    return _cache(cache, "labs", go)


def _load_vitals(root, subjects, cache):
    di, ev = _find(root, "icu", "d_items"), _find(root, "icu", "chartevents")
    if di is None or ev is None:
        return None
    items = pd.read_csv(di)
    id2g = {}
    for g, rx in VITAL_GROUPS.items():
        for i in items.loc[items["label"].astype(str).str.match(rx, flags=re.I), "itemid"]:
            id2g[int(i)] = g
    print("vital itemids:", {g: sorted(i for i, gg in id2g.items() if gg == g) for g in VITAL_GROUPS})

    def go():
        df = _read_filtered(ev, ["subject_id", "itemid", "charttime", "valuenum"], subjects, list(id2g))
        df["group"] = df["itemid"].map(id2g)
        df["time"], df["value"] = _ns(df["charttime"]), df["valuenum"].astype(float)
        fi = df["group"] == "fio2"
        df.loc[fi & (df["value"] <= 1.0), "value"] *= 100.0
        return df[["subject_id", "group", "time", "value"]]
    return _cache(cache, "vitals", go)


def _load_fluids(root, subjects, cache):
    inp, outp = _find(root, "icu", "inputevents"), _find(root, "icu", "outputevents")
    if inp is None or outp is None:
        return None, None

    def go_in():
        df = _read_filtered(inp, ["subject_id", "starttime", "amount", "amountuom"], subjects,
                            extra_filter=lambda c: c[c["amountuom"].astype(str).str.lower() == "ml"])
        return pd.DataFrame({"subject_id": df["subject_id"], "group": "fluid_in",
                             "time": _ns(df["starttime"]), "value": df["amount"].astype(float)})

    def go_out():
        df = _read_filtered(outp, ["subject_id", "charttime", "value", "valueuom"], subjects,
                            extra_filter=lambda c: c[c["valueuom"].astype(str).str.lower() == "ml"])
        return pd.DataFrame({"subject_id": df["subject_id"], "group": "fluid_out",
                             "time": _ns(df["charttime"]), "value": df["value"].astype(float)})
    return _cache(cache, "fluid_in", go_in), _cache(cache, "fluid_out", go_out)


def _load_drugs(root, subjects, cache):
    p = _find(root, "hosp", "prescriptions")
    if p is None:
        return None

    def go():
        df = _read_filtered(p, ["subject_id", "starttime", "stoptime", "drug"], subjects)
        parts = []
        for g, rx in DRUG_GROUPS.items():
            m = df["drug"].astype(str).str.contains(rx, case=False, regex=True)
            parts.append(pd.DataFrame({"subject_id": df.loc[m, "subject_id"], "group": g,
                                       "start": _ns(df.loc[m, "starttime"]), "stop": _ns(df.loc[m, "stoptime"])}))
        out = pd.concat(parts, ignore_index=True)
        out.loc[out["stop"] < 0, "stop"] = np.nan  # NaT -> NaN
        return out
    return _cache(cache, "drugs", go)


def _load_stays(root, subjects, cache):
    out = []
    for kind, rel, a, b in (("inpatient", ("hosp", "admissions"), "admittime", "dischtime"),
                            ("icu", ("icu", "icustays"), "intime", "outtime")):
        p = _find(root, *rel)
        if p is not None:
            df = pd.read_csv(p, usecols=["subject_id", a, b])
            df = df[df["subject_id"].isin(subjects)]
            out.append(pd.DataFrame({"subject_id": df["subject_id"], "group": kind,
                                     "start": _ns(df[a]), "stop": _ns(df[b])}))
    return pd.concat(out, ignore_index=True) if out else None


# ----------------------------------------------------------------------------- main builder
def build_pair_features(manifest, input_col, target_col, mimic_root, cache_dir, out_csv=None):
    """manifest: pair DataFrame (needs subject_id, input_study_id, target_study_id + image path columns)."""
    mimic_root, cache_dir = Path(mimic_root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    m = manifest.copy()
    for k in ("subject_id", "input_study_id", "target_study_id"):
        m[k] = m[k].astype(str)
    subjects = set(pd.to_numeric(m["subject_id"]).astype(int))
    times = read_study_times(m, input_col, target_col, cache_dir / "study_times.csv")
    tmap = {(r.subject_id, r.study_id): r.study_time for r in times.itertuples()}

    labs, vitals = _load_labs(mimic_root, subjects, cache_dir), _load_vitals(mimic_root, subjects, cache_dir)
    f_in, f_out = _load_fluids(mimic_root, subjects, cache_dir)
    drugs, stays = _load_drugs(mimic_root, subjects, cache_dir), _load_stays(mimic_root, subjects, cache_dir)
    print("available sources:", {k: v is not None for k, v in dict(labs=labs, vitals=vitals, fluids=f_in,
                                                                     drugs=drugs, stays=stays).items()})
    pat = _find(mimic_root, "hosp", "patients")
    pat = pd.read_csv(pat, usecols=["subject_id", "gender", "anchor_age", "anchor_year"]).set_index("subject_id") if pat else None

    S_lab, S_vit = Series(labs), Series(vitals)
    S_in, S_out = Series(f_in), Series(f_out)
    I_drug, I_stay = Intervals(drugs), Intervals(stays)

    def snapshot(s, t):
        f = {}
        for g in LAB_GROUPS:
            f[g] = S_lab.last(s, g, t, LAB_LOOKBACK_H)
        for g in VITAL_GROUPS:
            f[g] = S_vit.window(s, g, t, WINDOW_H)
        fin, fout = S_in.window(s, "fluid_in", t, WINDOW_H, "sum"), S_out.window(s, "fluid_out", t, WINDOW_H, "sum")
        f["net_fluid_24h"] = np.nan if (np.isnan(fin) and np.isnan(fout)) else np.nan_to_num(fin) - np.nan_to_num(fout)
        for g in DRUG_GROUPS:  # NaN (not 0) when the source table is absent, so coverage shows it
            f[f"{g}_new24h"] = I_drug.starts_between(s, g, t - WINDOW_H * H, t) if drugs is not None else np.nan
            f[f"{g}_active"] = I_drug.active(s, g, t) if drugs is not None else np.nan
        f["inpatient"] = I_stay.active(s, "inpatient", t) if stays is not None else np.nan
        f["icu"] = I_stay.active(s, "icu", t) if stays is not None else np.nan
        return f

    rows = []
    for r in m.itertuples(index=False):
        s = r.subject_id
        t0, t1 = tmap.get((s, r.input_study_id), np.nan), tmap.get((s, r.target_study_id), np.nan)
        row = {"subject_id": s, "input_study_id": r.input_study_id, "target_study_id": r.target_study_id}
        if np.isnan(t0) or np.isnan(t1):
            rows.append(row)
            continue
        t0, t1 = int(t0), int(t1)
        prev, cur = snapshot(s, t0), snapshot(s, t1)
        for k in cur:
            row[f"{k}_cur"], row[f"{k}_prev"] = cur[k], prev[k]
            row[f"{k}_delta"] = cur[k] - prev[k] if not (np.isnan(cur[k]) or np.isnan(prev[k])) else np.nan
        row["static_gap_hours_log"] = float(np.log1p(max(t1 - t0, 0) / H))
        for g in DRUG_GROUPS:
            row[f"interval_n_new_{g}"] = float(np.log1p(I_drug.starts_between(s, g, t0, t1))) if drugs is not None else np.nan
        if pat is not None and int(s) in pat.index:
            p = pat.loc[int(s)]
            year = pd.Timestamp(t1).year
            row["static_age"] = float(p["anchor_age"] + (year - p["anchor_year"]))
            row["static_male"] = float(p["gender"] == "M")
        rows.append(row)
    out = pd.DataFrame(rows)
    out = _clip_implausible(out)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
    return out


def _clip_implausible(df):
    for g, (lo, hi) in PLAUSIBLE.items():
        for suf in ("_cur", "_prev"):
            c = g + suf
            if c in df:
                df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
        if f"{g}_delta" in df and f"{g}_cur" in df:
            df[f"{g}_delta"] = np.where(df[f"{g}_cur"].isna() | df[f"{g}_prev"].isna(), np.nan,
                                        df[f"{g}_cur"] - df[f"{g}_prev"])
    return df


def coverage_report(features, manifest=None):
    """Fraction of pairs with data per feature group: current snapshot, and both snapshots (delta usable)."""
    f = features.copy()
    for k in ("subject_id", "input_study_id", "target_study_id"):
        f[k] = f[k].astype(str)
    if manifest is not None and "split" in manifest:
        f = f.merge(manifest[["subject_id", "input_study_id", "target_study_id", "split"]].astype(
            {"subject_id": str, "input_study_id": str, "target_study_id": str}).drop_duplicates(
            ["subject_id", "input_study_id", "target_study_id"]),
            on=["subject_id", "input_study_id", "target_study_id"], how="left")
    bases = sorted({c[:-4] for c in f.columns if c.endswith("_cur")})
    rows = [dict(feature=b, cur=f[f"{b}_cur"].notna().mean(), prev=f[f"{b}_prev"].notna().mean(),
                 delta=f[f"{b}_delta"].notna().mean()) for b in bases]
    return pd.DataFrame(rows).set_index("feature").round(3).sort_values("delta", ascending=False)