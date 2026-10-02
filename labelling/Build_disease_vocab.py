#This is almost like Bibit's code in BN5212_Custom_Disease_Loader.ipynb to build a disease vocabulary from the MIMIC-IV dataset
#But i used the ICD-to-Phecode mapping from PheWAS instead of the one from the MIMIC-IV dataset, since it has more diseases and is more up to date
#So it's icd_code + icd_version --> phecode --> disease name

from pathlib import Path
import pandas as pd


def load_icd_diagnoses(path):
    """Load the ICD diagnoses from the MIMIC-IV dataset."""
    df = pd.read_csv(path, dtype={"icd_code": str})
    return df

def load_icd_to_phecode_mapping():
    """Load the ICD-to-Phecode mapping and disease names."""
    base = "https://raw.githubusercontent.com/PheWAS/PhecodeX/main/"
    mapping = pd.read_csv(base + "phecodeX_R_map.csv", dtype=str)
    labels = pd.read_csv(base + "phecodeX_R_labels.csv", dtype=str)
    return mapping, labels

def map_icd_to_phecode(df, mapping, labels):
    """Map ICD codes to Phecodes and retrieve disease names."""
    # Match the ICD version and normalize code formatting.
    df["vocabulary_id"] = df["icd_version"].map({
        9: "ICD9CM",
        10: "ICD10CM",
    })

    df["code"] = df["icd_code"].str.replace(".", "", regex=False).str.upper()
    mapping["code"] = (
        mapping["code"].str.replace(".", "", regex=False).str.upper()
    )

    mapped = df.merge(
        mapping,
        on=["vocabulary_id", "code"],
        how="left",
    )

    mapped = mapped.merge(
        labels[["phenotype", "description"]],
        left_on="phecode",
        right_on="phenotype",
        how="left",
    )

    return mapped

def drop_unmapped(mapped):
    """Drop unmapped ICD codes and return the unique disease names."""
    disease_list = sorted(mapped["description"].dropna().unique().tolist())
    return disease_list

if __name__ == "__main__":
    path = Path(
        "/mnt/c/Users/Zila/Downloads/BN5212/MIMIC-IV/3.1/hosp/d_icd_diagnoses.csv")
    df = load_icd_diagnoses(path)
    mapping, labels = load_icd_to_phecode_mapping()
    mapped = map_icd_to_phecode(df, mapping, labels)
    disease_list = drop_unmapped(mapped)

    print(f"Unique mapped disease/phenotype names: {len(disease_list)}")
    print(disease_list[:30])

    pd.DataFrame({"disease": disease_list}).to_csv(
        "unique_diseases.csv", index=False)

    # Check which ICD codes were not covered by the mapping, but i guess we can ignore for now
    unmapped = mapped.loc[
        mapped["description"].isna(),
        ["icd_version", "icd_code", "long_title"],
    ].drop_duplicates()

    print(f"ICD codes without a mapped name: {len(unmapped)}")
    unmapped.to_csv("unmapped_diagnoses.csv", index=False)
