"""Extensions for longitudinalProgressionDisease:
   (1) stronger/alternative image encoders, (2) MIMIC-IV tabular features, (3) image+tabular fusion.
Nothing here modifies the original model/ or data_loader/ packages."""
DISEASES = ["Pleural effusion", "Cardiomegaly", "Edema", "Pneumonia", "Pulmonary edema"]
CLASS_NAMES = ["improving", "stable", "worsening"]
