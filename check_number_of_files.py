from pathlib import Path

# Since there are many doubled folders, we checked first and used the one where it has the most p10 patients
def count_folders(directory):
    return sum(item.is_dir() for item in directory.iterdir())

# Check the number of DICOM images and reports for each patient folder
def check_num_dcm_report(path):
    total_dcm = 0
    total_reports = 0
    num_patients = 0

    for patient in sorted(path.iterdir()):
        if not patient.is_dir():
            continue
        num_patients += 1
        num_dcm = sum(1 for file in patient.rglob("*.dcm") if file.is_file())
        num_reports = sum(1 for file in patient.rglob("*.txt") if file.is_file())

        total_dcm += num_dcm
        total_reports += num_reports

        print(f"{patient.name}: "
        f"{num_dcm} DICOM images, {num_reports} reports")
    print(f"\nTotal patients: {num_patients}")
    print(f"Total DICOM images: {total_dcm}")
    print(f"Total reports: {total_reports}")


if __name__ == "__main__":
    path1 = Path("/mnt/c/Users/Zila/Downloads/BN5212/MIMIC-CXR/p10/p10/")
    # path2 = Path("/mnt/c/Users/Zila/Downloads/BN5212/MIMIC-CXR/p10_2/p10/")
    print("Folders in path1:", count_folders(path1))
    # print("Folders in path2:", count_folders(path2))
    # Folders in path1: 1000
    # Folders in path2: 1000
    check_num_dcm_report(path1)

