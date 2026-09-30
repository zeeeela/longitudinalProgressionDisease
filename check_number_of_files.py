from pathlib import Path

# Since there are many doubled folders, we checked first and used the one where it has the most p10 patients
def count_folders(directory):
    return sum(item.is_dir() for item in directory.iterdir())

# Check the overlap of these folders, if there is an overlap, we will not use the second path
def check_folder_overlap(path1, path2):
    folders1 = {item.name for item in path1.iterdir() if item.is_dir()}
    folders2 = {item.name for item in path2.iterdir() if item.is_dir()}
    overlap = folders1 & folders2

    if overlap:
        print(f"Found {len(overlap)} shared folder names:")
        '''for name in sorted(overlap):
            print(name)'''
        return False

    print("No shared folder names between the two paths.")
    return True

if __name__ == "__main__":
    path1 = Path("/mnt/c/Users/Zila/Downloads/BN5212/MIMIC-CXR/p10_1/p10/")
    path2 = Path("/mnt/c/Users/Zila/Downloads/BN5212/MIMIC-CXR/p10_2/p10/")
    print("Folders in path1:", count_folders(path1))
    # print("Folders in path2:", count_folders(path2))
    # Folders in path1: 6397
    # Folders in path2: 1000
    # check_folder_overlap(path1, path2)
    # Found 1000 shared folder names
    # So we used the first path since it has more p10 patients

