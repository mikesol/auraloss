import tarfile

file_path = 'SignalTrain_LA2A_Dataset_1.1.tgz'
extract_folder = 'data'

# Open the .tgz file
with tarfile.open(file_path, 'r:gz') as tar:
    # Extract all files to the specified folder
    tar.extractall(extract_folder)

print(f"Extraction completed. Files extracted to {extract_folder}.")