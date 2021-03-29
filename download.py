### Module imports ###
import os
import requests
import shutil
import subprocess


### Global Variables ###


### Class declarations ###


### Function declarations ###
def download_and_extract(url, save_path):
    """
    Download and extract zip file to folder

    Args:
        url: url to download zip file
        save_path: folder to save data
    """
    local_filename = url.split('/')[-1]
    file_path = os.path.join(save_path, local_filename)
    folder_path = os.path.join(save_path, '.'.join(local_filename.split('.')[:-1]))
    # download file
    with requests.get(url, stream=True) as r:
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # create folder to extract
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # unzip
    subprocess.run(['unzip', file_path, '-d', folder_path])

    # remove zip file
    subprocess.run(['rm', file_path])


if __name__ == '__main__':
    download_and_extract('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip', 'data')
    download_and_extract('http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip', 'data')
