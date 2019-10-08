from urllib.request import urlretrieve
from pathlib import Path
import csv
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import os
from tqdm import tqdm
import mmap  # For finding amount of lines in file
import awscli


def progress_bar(count, block_size, total_size, bar_length=30):
    current_data = count * block_size
    progress = current_data / total_size

    if progress == 0:
        the_bar = "-" * bar_length
    elif progress >= 1:
        the_bar = "+" * bar_length
    else:
        the_bar = "+" * int(bar_length * progress) + "-" * int(bar_length * (1 - progress))

    while len(the_bar) < bar_length:
        the_bar += '-'

    print(f"\r|{the_bar}| {current_data / 1000:.1f}/{total_size / 1000:.1f} kb", end="")


def download_file(url, sub_folder):
    file = Path(url.split('/')[-1])
    if not file.exists():
        print(f"Downloading: {str(file)}")
        urlretrieve(url, filename=f"{sub_folder}{str(file)}", reporthook=progress_bar)
        print()
    else:
        print(f"Already exists: {str(file)}")


def google_open_images_download(folder, mode, annotations_bbox, occluded, truncated, groupOf, depiction, inside):    # Spread out the variable so it's easier to follow the flow of the function
    ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside = annotations_bbox

    # Check if the image is actually wanted
    if not occluded and IsOccluded:
        return "No download: Occluded"
    if not truncated and IsTruncated:
        return "No download: Truncated"
    if not groupOf and IsGroupOf:
        return "No download: Group of"
    if not depiction and IsDepiction:
        return "No download: Depiction"
    if not inside and IsInside:
        return "No download: Inside"

    command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'+mode+'/'+ImageID+'.jpg '+ folder+'/'+ImageID+'.jpg'
    os.system(command)
    return "Downloaded!"

# 's3://open-images-dataset/'+'train'+'/'+'000026e7ee790996'+'.jpg '+ 'train'+'/'+'/m/07j7r'+'/'+'000026e7ee790996'+'.jpg'
#

def mapcount(filename):  # Gets number of lines in file
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines


def download_classes(class_names, mode, occluded=False, truncated=False, groupOf=False, depiction=False, inside=False, max_threads=4):
    mode = ["train", "test", "validation"][mode]
    needed_files = ["https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
                    "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
                    "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv",
                    "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv "]

    # Download the index files
    for i in needed_files:
        download_file(i, "")

    print("Getting class names")
    image_directory = Path(f"{mode}-annotations-bbox.csv")
    image_lines = mapcount(str(image_directory))

    descriptions = Path("class-descriptions-boxable.csv")
    with descriptions.open(mode='r') as file:
        reader = csv.reader(file)
        dict_list = {rows[1]: rows[0] for rows in reader}  # Sort all of the class names
        dict_list = {dict_list[i]: i for i in dict_list if i in class_names}  # Filter out only the relevant names

    print("Getting image list")

    # Make the mode folder if it doesn't exist
    mode_dir = Path(mode)
    if not mode_dir.exists():
        mode_dir.mkdir()

    print("Starting thread pool")
    # Start a thread pool to run through the downloads in parallel (with as many threads as allowed)
    with PoolExecutor(max_workers=max_threads) as executor:
        processing_list = []
        with image_directory.open() as file:
            reader = csv.reader(file)
            for i in tqdm(reader, total=image_lines):
                if i[2] in dict_list:
                    processing_list.append(executor.submit(google_open_images_download, f"{str(mode_dir)}/{dict_list[i[2]]}", mode, i, occluded, truncated, groupOf, depiction, inside))


        print("Output list:")
        for i in tqdm(processing_list):  # Go through each to check if they worked
            print(f" {i.result()}")


if __name__ == '__main__':
    download_classes(["Screwdriver", "Wrench"], 0, max_threads=6, occluded=True, truncated=True, groupOf=True, depiction=True, inside=True)