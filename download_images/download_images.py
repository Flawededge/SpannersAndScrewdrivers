import csv
import multiprocessing
import subprocess
from pathlib import Path
import argparse
import csv
import subprocess
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool as thread_pool
import wget

screwdriver_id = "n04154565"  # Any screwdrivers
wrench_id = "n02680754"  # Crescent wrenches

# Required files. [[Filename to check], [Url if missing]]
index_files = [
    [Path("class-descriptions-boxable.csv"),
     "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"],
    [Path("train-annotations-bbox.csv"),
     "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv"],
    [Path("validation-annotations-bbox.csv"),
     "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv"],
    [Path("test-annotations-bbox.csv"),
     "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv"]
]


# Loading bar for wGet
def bar_custom(current, total, width=80):
    print(
        "\rDownloading: %d%% [%d / %d] Mb     " % (current / total * 100, int(current / 1000000), int(total / 1000000)),
        end='')


# The actual download images function
def download_images(mode, classes, nthreads=multiprocessing.cpu_count()*2, occluded=1, truncated=1, groupOf=1,
                    depiction=1, inside=1):
    """ The actual download images function
    :param mode: Dataset category - train, validation or test
    :param classes: Names of object classes to be downloaded
    :param nthreads: Number of threads to use
    :type nthreads: int
    :param occluded: Include occluded images
    :type occluded: int
    :param truncated: Include truncated images
    :type truncated: int
    :param groupOf: Include groupOf images
    :type groupOf: int
    :param depiction: Include depiction images
    :type depiction: int
    :param inside: Include inside images
    :type inside: int
    :return:
    """
    # # Initial parameter setup ----------------------------------------------------------------------------------------
    # Check if the index files are present
    for i in index_files:
        if i[0].exists():
            print(f"{i[0]} found!")
        else:
            print(f"{i[0]} not found, downloading")
            wget.download(i[1], bar=bar_custom)
            print()

    # Thread setup
    threads = nthreads
    if nthreads > multiprocessing.cpu_count()*2:
        print(f"Error, {nthreads} is more threads than you have =(. ({multiprocessing.cpu_count()*2} is max)")
        nthreads = multiprocessing.cpu_count()*2

    # Read the class descriptions
    with open('./class-descriptions-boxable.csv', mode='r') as infile:
        reader = csv.reader(infile)
        dict_list = {rows[1]: rows[0] for rows in reader}

    # subprocess.run(['rm', '-rf', mode])
    # subprocess.run(['mkdir', mode])
    if not os.path.isdir(mode):
        os.mkdir(mode)

    pool = thread_pool(threads)
    commands = []
    cnt = 0

    print("Start loop")
    for ind in range(0, len(classes)):
        class_name = classes[ind]
        print("Class "+str(ind) + " : " + class_name)

        if not os.path.isdir(mode+'/'+class_name):
            os.mkdir(mode+'/'+class_name)

        current_class = dict_list[class_name.replace('_', ' ')]  # Contains the code for the list
        command = "grep "+dict_list[class_name.replace('_', ' ')], " /" + mode + "-annotations-bbox.csv"
        class_annotations = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        class_annotations = class_annotations.splitlines()

        for line in class_annotations:

            line_parts = line.split(',')

            #IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
            if occluded==0 and int(line_parts[8])>0:
                print("Skipped %s",line_parts[0])
                continue
            if truncated==0 and int(line_parts[9])>0:
                print("Skipped %s",line_parts[0])
                continue
            if groupOf==0 and int(line_parts[10])>0:
                print("Skipped %s",line_parts[0])
                continue
            if depiction==0 and int(line_parts[11])>0:
                print("Skipped %s",line_parts[0])
                continue
            if inside==0 and int(line_parts[12])>0:
                print("Skipped %s",line_parts[0])
                continue

            cnt = cnt + 1

            command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'+run_mode+'/'+line_parts[0]+'.jpg '+ run_mode+'/'+class_name+'/'+line_parts[0]+'.jpg'
            commands.append(command)

            with open('%s/%s/%s.txt'%(mode,class_name,line_parts[0]),'a') as f:
                f.write(','.join([class_name, line_parts[4], line_parts[5], line_parts[6], line_parts[7]])+'\n')

        print("Annotation Count : "+str(cnt))
        commands = list(set(commands))
        print("Number of images to be downloaded : "+str(len(commands)))

        list(tqdm(pool.imap(os.system, commands), total = len(commands) ))

        pool.close()
        pool.join()


if __name__ == '__main__':
    download_images('train', ['Ice_cream', 'cookie'])
