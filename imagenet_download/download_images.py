# from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve
import cv2 as cv


def get_image(joined_input):  # line, url, output_folder
    line, url, output_folder = joined_input.split(' ')

    accepted_filenames = ['jpg', 'png', 'gif']

    if url[-3:].lower() not in accepted_filenames:
        return [f"{line}|Error: Invalid url (end of line)", False]

    # Remove invalid characters to make the filename valid
    filename = str(output_folder) + "\\" + url.split('/')[-1].strip() \
        .replace('=', '').replace('?', '').replace('&', '').replace('*', '')
    filename = Path(filename[0:-4].replace('.', '') + filename[-4:])

    # Check if the file already exists
    if filename.exists():
        return [f"{line}|Already exists", filename]
    else:
        try:  # Attempt to download the file
            urlretrieve(url, str(filename))
            # wget.download(url, str(filename))
        except HTTPError as e:  # If bad link, skip it
            return [f"{line}|HTTP error: {e}", False]
        except Exception as e:
            return [f"{line}|Unhandled error: {e}", False]

        return [f"{line}|Downloaded!", filename]  # The image has successfully been downloaded


def download_images(url_list_file, output_folder, max_threads=None):
    """ Downloads images from a list of urls in a txt file

    :param max_threads: The threads to use for downloads
    :type max_threads: int
    :param url_list_file: The location of the file containing a url per line
    :type url_list_file: pathlib.Path or string
    :param output_folder: A folder name to put all of the images
    :type output_folder: : pathlib.Path or string
    """
    # Ensure that filename and output_folder are in pathlib.Path format
    url_list_file = Path(url_list_file)
    output_folder = Path(output_folder)

    # Check if output_folder exists, and make it if it doesn't
    if not output_folder.exists():
        output_folder.mkdir()

    with url_list_file.open(encoding='utf8') as file:
        url_list = [f"{cnt} {i.strip()} {str(output_folder)}" for cnt, i in enumerate(file.readlines())]
        print(f"{len(url_list)} lines found!")

        # Start a thread pool to run through the downloads in parallel (with as many threads as allowed)
        with PoolExecutor(max_workers=max_threads) as executor:
            cv.namedWindow("Display", cv.WINDOW_GUI_EXPANDED)
            for i, result in executor.map(get_image, url_list):
                if result is not False:  # If the download was a success
                    try:  # Just in case, as some files end up having weird stuff in them
                        # Imshow the file and decide if it's acceptable for the dataset
                        print(i, end=' | ')
                        result = Path(result)
                        image = cv.imread(str(result))
                        if image is None:  # Image is invalid
                            result.unlink()  # Delete the file
                            print("Unreadable image ='(")
                            continue
                        cv.imshow("Display", image)  # Show the image cause it looks cool
                        cv.waitKey(1)  # Let the image show
                        print("Image is good!")
                    except:
                        print("Something went wrong, oh well")


if __name__ == '__main__':
    download_images("spanner.txt", "spanner")
    download_images("screwdriver.txt", "screwdriver")
