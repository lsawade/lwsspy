import os
import urllib.request
import threading
from typing import List


def downloadfile(url: str, floc: str):
    """Downloads file to location

    Parameters
    ----------
    url : str
        Source URL
    floc : str
        Destination
    """
    try:
        urllib.request.urlretrieve(url, floc)
    except Exception as e:
        print(f"Error when downloading {url}: {e}")


def download_threaded(urls: List[str], destination: str, mult: bool = True):
    """Downloading multiple files using multithreading.

    Parameters
    ----------
    urls : List[str]
        List of urls to download
    destination : str
        destination directory
    """

    # Create destination list from url list
    destination_list = []
    for _url in urls:
        destination_list.append(os.path.join(
            destination, os.path.basename(_url)))

    # Number of tasks
    nURL = len(urls)

    # Threaded download!
    threads = []
    for _i, (_url, _dest) in enumerate(zip(urls, destination_list)):
        if mult:
            threads.append(
                threading.Thread(target=downloadfile, args=(_url, _dest)))
        else:
            downloadfile(_url, _dest)
            print(f"{100 * (_i+1)/nURL:>5.1f}%", end="\r")

    if mult:
        # Execution
        for t in threads:
            t.start()

        # Finishing
        for _i, t in enumerate(threads):
            t.join()
            print(f"{100 * (_i+1)/nURL:>5.1f}%", end="\r")
