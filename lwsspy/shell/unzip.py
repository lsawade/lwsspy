import zipfile


def unzip(file: str, destination: str):
    """Unzip a file

    Parameters
    ----------
    file : str
        Zip file to extract
    destination : str
        Destination directory

    """
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(destination)
