import tarfile


def untar(fname, path=None):
    """Untars a tar archive and extracts it to path."""

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=path)
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path=path)
        tar.close()
