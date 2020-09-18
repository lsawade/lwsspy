

def readfile(filename: str):
    """Takes in a string with a filename and outputs the content as list
    of linestrings

    Args:
        filename (str): srting with filename

    Returns:
        list[str]: List of line strings.
    """
    # Open, read, return
    with open(filename, 'r') as f:
        lines = f.readlines()
        content = ("").join(lines) + "\n"
    return content
