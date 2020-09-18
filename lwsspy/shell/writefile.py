
def writefile(content: list or str, filename: str):
    """Takes in a string with a filename and write list of strings to the file.
    If lines is 

    Args:
        content (list or str):
            List of strings or string to write to file.
        filename (str): 
            string with filename
    """

    # Check for content type
    if type(content) == str:
        lines = content.splitlines()
    else:
        lines = content

    # Open, write
    with open(filename, 'w') as f:
        f.writelines(lines)
    return None
