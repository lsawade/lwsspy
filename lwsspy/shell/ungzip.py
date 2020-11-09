import gzip
import shutil

def ungzip(file: str, outfile: str):
    with gzip.open(file, 'rb') as f_in:
        with open(outfile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
