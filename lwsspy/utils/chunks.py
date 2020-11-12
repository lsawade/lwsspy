

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Useful for multi-threading"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]