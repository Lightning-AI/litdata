def filter_lock_files(files):
    return [f for f in files if not f.endswith(".lock")]