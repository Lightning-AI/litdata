def filter_lock_files(files):
    return [f for f in files if not f.endswith((".lock", ".cnt"))]


def get_lock_files(files):
    return [f for f in files if f.endswith(".lock") or f.endswith("cnt")]
