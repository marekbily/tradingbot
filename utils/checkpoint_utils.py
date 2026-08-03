import os
import glob


def find_latest_checkpoint(folder='train/dreamer_ultimate', pattern='*.pt'):
    """Return the latest checkpoint path in a folder or None if not found."""
    if not os.path.isdir(folder):
        return None
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def ensure_checkpoint_path(path_or_none, default_folder='train/dreamer_ultimate'):
    """If path_or_none exists, return it. Otherwise try to find latest in default_folder."""
    if path_or_none and os.path.exists(path_or_none):
        return path_or_none
    ckpt = find_latest_checkpoint(default_folder)
    return ckpt
