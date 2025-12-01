import os


def normalize_file_url(path: str) -> str:
    """Convert file:// URL to local path, handling malformed Windows URLs."""
    if not path.startswith("file://"):
        return path

    path = path[7:]

    if os.name == "nt":
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith("/") and len(path) > 2 and path[2] in (":", "|"):
            path = path[1:]

        path = path.replace("/", "\\")
        path = path.replace("|", ":")
    else:
        if not path.startswith("/"):
            path = "/" + path

    return path
