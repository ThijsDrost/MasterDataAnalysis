import os
import pathlib
import functools
import warnings


def move_files_filtered(folder_loc, name_dest_dict, *, raise_exists=True, relative_path=True, filter_func='in'):
    """
    Split the files in the given folder into the given destinations.

    Parameters
    ----------
    folder_loc: str or pathlib.Path
        The location of the folder with the files to split.
    name_dest_dict: dict[str or pathlib.Path, str]
        A dictionary with the names of the new folders as keys and the value to match to files to as values.
    raise_exists: bool
        Whether to raise an error if the destination folder already exists and has files.
    relative_path:
        Whether the paths in `name_dest_dict` are relative to `folder_loc`, if False, the paths are assumed to be absolute.
    filter_func: str | callable
        The function to use to filter the files. Must be a string containing 'in', 'startswith', or 'endswith', or a callable.
        The callable should take two arguments, the first is  the string to filter by (the items in
        the `name_dest_dict`), the second is the file name.

    Notes
    -----
    Wrapper around :py:func:`filter_and_move_files`
    """
    for dest, name in name_dest_dict.items():
        if relative_path:
            path = os.path.join(folder_loc, dest)
        else:
            path = dest
        if os.path.exists(path):
            if raise_exists and os.listdir(path):
                raise FileExistsError(f"`{path}` already exists")
            if not os.path.isdir(path):
                raise ValueError(f"`{path}` is not a directory")

    filter_func = _set_filter_func(filter_func)

    for dest, name in name_dest_dict.items():
        filter_function = functools.partial(filter_func, name)
        filter_and_move_files(folder_loc, dest, relative_path=relative_path, filter_func=filter_function)


def filter_and_move_files(folder_loc, dest, relative_path=None, filter_func=None):
    """
    Filters and moves files from one location to another.

    Parameters
    ----------
    folder_loc : str or pathlib.Path
        The location of the folder with the files to filter and move.
    dest : str or pathlib.Path
        The destination folder where the filtered files will be moved.
    filter_func : str or callable
        The function to use to filter the files. Must be a string containing 'in', 'startswith', or 'endswith', or a callable.
        The callable should take one argument, the file name.
    relative_path : bool
        Whether the paths in `name_dest_dict` are relative to `folder_loc`, if False, the paths are assumed to be absolute. If
        None (default), the value is determined by the presence of a colon in `dest` (if a colon is present, assumed to be absolute).
    """
    files_moved_counter = 0

    if relative_path is None:
        if ':' in dest:
            relative_path = False

    if relative_path:
        path = os.path.join(folder_loc, dest)
    else:
        path = dest

    if not os.path.exists(path):
        os.mkdir(path)
    for file in os.listdir(folder_loc):
        if filter_func is None or filter_func(file):
            files_moved_counter += 1
            os.rename(os.path.join(folder_loc, file), os.path.join(path, file))

    if files_moved_counter == 0:
        warnings.warn(f"No files were moved from {folder_loc} to {dest}")


def _set_filter_func(filter_func):
    if callable(filter_func) or filter_func is None:
        return filter_func
    elif isinstance(filter_func, str):
        if filter_func in ['startswith', 'endswith']:
            def func(filter_str, file): return getattr(file, filter_func)(filter_str)
            return func
        elif filter_func == 'in':
            def func(filter_str, file): return filter_str in file
            return func
        else:
            raise ValueError(f"Valid string values for `filter_func` are 'in', 'startswith', and 'endswith', not {filter_func}")
    else:
        raise TypeError(f"`filter_func` must be a string of 'in', 'startswith', or 'endswith', or a callable, not {filter_func}"
                        f"of type {type(filter_func)}")
