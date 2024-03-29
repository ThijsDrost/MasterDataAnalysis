"""
This script is used to generate the latex code for the project structure of this project in an itemize environment.
"""

import os


def structure_to_itemize(loc, skip_names=('.idea', '.git', '__init__', '__pycache__')):
    """
    Convert a directory structure to a latex itemize environment

    Parameters
    ----------
    loc

    Returns
    -------

    """
    def sort_func(x):
        if os.path.isdir(x):
            return '0' + x.split('\\')[-1]
        else:
            return '1' + x.split('\\')[-1]

    def check_skip(name):
        return any([(skip in name) for skip in skip_names])

    def _structure_to_itemize(loc, depth=0):
        if depth == 1:
            item_thingy = '$\\rightarrow$'
        else:
            item_thingy = '$\\drsh$'

        if depth == 0:
            around = ('', '')
        else:
            around = (rf"\item[{item_thingy}] {loc.split('\\')[-1]}" + '\n' + '\\begin{itemize}\n',
                      '\n\\end{itemize}')

        if os.path.isfile(loc):
            return rf"\item[{item_thingy}] {loc.split('\\')[-1]}"
        else:
            sorted_list = sorted((os.path.join(loc, item) for item in os.listdir(loc)), key=sort_func)
            return around[0] + '\n'.join([_structure_to_itemize(os.path.join(loc, item), depth + 1) for item in sorted_list if not check_skip(item)]) + around[1]
    result = f"\\begin{{itemize}}[noitemsep]\n{_structure_to_itemize(loc)}\n\\end{{itemize}}"

    # without the directory loc

    return result.replace('_', '\\_')


result = structure_to_itemize(r"C:\Users\20222772\PycharmProjects\MasterDataAnalysis")
data_analysis = structure_to_itemize(r"C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Data_analysis")
general = structure_to_itemize(r"C:\Users\20222772\PycharmProjects\MasterDataAnalysis\General")
Guis = structure_to_itemize(r"C:\Users\20222772\PycharmProjects\MasterDataAnalysis\Guis")


