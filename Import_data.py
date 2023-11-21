import numpy as np


def import_txt(loc):
    with open(loc) as file:
        lines = file.readlines()
    lines = [[float(l[0].replace(',', '.')), float(l[1].replace(',', '.')),
              float(l[2].replace(',', '.')), float(l[3].replace(',', '.'))] for l in [l.split(';') for l in lines[8:-2]]]
    return np.array(lines).T
