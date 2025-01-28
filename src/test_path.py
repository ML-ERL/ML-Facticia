import os


def list_dir(path="./"):
    elements = os.listdir(path)
    print(elements)