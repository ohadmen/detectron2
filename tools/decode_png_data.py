import base64
from io import BytesIO

import numpy as np
from PIL import Image


def decode_png_data(shape, s):
    """
    Decode array data from a string that contains PNG-compressed data
    @param Base64-encoded string containing PNG-compressed data
    @return Data stored in an array of size (3, M, N) of type uint8
    """
    fstream = BytesIO(base64.decodebytes(s.encode()))
    im = Image.open(fstream)
    data = np.moveaxis(np.array(im.getdata(), dtype=np.uint8), -1, 0)
    return data.reshape(shape)