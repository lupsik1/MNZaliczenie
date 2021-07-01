from PIL import Image
import cupy as cp
import numpy as np
from cupy import float64, uint8

def normalize_and_scale(a):

def show_image_from_array(a):
    img = Image.fromarray(cp.asnumpy(a), 'L')
    img.show()

def t4():
    n = 401
    images = [Image.open('face_SVD/foto (' + str(x) + ').jpg') for x in range(1, n)]  # wczytanie zdjęć

    data = cp.array([cp.asarray(im).reshape((193 * 162)) for im in images]).astype(float64).T  # konwersja na gpu array
    print(data[0])
    u, s, v = cp.linalg.svd(data,full_matrices=False)  # u = S , s = V, v = D
    s = cp.diag(s)


    choice = 11

    u_reduced = abs(u[:, choice].reshape(193, 162))
    print(u_reduced.shape)
    u_reduced = ((u_reduced/cp.amax(u_reduced))*255).astype(uint8)
    showImageFromArray(u_reduced)

t4()
