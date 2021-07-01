from PIL import Image
import cupy as cp
import numpy as np
from cupy import float64, float32, int32, uint8
from numba import cuda
from sklearn.metrics.pairwise import cosine_similarity


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
gpu_memory = 11  # ustawić limit pamięci w GB zależnie od posiadanego gpu
memory_scaling_factor = gpu_memory / 11
with cp.cuda.Device(0):
    mempool.set_limit(size=gpu_memory * 1024 ** 3)


def reshape_normalize_and_scale(a, k):
    return ((a / cp.amax(a)) * k).reshape(193, 162).astype(uint8)


def show_image_from_array(a):
    img = Image.fromarray(cp.asnumpy(a), 'L')
    img.show()


def get_svd_attributes(data, u, attr_cnt=10):
    vectors = u[:, :attr_cnt]
    return cp.matmul(vectors, cp.matmul(data.T, vectors).T)  # vectors * (face_data` * vectors)`



def get_n_best_pairs(n, attrs, result):
    face_count = attrs.shape[1]
    result = cosine_similarity(attrs)

def t4():
    n = 401
    images = [Image.open('face_SVD/foto (' + str(x) + ').jpg') for x in range(1, n)]  # wczytanie zdjęć

    data = cp.array([cp.asarray(im).reshape((193 * 162)) for im in images]).astype(float64).T  # konwersja na gpu array

    u, s, v = cp.linalg.svd(data, full_matrices=False)  # u = S , s = V, v = D
    s = cp.diag(s)

    result = cp.zeros(shape=(n-1, n-1))

    result = get_n_best_pairs(5,get_svd_attributes(data, u),result)
    print(result)


t4()
