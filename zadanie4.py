from PIL import Image
import cupy as cp
import numpy as np
from cupy import float32, uint32, uint8
import itertools as it
from PIL import ImageFont
from PIL import ImageDraw
from numba import guvectorize
from pytictoc import TicToc
import cProfile

# params
N = 401
gpu_memory = 11  # ustawić limit pamięci w GB zależnie od posiadanego gpu

# setup
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
memory_scaling_factor = gpu_memory / 11

with cp.cuda.Device(0):
    mempool.set_limit(size=gpu_memory * 1024 ** 3)


# array preparation to show as a BW image
def reshape_normalize_and_scale(a, k):
    return ((a / cp.amax(a)) * k).reshape(193, 162).astype(uint8)


# used to get our matrix of n chosen attributes
def get_svd_attributes(data, u, attr_cnt=5):
    vectors = u[:, :attr_cnt]
    print(vectors.shape)
    return data.T @ vectors  # face_data` * vectors


# vectorised dot product of corresponding rows in 2 matrices
@guvectorize(['(float32[:], float32[:], float32[:])'], '(n),(n)->()')
def product_of_rows(u, v, res):
    res[0] = 0
    for i in range(len(u)):
        res[0] += u[i] * v[i]


# calculates cosine similarity between all 2-combinations of photographs and returns n best answers
def get_n_best_pairs(n, attrs, result):
    t = TicToc()

    t.tic()
    iterator_par = np.array(list(it.combinations(range(400), 2)), dtype=uint32)
    t.toc("Generated pair it. vector in")
    t.tic()
    xattr = np.array(attrs[iterator_par[:, 0], :], dtype=float32)
    print(xattr.shape)
    yattr = np.array(attrs[iterator_par[:, 1], :], dtype=float32)
    t.toc("Generated x y vectors in ")
    xyprods = np.empty(xattr.shape[0])

    t.tic()
    product_of_rows(xattr, yattr, xyprods)

    xyprods = cp.array(xyprods)
    t.toc("Calculated products of all pairs of rows in")
    xattr = cp.array(xattr)
    yattr = cp.array(yattr)
    normsx = cp.linalg.norm(cp.array(xattr), axis=1).astype(float32)
    normsy = cp.linalg.norm(cp.array(yattr), axis=1).astype(float32)

    cosin_similarit = cp.ElementwiseKernel(
        'T xydot, X normx, X normy',
        'X z',
        'z = abs(xydot/(normx * normy))',
        'cosine_similarity')

    cosin_similarit(xyprods, normsx, normsy, result)

    winners = cp.asnumpy(sorted_indices(result, n)[0])

    return np.array(iterator_par[winners]), result[winners]


# returns indices of n biggest numbers in an array/vector
def sorted_indices(ary, n, f=-1):
    if f == 1:
        ary = cp.abs(ary-1)
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]

    return np.unravel_index(indices, ary.shape)


# task 4 main function
def t4():
    t = TicToc()
    t2 = TicToc()
    t.tic()
    images = [Image.open('face_SVD/foto (' + str(x) + ').jpg') for x in range(1, N)]  # wczytanie zdjęć

    t2.tic()
    data = cp.array([np.asarray(im, dtype=float32).reshape((193 * 162)) for im in images]).T  # konwersja na gpu array
    t2.toc("Loaded data in:")

    t2.tic()
    u, s, v = cp.linalg.svd(data, full_matrices=False)
    t2.toc("SVD in :")

    how_many = 100  # ile par zwrócić
    result = cp.zeros(79800, dtype=float32)

    t2.tic()
    result, sims = get_n_best_pairs(how_many, cp.asnumpy(get_svd_attributes(data, u, 10)),
                                    result)  # liczy n najpodobniejszych par zdjęć
    t2.toc("Calculated best pairs in :")
    tim = t.toc('Total computation time:')
    # na podstawie 10 atrybutów wyliczonych w get_svd_attributes
    final = cp.zeros((1, 162))

    imgs_merged = Image.new('L', (170 * 2, how_many * 200), 0)

    for i in range(how_many):
        img1 = reshape_normalize_and_scale(data[:, result[i][1]], 255)
        img2 = reshape_normalize_and_scale(data[:, result[i][0]], 255)

        merge = Image.fromarray(cp.asnumpy(cp.concatenate((img1, img2), axis=1)), 'L')

        draw = ImageDraw.Draw(merge)
        font = ImageFont.truetype("sans-serif.ttf", 19)
        draw.text((80, 0), str(i + 1) + ". #" + str(result[i][1]) + " ,#" + str(result[i][0]) + "sim=" + str(sims[i]),
                  0, font=font)
        imgs_merged.paste(merge, (0, i * 200))
    return imgs_merged, tim


img, time = t4()
img.save("faces_ranked.png", 'png')
img.show()

# Profile.run('t4()', sort=1)
