import cupy as cp
from cupy import float64, float32, float16
import timeit

# zadanie 3 wykonane za pomocą układu GPU za pomocą toolkitu cuda 11.3 oraz bibliotece cupy

# ustawienie ilosci potrzebnej pamieci vram

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
gpu_memory = 11  # ustawić limit pamięci w GB zależnie od posiadanego gpu
memory_scaling_factor = gpu_memory/11
with cp.cuda.Device(0):
    mempool.set_limit(size=gpu_memory * 1024 ** 3)


# funkcja wielomianowa f1
# @cuda.jit('float32[:](float32[:])', device=True)
def f1(x, y):
    return cp.power(x, 5) + cp.power(y, 5) + 5


# funkcja trygonometryczna f2
def f2(x, y):
    return -cp.sin(3 * x) - cp.cos(3 * y) - 4


def check_conditions(z, zf1, zf2):
    return cp.logical_and((z <= zf1), (z >= zf2))


def t3(N, prec):
    print("-------------------------------------")
    print("URUCHOMIONO")
    a = [0, 1, 2, 3, 4, 5, 6]
    # Parametry prostopadłościanu policzone ręcznie

    # funkcje trygonometryczne o równym okresie i całej swojej przedziwdziedzinie zawartej w cylindrze więc w
    # najniższym punkcie (x,y) sin(3x) i cos(3y) przyjmą wartości równe 1 więc
    #  f2(x,y) = -1-1-4 = -6
    box_z_min = -6

    # Punkt stacjonarny funkcji f1(x,y) nie znajduje się w zadanej dziedzinie więc odpowiedz znajduje się na
    # krańcach przedziałów czyli spełnia równanie x^2 + y^y = 1, z czego możemy łatwo dowieść że punkt maximum znajduje
    # się w punkcie (1,0) lub (0,1) a funkcja zwraca w nim wartość f2(0,1 lub 1,0) = 1+0+5=6
    box_z_max = 6

    # wartości maksymalne x i y w prostopadłościanie
    box_xy_max = 1
    box_xy_min = -1

    # 3 losowe współrzędne losowane w określonym prostopadłościanie, z generowane później ze względu na ograniczenia
    # pamięci
    x = cp.random.uniform(box_xy_min, box_xy_max, size=N).astype(precyzja)
    y = cp.random.uniform(box_xy_min, box_xy_max, size=N).astype(precyzja)

    print("Ilość GB zajętych po generacji xyz", mempool.used_bytes() / 1000000000, "GB")
    print("Ilość losowanych punktów to :", N / 10 ** 6, "Mln.")
    zf1 = f1(x, y)  # ograniczenie z góry
    zf2 = f2(x, y)  # ograniczenie z dołu
    xsquared = cp.power(x, 2)
    ysquared = cp.power(y, 2)

    x = None
    y = None

    # generacja z
    z = cp.random.uniform(box_z_min, box_z_max, size=N)
    p_in_volume = check_conditions(z, zf1, zf2)  # wektor binarny, pyt : czy punkt należy do poszukiwanej objetosci ?
    print("Ilość GB zajętych po generacji f1(x,y) i f2(x,y)", mempool.used_bytes() / 1000000000, "GB")
    # niebezpośrednio zwalniamy pamięć w Pythonie
    zf1 = None
    zf2 = None

    print("Ilość GB zajętych po zwolnieniu1", mempool.used_bytes() / 1000000000, "GB")



    z = None

    print("Ilość GB zajętych po zwolnieniu xyz", mempool.used_bytes() / 1000000000, "GB")
    p_in_cylinder = cp.logical_and((xsquared + ysquared) <= 1, p_in_volume).astype(bool)
    h = box_z_max - box_z_min
    r = 1.
    box_vol = (4. * (r ** 2.) * h)
    ratio = cp.count_nonzero(p_in_cylinder) / N

    print("box vol = ", box_vol)
    print("ratio = ", ratio)
    result = box_vol * ratio
    print("Szukana objętość to :", result)


test_count = 6
N64 = int((1.2 * 10 ** 8)*memory_scaling_factor)
precyzja = float64

t64 = timeit.Timer(lambda: t3(N64, precyzja)).timeit(test_count) / test_count

mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

N32 = int((3 * 10 ** 8)*memory_scaling_factor)
precyzja = float32
t32 = timeit.Timer(lambda: t3(N32, precyzja)).timeit(test_count) / test_count

mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

N16 = int((5.2 * 10 ** 8)*memory_scaling_factor)
precyzja = float16
t16 = timeit.Timer(lambda: t3(N16, precyzja)).timeit(test_count) / test_count

print("Precyzja = 64bity")
print("Ilość losowanych pkt.", N64 / 10 ** 6, "mln")
print("Czas(śr. z ", test_count, ") = ", t64, '\n')
print("Precyzja = 32bity")
print("Ilość losowanych pkt.", N32 / 10 ** 6, "mln")
print("Czas(śr. z ", test_count, ") = ", t32, '\n')
print("Precyzja = 16bity")
print("Ilość losowanych pkt.", N16 / 10 ** 6, "mln")
print("Czas(śr. z ", test_count, " ) = ", t16, '\n')
