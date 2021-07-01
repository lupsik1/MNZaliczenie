import cupy as cp
from cupy import float32, float16
import timeit

# zadanie 3 wykonane za pomocą układu GPU za pomocą bibliotekii cupy

# ustawienie ilosci potrzebnej pamier vram

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
with cp.cuda.Device(0):
    mempool.set_limit(size=11 * 1024 ** 3)  # 11 GiB czyli dałem co miałem


# funkcja wielomianowa f1
# @cuda.jit('float32[:](float32[:])', device=True)
def f1(x, y):
    return cp.power(x, 5) + cp.power(y, 5) + 5


# funkcja trygonometryczna f2

def f2(x, y):
    return -cp.sin(3 * x) - cp.cos(3 * y) - 4


def check_conditions(z, zf1, zf2):
    # definiujemy wektory Z w pamięci współdzielonej
    t1 = (z <= zf1)
    t2 = (z >= zf2)
    return cp.logical_and(t1, t2)


def t3(N, type):
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

    # generacja z
    z = cp.random.uniform(box_z_min, box_z_max, size=N)
    p_in_volume = check_conditions(z, zf1, zf2)  # wektor binarny, pyt : czy punkt należy do poszukiwanej objetosci ?
    print("Ilość GB zajętych po generacji f1(x,y) i f2(x,y)", mempool.used_bytes() / 1000000000, "GB")
    # niebezpośrednio zwalniamy pamięć w Pythonie
    zf1 = None
    zf2 = None
    xy = None

    print("Ilość GB zajętych po zwolnieniu1", mempool.used_bytes() / 1000000000, "GB")
    xsquared = cp.power(x, 2)
    ysquared = cp.power(y, 2)

    x = None
    y = None
    z = None

    print("Ilość GB zajętych po zwolnieniu xyz", mempool.used_bytes() / 1000000000, "GB")
    p_in_cylinder = cp.logical_and((xsquared + ysquared) <= 1, p_in_volume).astype(bool)
    h = box_z_max - box_z_min
    r = 1
    box_vol = (4 * (r ** 2) * h)
    ratio = cp.count_nonzero(p_in_cylinder) / N

    print("box vol = ", box_vol)
    print("ratio = ", ratio)
    result = box_vol * ratio
    print("Szukana objętość to :", result)


N = int(3 * 10 ** 8)
precyzja = float32
t = timeit.Timer(lambda: t3(N, precyzja))
print("WYNIK :\nCzas wykonania programu(średnia z 10 wykonań) =", t.timeit(10) / 10, "s")
