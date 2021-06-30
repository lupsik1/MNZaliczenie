import cupy as cp
import timeit

# zadanie 3 wykonane za pomocą układu GPU za pomocą bibliotekii cupy

# ustawienie ilosci potrzebnej pamier vram

mempool = cp.get_default_memory_pool()

with cp.cuda.Device(0):
    mempool.set_limit(size=10 * 1024 ** 3)  # 10 GiB

# liczba punktów do wylosowania
N = int(1.2 * (10 ** 8))

# miejsce podziału połączonej tablicy xy
y_start = N
x_end = N

# ilość wątkow przypadających na 1 blok oraz posiadających rownoczesny dostęp do pamieci wspolnej
TPB = 64


# funkcja wielomianowa f1
# @cuda.jit('float32[:](float32[:])', device=True)
def f1(xy):
    return cp.add(cp.add(cp.power(xy[0:x_end], 5), cp.power(xy[y_start:2 * N], 5)), 5)


# funkcja trygonometryczna f2

def f2(xy):
    return -cp.sin(3 * xy[0:x_end]) - cp.cos(3 * xy[y_start:2 * N]) - 4


def check_conditions(z, zf1, zf2):
    # definiujemy wektory Z w pamięci współdzielonej
    t1 = (z <= zf1)
    t2 = (z >= zf2)
    return cp.logical_and(t1, t2)


def t3():
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

    # 3 losowe współrzędne losowane w określonym prostopadłościanie
    x = cp.random.uniform(box_xy_min, box_xy_max, size=N)
    y = cp.random.uniform(box_xy_min, box_xy_max, size=N)
    xy = cp.concatenate((x, y)).astype(cp.float32)
    z = cp.random.uniform(box_z_min, box_z_max, size=N)

    print("Ilość losowanych punktów to :", N / 10 ** 6, "Mln.")
    zf1 = f1(xy)  # ograniczenie z góry
    zf2 = f2(xy)  # ograniczenie z dołu

    p_in_volume = check_conditions(z, zf1, zf2)  # wektor binarny, pyt : czy punkt należy do poszukiwanej objetosci ?
    p_in_cylinder = cp.logical_and((cp.power(x, 2) + cp.power(y, 2)) <= 1, p_in_volume)
    h = box_z_max - box_z_min
    r = 1
    box_vol = (4 * (r ** 2) * h)
    ratio = cp.count_nonzero(p_in_cylinder) / N

    print("box vol = ", box_vol)
    print("ratio = ", ratio)
    result = box_vol * ratio
    print("Szukana objętość to :", result)


print("Czas wykonania programu(średnia z 50 wykonań) =", timeit.timeit(stmt=t3, number=50) / 50, "s")
