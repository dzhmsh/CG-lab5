import numpy as np
from PIL import Image, ImageOps
import random
import time
random.seed(time.time())


file1 = open('model_1.obj')
file2 = open('Cod.obj')
skin = Image.open('frog-legs.jpg')
voda = Image.open('oda.jpg')

print("I'm trying my hardest")
H = 2000
W = 2000

# Простое создание фона - меняем размер и конвертируем в numpy
voda_resized = voda.resize((W, H))
img_mat = np.array(voda_resized)[::-1]
'''
# Создание холста для изображения размером 2000x2000 пикселей
img_mat = np.zeros((H, W, 3), dtype=np.uint8)

# Создание фона
for i in range(H):
    for j in range(W):
        img_mat[i, j] = [0, 0, 0]
'''


def multi_chetirka(che1, che2):
    a1, b1, c1, d1 = che1
    a2, b2, c2, d2 = che2
    new_chetirka = np.array([a1*a2 - b1*b2 - c1*c2 - d1*d2, a1*b2 + b1*a2 + c1 *
                             d2 - d1*c2, a1*c2 - b1*d2 + c1*a2 + d1*b2, a1*d2 + b1*c2 - c1*b2 + d1*a2])
    return new_chetirka


def chetirka_rotation(angle, ushki, vertices, sdvig):  # ushki - axis
    q = np.array([np.cos(angle/2), 0.0, 0.0, 0.0])
    q[1:4] = np.sin(angle/2) * ushki/np.linalg.norm(ushki)
    starq = np.array(q)
    starq[1:4] = -q[1:4]
    for i in range(len(vertices)):
        vertices[i] = multi_chetirka(multi_chetirka(
            q, [0, vertices[i][0], vertices[i][1], vertices[i][2]]), starq)[1:4] + sdvig


def triing(s_list, poligons, bones, normis, textureExist):
    fix = s_list[1]
    for i in range(2, len(s_list)-1):
        poligons.append([int(fix.split('/')[0]), int(s_list[i].split(
            '/')[0]), int(s_list[i+1].split('/')[0])])
        if (textureExist):
            bones.append([int(fix.split('/')[1]), int(s_list[i].split(
                '/')[1]), int(s_list[i+1].split('/')[1])])
        if (fix.split('/')[2]):
            normis.append([int(fix.split('/')[2]), int(s_list[i].split(
                '/')[2]), int(s_list[i+1].split('/')[2])])


def read_obj_data(file, textureExist):
    vertices = []  # вершины
    poligons = []  # полигоны (треугольники)
    meat = []  # координаты текстуры
    bones = []
    normis = []
    for s in file:
        if (s == '' or s == '\n'):
            continue
        s_list = s.split()  # не работает с пустыми строками
        if (s_list[0] == 'v'):
            vertices.append(
                [float(s_list[1]), float(s_list[2]), float(s_list[3])])
        if (textureExist):
            if (s_list[0] == 'vt'):
                meat.append([float(s_list[1]), float(s_list[2])])

        if (s_list[0] == 'f'):
            if (len(s_list) > 3):
                triing(s_list, poligons, bones, normis, textureExist)
            else:
                poligons.append([int(s_list[1].split(
                    '/')[0]), int(s_list[2].split('/')[0]), int(s_list[3].split('/')[0])])
                if (textureExist):
                    bones.append([int(s_list[1].split(
                        '/')[1]), int(s_list[2].split('/')[1]), int(s_list[3].split('/')[1])])

    return vertices, poligons, meat, bones, normis


def normal_array(poligons, vertices):
    poly_n = []
    lenth = len(poligons)

    # a0 = vertices[poligons[k][0]]
    # a1 = vertices[poligons[k][1]]
    # a2 = vertices[poligons[k][2]]

    for k in range(lenth):
        a0 = vertices[poligons[k][0]-1]
        a1 = vertices[poligons[k][1]-1]
        a2 = vertices[poligons[k][2]-1]
        poly_n.append(np.cross([a1[0]-a2[0], a1[1]-a2[1], a1[2]-a2[2]],
                               [a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]]))

        poly_n[k] /= np.linalg.norm(poly_n[k])

    vertex_n = np.zeros((lenth, 3), dtype=np.float32)
    for k in range(lenth):
        v0 = poligons[k][0]-1
        v1 = poligons[k][1]-1
        v2 = poligons[k][2]-1
        vertex_n[v0] += poly_n[k]
        vertex_n[v1] += poly_n[k]
        vertex_n[v2] += poly_n[k]

    for k in range(len(vertex_n)):
        vertex_n[k] /= np.linalg.norm(vertex_n[k])

    return vertex_n

# Создание матрицы поворота по осям X, Y, Z


def rotation(rt_x, rt_y, rt_z):
    Rx = np.array([[1, 0, 0], [0, np.cos(rt_x), np.sin(rt_x)],
                   [0, -np.sin(rt_x), np.cos(rt_x)]])
    Ry = np.array([[np.cos(rt_y), 0, np.sin(rt_y)], [
        0, 1, 0], [-np.sin(rt_y), 0, np.cos(rt_y)]])
    Rz = np.array([[np.cos(rt_z), np.sin(rt_z), 0],
                   [-np.sin(rt_z), np.cos(rt_z), 0], [0, 0, 1]])

    R = Rx @ Ry @ Rz

    return R

# Применение преобразований (поворот и смещение) к вершинам


def apply_transformations(R, vertices, sdvig):

    for i in range(len(vertices)):
        vertices[i] = R@vertices[i] + sdvig

# Вычисление барицентрических координат для точки (x,y) относительно треугольника


def barycentric_coords(x, y, x0, y0, x1, y1, x2, y2):

    denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if denominator == 0:
        return [1, 1, 1]

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return [lambda0, lambda1, lambda2]

# Основная функция отрисовки 3D-модели


def render_model(vertices, poligons, vertex_n, meat, skin, bones, n, textureExist):

    for k in range(len(poligons)):
        # Получение координат вершин треугольника
        x0 = vertices[poligons[k][0]-1][0]
        y0 = vertices[poligons[k][0]-1][1]
        z0 = vertices[poligons[k][0]-1][2]
        x1 = vertices[poligons[k][1]-1][0]
        y1 = vertices[poligons[k][1]-1][1]
        z1 = vertices[poligons[k][1]-1][2]
        x2 = vertices[poligons[k][2]-1][0]
        y2 = vertices[poligons[k][2]-1][1]
        z2 = vertices[poligons[k][2]-1][2]

        I0 = np.dot(vertex_n[poligons[k][0]-1], [0, 0, 1])
        I1 = np.dot(vertex_n[poligons[k][1]-1], [0, 0, 1])
        I2 = np.dot(vertex_n[poligons[k][2]-1], [0, 0, 1])

        if (textureExist):
            x0t = meat[bones[k][0]-1][0]
            y0t = meat[bones[k][0]-1][1]
            x1t = meat[bones[k][1]-1][0]
            y1t = meat[bones[k][1]-1][1]
            x2t = meat[bones[k][2]-1][0]
            y2t = meat[bones[k][2]-1][1]

            pain_triangle(skin, [0, 200, 200], I0, I1, I2, textureExist, n, x0, y0, z0, x1, y1, z1, x2,
                          y2, z2, x0t, y0t, x1t, y1t, x2t, y2t)

        else:
            pain_triangle(skin, [0, 200, 200], I0, I1, I2, textureExist, n,
                          x0, y0, z0, x1, y1, z1, x2, y2, z2)

# Вычисление минимального и максимального значения с ограничением снизу нулем


def get_bounds(a1, a2, a3, limit):
    fmax = max(a1, a2, a3)
    fmin = min(a1, a2, a3)
    if (fmax > limit-1):
        fmax = limit-1
    if (fmin < 0):
        fmin = 0
    return [int(fmin), int(fmax)]

# Вычисление нормали к треугольнику и косинуса угла с направлением взгляда


def normal(a0, a1, a2):

    n = np.cross([a1[0]-a2[0], a1[1]-a2[1], a1[2]-a2[2]],
                 [a1[0]-a0[0], a1[1]-a0[1], a1[2]-a0[2]])

    normal_vec = n / np.linalg.norm(n)
    view_dir = [0, 0, 1]  # направление взгляда (по оси Z)
    cos_angle = np.dot(normal_vec, view_dir)
    return cos_angle

# Цвет пикселя


def get_color(x, y, cos_angle):
    contrast = cos_angle*3  # Кубическое усиление контраста

    rand = random.uniform(0, 200)

    # Красный канал - сильно зависит от угла
    r = max(255*cos_angle, min(rand, int(-255 * contrast)))

    # Зеленый канал - паттерн + угол
    g = max(rand, 10, int((y + x) ** abs(contrast) % 256))

    # Синий канал - добавляем глубину и зависимость от координат
    b = max(120*cos_angle, min(255, int(rand * abs(contrast) + (x * y) % 128)))

    # r = 255*cos_angle
    # b = 0*cos_angle
    # g = 120*cos_angle
    return [r, g, b]

# Отрисовка одного треугольника с учетом перспективы и z-буфера


def pain_triangle(skin, color, I0, I1, I2, textureExist, n, x0, y0, z0, x1, y1, z1, x2,
                  y2, z2, x0t=0, y0t=0, x1t=0, y1t=0, x2t=0,
                  y2t=0):

    # Центр изображения для смещения
    center_y = img_mat.shape[0] * 0.5
    center_x = img_mat.shape[1] * 0.5

    # Проекция вершин на экран с перспективой
    x0s = x0 * n / z0 + center_x
    x1s = x1 * n / z1 + center_x
    x2s = x2 * n / z2 + center_x
    y0s = y0 * n / z0 + center_y
    y1s = y1 * n / z1 + center_y
    y2s = y2 * n / z2 + center_y

    # Определение ограничивающего прямоугольника для треугольника
    x_bounds = get_bounds(x0s, x1s, x2s, W)
    y_bounds = get_bounds(y0s, y1s, y2s, H)

    # Вычисление нормали для отбраковки невидимых граней
    cos_angle = normal([x0, y0, z0], [x1, y1, z1], [x2, y2, z2])

    # Отбраковка задних граней
    if cos_angle > 0:
        return
    # Растеризация треугольника
    for x in range(x_bounds[0], x_bounds[1] + 1):
        for y in range(y_bounds[0], y_bounds[1] + 1):
            [l1, l2, l3] = barycentric_coords(
                x, y, x0s, y0s, x1s, y1s, x2s, y2s)
            if l1 >= 0 and l2 >= 0 and l3 >= 0:
                # Интерполяция z-координаты
                z_interpolated = l1 * z0 + l2 * z1 + l3 * z2
                I_interpolated = (l1 * I0 + l2 * I1 + l3 * I2)
                if I_interpolated > 0:
                    continue
                # Проверка z-буфера
                if z_interpolated < zbuf[y, x]:
                    if (textureExist):
                        cell = [1024*(l1*x0t + l2 * x1t + l3 * x2t),
                                1024 * (l1*y0t + l2*y1t + l3*y2t)]
                        img_mat[y, x] = np.array(skin.getpixel(
                            (cell[0], 1023 - cell[1])))*(-I_interpolated)
                    else:
                        img_mat[y, x] = get_color(x, y, I_interpolated)

                    zbuf[y, x] = z_interpolated


# Основная программа
def main(file, skin, sdvig, illusion, zbuf, el_rotation, angle, textureExist):

    vertices, poligons, meat, bones, normis = read_obj_data(file, textureExist)

    # Применение поворота и смещения

    # Эйлер
    if illusion == 1:
        R = rotation(el_rotation)
        apply_transformations(R, vertices, sdvig)

    # Поворот через квартенионы
    else:
        chetirka_rotation(angle, np.array(el_rotation), vertices, sdvig)

    vertex_n = normal_array(poligons, vertices)

    # Отрисовка модели
    render_model(vertices, poligons, vertex_n, meat,
                 skin, bones, 10000, textureExist)  # 10 000 для кролина, для остальных 2 000


zbuf = np.full((H, W), np.inf)  # Инициализация z-буфера
illusion = 2  # выбор это иллюзия

# main(file1, skin, [0, -0.06, 10], 2, zbuf, [1.0, 1.0, 2.0], 10, 1) #frog
print("HAlf done")
main(file1, voda, [0.0, -0.05, 0.7], 2, zbuf, [1, 1.0, 1.0], 0, 0)
# main(file2, voda, [-1, -1, 10], 2, zbuf, [1.0, 1.0, 1.0], 0, 0)

# Сохранение изображения
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)  # Отражаем по вертикали
img.save('img.png')
print("I drawt this beautiful rabbit")

file1.close()
