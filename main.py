import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QFileDialog
from matplotlib import cm
from pylab import imread
from skimage.transform import hough_line, hough_line_peaks

from my_des import Ui_MainWindow
from sec_win import Ui_MainWindow as Sec_Ui

"""
file_img - путь к открываемому изображению
image - изображение открытое
blur - размытое изображение
calibr:
[0] - калибровочный угол. вписывается при нажатии кнопки Калибровка. нужен для расчета угла поворота вопа
[1] - х координата круга I четверти
[2] - y координата круга I четверти 
[3] - х координата круга III четверти
[4] - y координата круга III четверти
[5] - x координата ЦТ
[6] - y координата ЦТ
[7] - расстояние центров 1-3 четверти шаблона(!) для расчета коэффициента увеличения
[8] - 1px в микронах
"""
"""
nastr
[0] - 120 - param1
[1] - 50 param2
[2] - 50 minRadius
[3] - 25 Порог. если меньше - 0, если больше 255
[4] - Шаг(точность) замера кривизны. Чем меньше, тем точнее, тем больше пикселей находит
[5] - Нижняя граница фильтра(кривизна)
"""

# circles[0-4] - 0-цт, 1-i, 2-ii, 3-iii, 4-iv
# yark_cal - калибровка яркости
file_img, calibr, yark_cal, image, blur, circles, thresh = None, [], None, None, None, [], None

# Настройки программы
nastr = []

# Потом удалить
red = []
i_red = 0


# Функция нахождения кругов
def find_circles():
    # нахождение кругов
    # 1.5 120 50 50 600
    d = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.5, 700, param1=nastr[0], param2=nastr[1], minRadius=nastr[2],
                         maxRadius=600)

    if d is None:
        return False

    if len(d[0]) < 5:
        return False

    d_c = d[0]

    if len(d_c) >= 5:
        circles = np.zeros((5, 3), int)
        for i in range(5):
            min = 1000
            a = -1
            for j in range(5):
                if d_c[j][2] < min:
                    min = d_c[j][2]
                    a = j
            circles[4 - i][0], d_c[a][0] = d_c[a][0], 1001
            circles[4 - i][1], d_c[a][1] = d_c[a][1], 1001
            circles[4 - i][2], d_c[a][2] = d_c[a][2], 1001

        # порядок как в четвертях
        circles[[0, 4]] = circles[[4, 0]]
        circles[[1, 3]] = circles[[3, 1]]
        circles[[2, 4]] = circles[[4, 2]]

        p1 = circles[1, 0:2]
        p2 = circles[3, 0:2]
        p3 = circles[0, 0:2]
        d = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
        if d > 350:
            circles[[4, 3]] = circles[[3, 4]]

        return circles

    else:
        return False


# Угол поворота ВОПа
def angles_five():  # угол поворота вопа

    # a - 3; b - 1
    ax, ay, bx, by = circles[3][0], circles[3][1], circles[1][0], circles[1][1]
    ab = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    al = math.degrees(math.acos(((bx - ax) * 10 + (by - ay) * (-10)) / (ab * math.sqrt(200))))

    # вычисление угла
    angle = math.atan2(by - ay, bx - ax)  # - math.atan2(10, -10)
    # print("angle 1 -", math.degrees(angle))

    # Если угол отклонения больше 145
    # if abs(angle - math.atan2(10, -10)) > 1.75:
    #     return 145

    angle = -math.degrees(angle - math.atan2(10, -10)) - calibr[0]
    if angle > 180:
        angle -= 360
    # angle = -(math.degrees(angle) + 45) - calibr[0]
    # print(angle)
    # calibr[0] *= -1
    # print("angle - {}".format(-math.degrees(angle) - calibr[0]))

    # if angle > math.pi:
    #     angle -= 180
    #     print("if angle -", angle)
    #
    # elif angle <= -math.pi:
    #     angle += 180
    #     print("elif angle -", angle)

    # Расчет знака угла. по траектории движения большого круга
    # I(х,у) и III(х,у) четверти
    # if angle < -180:
    #     angle += 180
    # elif angle > 180:
    #     angle -= 180
    x1, y1, x2, y2 = calibr[1], calibr[2], calibr[3], calibr[4]  # Значения координат круга I четверти

    if bx > x1 or by > y2 or (by == y2 and bx == x1):
        al *= -1

    # if bx < x2 or by < y1 or (bx == x2 and by == y1):
    #     al *= -1

    # print(al + calibr[0])

    # return al - calibr[0]
    return angle


# Угол. Потом удалить
def angles_six():  # угол поворота вопа

    # a - 3; b - 1
    ax, ay, bx, by = circles[3][0], circles[3][1], circles[1][0], circles[1][1]

    # А - вектор между кругами В - вектор с 45 градусами
    vec_a = [bx - ax, by - ay]
    vec_b = [10, -10]

    # A*B
    ab = vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]

    # A*B по модулю в onlinemschool.com
    mod_ab = math.sqrt(vec_a[0] ** 2 + vec_a[1] ** 2) * math.sqrt(vec_b[0] ** 2 + vec_b[1] ** 2)

    ugol = ab / mod_ab

    al = math.degrees(math.acos(((bx - ax) * 10 + (by - ay) * (-10)) / (ab * math.sqrt(200))))

    # Расчет знака угла. по траектории движения большого круга
    # I(х,у) и III(х,у) четверти
    x1, y1, x2, y2 = calibr[1], calibr[2], calibr[3], calibr[4]  # Значения координат большого круга в I четверти

    if bx > x1 or by > y2 or (by == y2 and bx == x1):
        al *= -1

    # print(al - calibr[0])
    return al - calibr[0]


# не используется. Потом удалить
def angles_two():
    image = imread(file_img[0])

    image = np.mean(image, axis=2)

    cv2.imwrite('out_mean_np.jpg', image)

    image = (image < 128) * 255

    # Classic straight-line Hough transform

    # Set a precision of 0.5 degree.

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 3600, endpoint=False)  # linspace(от , до, кол-во элементов между

    h, theta, d = hough_line(image, theta=tested_angles)

    # Generating figure 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)

    ax[0].set_title('Input image')

    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()  # 0.5

    d_step = 0.5 * np.diff(d).mean()  # 0.5

    bounds = [np.rad2deg(theta[0] - angle_step),

              np.rad2deg(theta[-1] + angle_step),

              d[-1] + d_step, d[0] - d_step]

    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)

    ax[1].set_title('Hough transform')

    ax[1].set_xlabel('Angles (degrees)')

    ax[1].set_ylabel('Distance (pixels)')

    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)

    ax[2].set_ylim((image.shape[0], 0))

    ax[2].set_axis_off()

    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])

        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()

    plt.show()

    angle = []

    dist = []

    for _, a, d in zip(*hough_line_peaks(h, theta, d)):
        angle.append(a)

        dist.append(d)

    angle = [a * 180 / np.pi for a in angle]
    angle_reel = np.max(angle) - np.min(angle)

    return angle_reel


# Проверить, иногда слишком высокие значения
# Кривизна четвертей. Не используется, можно удалить
def curve():  # p1[3, 4] p2[1,2] m[5, 6]

    # расстояние от ЦТ до оси I-III
    num0 = abs((circles[3][0] - circles[1][0]) * (circles[1][1] - circles[0][1]) - (circles[1][0] - circles[0][0]) *
               (circles[3][1] - circles[1][1]))
    den = math.sqrt((circles[3][0] - circles[1][0]) ** 2 + (circles[3][1] - circles[1][1]) ** 2)

    # расстояние от ЦТ до оси II-IV
    num1 = abs((circles[4][0] - circles[2][0]) * (circles[2][1] - circles[0][1]) - (circles[2][0] - circles[0][0]) *
               (circles[4][1] - circles[2][1]))
    den1 = math.sqrt((circles[4][0] - circles[2][0]) ** 2 + (circles[4][1] - circles[2][1]) ** 2)

    # суммароное расстояние
    # 1пикс = 7 мк (Потом пересчитать)
    return "Уровень кривизны(мк): \ni-iii: {0}\ti-iv: {1}\n" \
           "суммарно: {2}\n\n".format(round(num0 / den * calibr[8], 2), round(num1 / den1 * calibr[8], 2),
                                      round((num0 / den + num1 / den1) * calibr[8], 2))


# Смещение Центральной Точки
def circle():
    # х у калибровочной цт
    x, y = calibr[5], calibr[6]
    x2, y2 = circles[0][0], circles[0][1]
    dist = math.sqrt((x2 - round(x)) ** 2 + (y2 - round(y)) ** 2)

    return round(abs(dist * calibr[8]))


# Перевод десятичных градусов в градусы, минуты
def des_v_min(des):
    # print(des)
    aa = str((abs(des) % 1) * 0.6)

    # сокращение 0.0
    if aa[2] == "0" and len(aa) > 3:
        min = aa[3]

    elif aa[2:4] == "00" or aa == "0.0":
        min = "0"
    else:
        min = aa[2:4]

    if -1 < des < 0:
        return "-{}\N{DEGREE SIGN} {}\'".format(int(des), min)

    return "{}\N{DEGREE SIGN} {}\'".format(int(des), min)


# потом удалить
def yark_to_txt(s):
    # print("dadad")
    with open("yark.txt", 'r+') as f:
        # очистка и запись новых настроек из окна в nastr
        a = f.readlines()
        xs = '\n'.join(str(x) for x in s)

        # удаление всего текста в yark.txt
        f.seek(0)
        f.truncate()

        f.write(xs)


# Главная функция для калибровки
def yark(d):
    # d - найденный круг для яркости
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y_o = yark_func(d, image_gray)

    if y_o == 254:
        return 254

    if y_o == 255:
        return 255
    return round(y_o, 2)


# подфункции калибровки
def yark_func(d, image_gray):
    # обрезка фотографии
    img = image_gray[int(d[1]) - int(d[2]):int(d[1]) + int(d[2]), int(d[0]) - int(d[2]):int(d[0]) + int(d[2])]
    # cv2.imwrite("img_yark.jpg", img)

    # Проверка, входит ли круг в изображение полностью
    if d[0] + d[2] > image_gray.shape[1] or d[1] + d[2] > image_gray.shape[0]:
        return 254

    # создание маски
    a = img.shape[0]
    mask = np.zeros((a, a, 1), np.uint8)
    mask = cv2.circle(mask, (int(a / 2), int(a / 2)), int((a / 2) * 0.95), (255, 255, 255), -1)
    # cv2.imwrite("mask.jpg", mask)

    s = cv2.mean(img, mask)[0]

    # Вывод в текстовый файл значений яркости

    # потом удалить
    ss = []
    summ = [0, 0, 0]
    for i in range(len(img)):
        for j in range(len(img)):
            if mask[i, j] != 0:

                # поиск битых пикселей и пересвета.
                if img[i, j] > 230:
                    summ[2] += 1

                # сумматор значений
                summ[0] += img[i, j]

                # потом удалить
                ss.append(img[i, j])
                if img[i, j] < 50:
                    red.append([i, j])

                # сумматор кол-ва значений
                summ[1] += 1

    # потом удалить
    red_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(red)
    for i in red:
        cv2.circle(red_img, (i[0], i[1]), 1, (0, 0, 255), 1)
    global i_red
    s_red = "red_img_{}.jpg".format(str(i_red))
    i_red += 1
    cv2.imwrite(s_red, red_img)
    red.clear()
    # print("red")

    # потом удалить
    yark_to_txt(ss)

    # Конец вывода текстового файла

    # проверка, пересвечено ли изображение. Находит все точки внутри круга > 230. Если они соствляют > 1% от всего числа
    if s > 230:
        return 255
    # Удалить до этой строчки

    return s


# Главная функция для измерения яркости, остальные вызываются отсюда. перевод в проценты
def yark_out(circ):
    y = yark(circ)

    if y == 254:
        return 254

    elif y == 255:
        return 255

    else:
        # перевод яркости в процентное соотношение (яркость воп/начальная яркость * 100)
        a = round(y / yark_cal, 2)
        return a


# калибровка по 1 и 3 четверти
def calibration(circ_cal):
    ax, ay, bx, by = circ_cal[3][0], circ_cal[3][1], circ_cal[1][0], circ_cal[1][1]

    # # нахождение угла между векторами 45г. и ab(I, III). Находим дельту - это и будет отклонением от нормы (Калибровка)
    # ab = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    #
    # # калибровочный угол поворота
    # al = math.degrees(math.acos(((bx - ax) * 10 + (by - ay) * (-10)) / (ab * math.sqrt(200))))
    #
    # if circ_cal[1][1] > circ_cal[2][1]:
    #     al *= -1

    al = math.degrees(math.atan2(by - ay, bx - ax) - math.atan2(-10, 10))
    al *= -1

    # calibr[0] *= -1
    # print("angle - {}".format(math.degrees(angle) - calibr[0]))

    # потом удалить
    # калибровка яркости.Определение начальной яркости для сопоставления с последующими проверками и перевода в проценты
    # a = yark(circ_cal)

    # расстояние центров 1-3
    d = d = np.sqrt(sum(pow(a - b, 2) for a, b in zip(circ_cal[1, 0:2], circ_cal[3, 0:2])))

    mikr = px_in_mk(circ_cal)

    return [al, bx, by, ax, ay, circ_cal[0][0], circ_cal[0][1], d, mikr]


# Расчет микрона на пиксель(calibr[13])
def px_in_mk(circ_cal):
    d = np.sqrt(sum(pow(a - b, 2) for a, b in zip(circ_cal[1, 0:2], circ_cal[3, 0:2])))
    return round(12445 / d, 2)


# функция измерения коэффициента увеличения
# отключен, дает неверные данные из-за неточного фокусирования
def uvelich():
    d = np.sqrt(sum(pow(a - b, 2) for a, b in zip(circles[1, 0:2], circles[3, 0:2])))
    return round(d / calibr[12], 4)


# Функция загрузки настроек перед запуском программы
def nastr_nachalo():
    try:
        global nastr
        with open("settings.txt", 'r+') as f:
            # очистка и запись новых настроек из окна в nastr
            a = f.readlines()
            nastr.clear()
            for i in a:
                nastr.append(int(i.strip()))
            # Если не все настройки добавлены
            if len(nastr) != 6:
                raise Exception
    except:
        # Проверяет корректность файла с сохраненными настройками settings.txt
        # если есть ошибка в файле, запускает с настройками ниже
        nastr = [120, 50, 50, 25, 4, 180]
        # print(len(nastr))
        return 255

# Измерение кривизны изображения на ВОП
def curve_sec(gr, ngr):
    # Точность. Чем ниже, тем точнее. Оптимально - 3-5
    acc = 3
    img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = img.shape[1]
    height = img.shape[0]
    x, y = [], []

    for i in range(0, width, acc):
        for j in range(0, height, 5):
            if img[j][i] > ngr:
                x.append(i)
                y.append(j)

    d = 15
    theta = np.polyfit(x, y, deg=d)

    model = np.poly1d(theta)

    # Координаты крайних точек линии
    x1, x2 = x[5], x[len(x) - 6]
    y1, y2 = model(x1), model(x2)

    # ось
    plt.figure("baaaasp")
    plt.gca().set_aspect('equal')
    plt.plot(x, y, 'ro')
    plt.plot(x, model(x))
    plt.plot([x1, x2], [y1, y2], 'b--', marker='.')
    # plt.axis([x1 - x1*0.1, x2 + x2*0.1, y1 - y1 * 0.1, y2 + y2 * 0.1])
    plt.axis([0, width, 0, height])

    # Нахождение максимального отклонения от оси AB
    max_dist = 0
    x_max = 0
    for i in range(x1 + 1, x2, acc):
        y0 = model(i)

        dist = abs((x2 - x1) * (y1 - y0) - (x1 - i) * (y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1))
        if dist > max_dist:
            max_dist = dist
            x_max = i

    # перпендикуляр
    x3, y3 = x_max, model(x_max)
    k = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / ((y2 - y1) ** 2 + (x2 - x1) ** 2)
    x4 = x3 - k * (y2 - y1)
    y4 = y3 + k * (x2 - x1)
    plt.plot([x3, x4], [y3, y4], 'g', marker='.')
    plt.savefig('dop/func.pdf')

    # Вывод графика
    if gr:
        plt.show()

    return round(max_dist, 2), x_max


# меню
def menu(chbx):
    # Потом удалить
    # chbx = [0, 1]
    ty = "Результаты анализа ВОП:\n"

    # Угол поворота, смещение цт
    if chbx[0]:
        global circles
        circles = find_circles()

        # если нашел не все круги
        if type(circles) is bool:
            return "Найдены не все круги!"

        af = angles_five()
        if af == 145:
            ty += "Угол отклонения от 180\N{DEGREE SIGN} больше 145\N{DEGREE SIGN}. ВОП может быть прямым.\n\n"
        else:
            ty += "Угол отклонения от 180\N{DEGREE SIGN}: \n{0} град.\n\n".format(des_v_min(af))
        # ty += curve()
        # ty += "Смещение центральной точки: \n{0} мк\n\n".format(circle())

    # Яркость
    if chbx[1]:
        d = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.5, 700, param1=nastr[0], param2=nastr[1],
                             minRadius=nastr[2], maxRadius=5000)[0][0]
        if d is None:
            return "Круг не найден программой!"

        y = yark_out(d)
        if y == 254:
            return "Круг выходит за границы изображения. Откалибруйте установку и повторите."

        elif y == 255:
            return "Изображение пересвечено! Снизьте яркость и попробуйте снова."

        else:
            return "Коэффициент пропускания - {}".format(y)

    # Кривизна
    if chbx[2]:
        # Графика, нижняя граница
        cs = curve_sec(chbx[3], nastr[5])
        return "Максимальное отклонение - {} мк, при х = {}".format(round(cs[0] * 7, 1), cs[1])

    return tyg

def yark_cv2():
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(image_gray, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    # print(avg_color)


class Sec_win(QtWidgets.QMainWindow, Sec_Ui):  # +++
    def __init__(self, parent=None):
        super(Sec_win, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Настройка значений")
        self.setWindowIcon(QIcon("dop/logo.ico"))


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.s_w = None
        self.setWindowTitle("Поворот/кривизна/яркость")
        self.setWindowIcon(QIcon("dop/logo.ico"))
        # self.ui.label.setScaledContents(True)
        # подключение клик-сигнал к слоту btnClicked
        self.ui.pushButton.clicked.connect(self.check)
        self.ui.pushButton_2.clicked.connect(self.open)
        self.ui.pushButton_3.clicked.connect(self.calibr)
        self.ui.pushButton_4.clicked.connect(self.next_img)
        self.ui.action_2.triggered.connect(self.nastr_krugi)
        self.ui.action.triggered.connect(self.sec_win)
        self.ui.label.setPixmap(QPixmap("dop/rsz_1bg.jpg"))
        if nastr_nachalo() == 255:
            self.ui.textEdit.setText("При загрузке настроек произошла ошибка, приняты настройки по умолчанию")

    # Проверка
    def check(self):
        self.ui.textEdit.setText("")
        a = self.ui.radioButton.isChecked()
        b = self.ui.radioButton_2.isChecked()
        c = self.ui.radioButton_3.isChecked()
        graphic = self.ui.checkBox.isChecked()
        # Потом удалить
        # d = self.ui.checkBox_4.isChecked()

        # проверка на наличие калибровки
        if len(calibr) > 0 and a:
            ty = str(menu([a, 0, 0, graphic]))
            self.ui.textEdit.setText(ty)

        elif yark_cal is not None and b:
            self.ui.textEdit.setText(str(menu([0, b, 0, graphic])))

        elif c:
            if image is None:
                return self.ui.textEdit.setText("Пожалуйста, для начала работы выберите изображение")
            self.ui.textEdit.setText(str(menu([0, 0, c, graphic])))

        else:
            self.ui.textEdit.setText("Пожалуйста, для начала работы - откалибруйте!")

    # Загрузить изображение
    def open(self):
        global file_img
        file_img = QFileDialog.getOpenFileName()
        if len(file_img[0]) == 0:
            return self.ui.textEdit.setText("Выберите изображение!")
        Image.open(file_img[0]).resize((854, 641)).save("dop/out_res.png")
        pixmap = QPixmap("dop/out_res.png")
        fl = file_img[0].split("/")
        self.ui.textEdit_2.setText("{}/{}".format(fl[-2], fl[-1]))
        self.ui.label.setPixmap(pixmap)

        global image, blur, thresh
        image = cv2.imread(file_img[0])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_gray, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)

        # blur 70 255 thresh_binary
        ret, thresh = cv2.threshold(blur, nastr[3], 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)

    # Калибровка
    def calibr(self):
        # 7-11
        global calibr
        global yark_cal
        yark_cal = None
        calibr.clear()

        if file_img is None:
            return self.ui.textEdit.setText("Для начала работы выберите калибровочное изображение!")

        # Если радиобаттон прорверка
        if self.ui.radioButton.isChecked():
            circ_cal = find_circles()
            if type(circ_cal) is bool:
                self.ui.textEdit.setText("Найдены не все круги!")
            else:
                calibr.extend(calibration(circ_cal))
                self.ui.textEdit.setText("Калибровка прошла успешно. Угол калибровки: " + des_v_min(calibr[0]))

        # если радиобаттон яркость
        elif self.ui.radioButton_2.isChecked():

            d = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.5, 700, param1=nastr[0], param2=nastr[1],
                                 minRadius=nastr[2], maxRadius=5000)[0][0]

            # d = d[0]

            if d is None:
                return self.ui.textEdit.setText("Не найден круг!")

            yark_o = yark(d)
            if yark_o == 254:
                self.ui.textEdit.setText("Круг выходит за границы изображения. Откалибруйте установку и повторите.")
            elif yark_o == 255:
                self.ui.textEdit.setText("Изображение пересвечено! Снизьте яркость и попробуйте снова.")

            else:
                self.ui.textEdit.setText("Калибровочное значение яркости - {}".format(yark_o))
                yark_cal = yark_o

        else:
            self.ui.textEdit.setText("Выберите галочкой проверку или яркость")

    # Настройки -> Найденные круги
    def nastr_krugi(self):
        # print(nastr)
        circ = None
        cv2.imwrite("thresh5.jpeg", thresh)
        if file_img is not None:
            if len(file_img[0]) > 2 and self.ui.radioButton.isChecked():
                circ = find_circles()

            elif len(file_img[0]) > 2 and self.ui.radioButton_2.isChecked():
                circ = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.5, 700, param1=nastr[0], param2=nastr[1],
                                        minRadius=nastr[2], maxRadius=2000)
                if circ is not None:
                    circ = circ[0]
                else:
                    self.ui.textEdit.setText("Круг не найден.")

            if circ is not None:
                # Convert the circle parameters a, b and r to integers.
                circles = np.uint16(np.around(circ))
                s = 1
                for pt in circ:
                    a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
                    # print(a, b, r)
                    # Draw the circumference of the circle.
                    cv2.circle(image, (a, b), r, (0, 255, 0), 2)

                    # Draw a small circle (of radius 1) to show the center.
                    cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

                    cv2.putText(image, str(s), (a + 15, b + 15), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255))
                    s += 1
                scale_percent = 25  # percent of original size

                width = int(image.shape[1] * scale_percent / 100)

                height = int(image.shape[0] * scale_percent / 100)

                dim = (width, height)

                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                cv2.imwrite("img.png", image)

                cv2.imshow("Resized image", resized)

                cv2.waitKey(0)

            else:
                self.ui.textEdit.setText("Круг не найден.")

        else:
            self.ui.textEdit.setText("Для начала выберите изображение.")

    # Настройки -> Заданные значения -> Ок
    def nastr_ok(self):

        with open("settings.txt", 'r+') as f:
            awd = f.readlines()
            # очистка и запись новых настроек из окна в nastr
            nastr.clear()
            nastr.append(int(self.s_w.textEdit.toPlainText().strip()))
            nastr.append(int(self.s_w.textEdit_2.toPlainText().strip()))
            nastr.append(int(self.s_w.textEdit_3.toPlainText().strip()))
            nastr.append(int(self.s_w.textEdit_4.toPlainText().strip()))
            nastr.append(int(self.s_w.textEdit_5.toPlainText().strip()))
            nastr.append(int(self.s_w.textEdit_6.toPlainText().strip()))

            # удаление содержимого в settings.txt
            f.seek(0)
            f.truncate()

            # запись nastr в 'settings.txt'
            f.write("{}\n{}\n{}\n{}\n{}\n{}".format(nastr[0], nastr[1], nastr[2], nastr[3], nastr[4], nastr[5]))

        self.ui.textEdit.setText("Настройки сохранены!")
        self.s_w.close()

    # Настройки -> Заданные значения -> Отмена
    def nastr_cancel(self):
        self.s_w.close()
        self.ui.textEdit.setText("Cancel")

    # Настройки -> Заданные значения
    def sec_win(self):
        if self.s_w is None:
            self.s_w = Sec_win()

        if len(nastr) == 6:
            self.s_w.textEdit.setText(str(nastr[0]))
            self.s_w.textEdit_2.setText(str(nastr[1]))
            self.s_w.textEdit_3.setText(str(nastr[2]))
            self.s_w.textEdit_4.setText(str(nastr[3]))
            self.s_w.textEdit_5.setText(str(nastr[4]))
            self.s_w.textEdit_6.setText(str(nastr[5]))
        self.s_w.pushButton.clicked.connect(self.nastr_ok)
        self.s_w.pushButton_2.clicked.connect(self.nastr_cancel)
        self.s_w.show()
        # else:
        #     self.ui.textEdit.setText("Файл с настройками поврежден. Пожалуйста загрузите корректный \"settings.txt\"")

    def next_img(self):
        import os.path
        global file_img
        if file_img is None:
            self.ui.textEdit.setText("Выберите первое изображение")
        else:
            # 0001.png
            fl = file_img[0].split("/")

            # 0001 and png
            fla = fl[-1].split(".")

            # 2
            flb = str(int(fla[0]) + 1)

            # 3
            ln = len(fla[0]) - len(flb)

            st = '0' * ln + flb + '.' + fla[1]

            file_img = (file_img[0].replace(fl[-1], st), file_img[1],)

            flc = file_img[0].split("/")
            self.ui.textEdit_2.setText("{}/{}".format(flc[-2], flc[-1]))

            global image, blur, thresh

            if not os.path.exists(file_img[0]):
                self.ui.textEdit.setText("Изображение не найдено.")
            else:
                image = cv2.imread(file_img[0])
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(image_gray, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)

                # blur 70 255 thresh_binary
                ret, thresh = cv2.threshold(blur, nastr[3], 255, cv2.THRESH_BINARY)
                thresh = cv2.GaussianBlur(thresh, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)
                Image.open(file_img[0]).resize((854, 641)).save("dop/out_res.png")
                pixmap = QPixmap("dop/out_res.png")
                self.ui.label.setPixmap(pixmap)

                self.check()


app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())
