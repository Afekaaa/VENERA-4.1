import matplotlib.pyplot as plt
from math import atan, sqrt, exp, sin, cos, pi


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def next_step():
    global x, y, h, v, a, t_now, mass_now, v_x, v_y
    x = next_x()
    y = next_y()
    v_x = next_velocity_x()
    v_y = next_velocity_y()
    v = next_velocity()
    h = altitude()
    t_now += delta_t
    a = acceleration(t_now)
    mass_now = mass(t_now)


def create_time_list(N):
    return [i * delta_t for i in range(N)]


def next_x():
    return x + v_x * delta_t


def next_y():
    return y + v_y * delta_t


# коэф для проекции на ось Oy
def y_coef_fi():
    ang_teta = abs(teta())
    ang_gamma = abs(gamma())
    if x * y >= 0 and ang_teta + pi / 2 - ang_gamma <= pi / 2:
        return -sin(abs(ang_teta + ang_gamma))
    elif x * y >= 0 and ang_teta + pi / 2 - ang_gamma > pi / 2:
        return abs(sin(abs(ang_teta - ang_gamma)))
    elif x * y < 0 and (ang_teta - ang_gamma) <= 0:
        return sin(abs(ang_gamma - ang_teta))
    elif x * y < 0 and (ang_teta - ang_gamma) > 0:
        return -sin(abs(ang_gamma - ang_teta))


# коэф для проекции на ось Ох
def x_coef_fi():
    ang_teta = abs(teta())
    ang_gamma = abs(gamma())
    if x * y >= 0 and ang_teta + ang_gamma <= pi / 2:
        return -abs(cos(ang_teta + ang_gamma))
    elif x * y >= 0 and ang_teta + ang_gamma > pi / 2:
        return abs(cos(ang_teta + ang_gamma))
    elif x * y < 0 and ang_teta + ang_gamma <= pi / 2:
        return round(-cos(abs(ang_gamma - ang_teta)), 15)
    elif x * y < 0 and ang_teta + ang_gamma > pi / 2:
        return round(-cos(abs(ang_gamma - ang_teta)), 15)


def mass(time_now):
    global initial_mass, t1, t2, t3, t4
    global fuel_consumption1, fuel_consumption2, fuel_consumption3, fuel_consumption4
    global m_stage1, m_stage2, m_stage3, m_stage4
    global stage1_mass, stage12_mass, stage123_mass, stage1234_mass
    global time_list, delta_t
    global real_mass

    if time_now <= t1:
        real_mass = initial_mass - time_now * (fuel_consumption1 + fuel_consumption2)
    elif time_now <= t2:
        real_mass = initial_mass - (time_now - t1) * fuel_consumption2 - m_stage1
    elif time_now <= t3:
        real_mass = initial_mass - (time_now - t2) * fuel_consumption3 - m_stage2 - m_stage1
    elif time_now <= t4:
        real_mass = initial_mass - (time_now - t2) * fuel_consumption3 - m_stage2 - m_stage1
    elif time_now <= t5:
        real_mass = real_mass
    elif time_now <= t6:
        real_mass = initial_mass - (t3 - t2) * fuel_consumption3 - (
                t4 - t3) * fuel_consumption3 - m_stage2 - m_stage1 - (time_now - t5) * fuel_consumption3
    # print(real_mass)
    return real_mass


def altitude():
    return sqrt(x ** 2 + y ** 2) - r


# сила тяги?
def traction_force(time_now):
    global specific_impulse1, specific_impulse2, specific_impulse3, specific_impulse4

    if time_now <= t1:
        return specific_impulse1[0] + specific_impulse2[0]
    elif time_now <= t2:
        return specific_impulse2[1]
    elif time_now <= t3:
        return specific_impulse3[1]
    elif time_now <= t4:
        return specific_impulse3[1] * 0.25
    elif time_now <= t5:
        return 0
    else:
        return specific_impulse3[1]


# 90 - крена
def gamma():
    return atan(y / x)


# тангажа + 90 - крена
def teta():
    h1 = turn_start_altitude
    h2 = turn_end_altitude
    chill = 0
    if h <= h1:
        angle = 0
    elif h <= h2:
        angle = (pi / 2 - chill) * (h - h1) / (h2 - h1)
    else:
        angle = pi / 2 - chill
    return angle


# Ускорение свободного падения
def acceleration_of_gravity():
    center_distance = (r + h)
    return (G * M) / (center_distance ** 2)


# сида тяжести
def gravity(time_now):
    chill_coef = 0.83
    if h > 45000:
        return mass_now * acceleration_of_gravity() * chill_coef
    return mass_now * acceleration_of_gravity()


# плотность воздуха
def air_density() -> float:
    if t_now < 40:
        return p0 * exp(-acceleration_of_gravity() * m * h / (R * (t0 - 6.5 * (h / 1000))))
    return p0 * exp(-acceleration_of_gravity() * m * h / (R * (t0 - 240)))


# сила сопротивления воздуха, Fсопр
def air_resistance_force():
    # print('air_density', air_density())
    return 0.5 * Cd * S * v ** 2 * air_density()


def acceleration_x(time_now):
    global a_x
    #print('traction_force', traction_force(time_now))
    #print('x_coef_fi', x_coef_fi())
    b = traction_force(time_now) * x_coef_fi()
    c = air_resistance_force() * x_coef_fi()
    d = sign(x) * gravity(time_now) * cos(abs(gamma()))
    e = mass_now
    a_x = (b - c - d) / e
    #print('a_x:', a_x)
    return a_x


def acceleration_y(time_now):
    global a_y
    # print(y_coef_fi())
    b = traction_force(time_now) * y_coef_fi()
    c = air_resistance_force() * y_coef_fi()
    d = sign(y) * gravity(time_now) * sin(abs(gamma()))
    e = mass_now
    a_y = (b - c - d) / e
    return a_y


def acceleration(time_now):
    return sqrt(acceleration_x(time_now) ** 2 + acceleration_y(time_now) ** 2)


def next_velocity_x():
    return v_x + a_x * delta_t


def next_velocity_y():
    return v_y + a_y * delta_t


# скорость в следующий момент времени, v
def next_velocity() -> float:
    return sqrt(v_x ** 2 + v_y ** 2)


# график скоростей
def plots():
    velocity_list = []
    x_list = []
    y_list = []
    a_list = []
    h_list = []
    mass_list = []
    teta_list = []
    gamma_list = []
    a_x_list = []
    a_y_list = []
    x_okr = [i for i in range(-600_000, 600_001)]
    y1 = [sqrt(r ** 2 - x ** 2) for x in x_okr]
    y2 = [-sqrt(r ** 2 - x ** 2) for x in x_okr]

    for i in time_list:
        velocity_list.append(v)
        mass_list.append(mass_now)
        x_list.append(x)
        y_list.append(y)
        a_list.append(a)
        h_list.append(h)
        a_x_list.append(a_x)
        a_y_list.append(a_y)
        teta_list.append(teta() * 180 / pi)
        gamma_list.append(gamma() * 180 / pi)
        next_step()

        # print(i)
        # print('v = ', v)
        # print('t: ', t_now, 'x: ', x, 'y: ', y, 'a: ', a, 'h: ', h, 'v:', v, 'teta: ', teta(), 'gamma: ', gamma())

    tmp, plot = plt.subplots()
    plot.set_title('Изменение скорости ракеты')
    plot.set_xlabel('Время в секундах')
    plot.set_ylabel('Скорость в м/с')
    plot.plot(time_list, velocity_list)
    plot.plot(ksp_time_list, ksp_velocity_list)
    plot.grid()

    tmp, plot1 = plt.subplots()
    plot1.set_title('Изменение положения ракеты от-но центра Кербина')
    plot1.set_xlabel('х кооридната в м')
    plot1.set_ylabel('y координата в м')
    plot1.plot(x_list, y_list)
    plot1.plot(ksp_x_coor_list, ksp_y_coor_list)
    plot1.plot(x_okr, y1, color='blue')
    plot1.plot(x_okr, y2, color='blue')
    plot1.grid()

    tmp, plot2 = plt.subplots()
    plot2.set_title('Изменение высоты')
    plot2.set_xlabel('Время в секундах')
    plot2.set_ylabel('Высота в метрах')
    plot2.plot(time_list, h_list)
    plot2.plot(ksp_time_list, ksp_h_list)
    plot2.grid()

    tmp, plot3 = plt.subplots()
    plot3.set_title('Изменение ускорения ракеты')
    plot3.set_xlabel('Время в секундах')
    plot3.set_ylabel('Ускорение в м/с^2')
    plot3.plot(time_list, a_list)
    plot3.grid()

    tmp, plot4 = plt.subplots()
    plot4.set_title('Изменение массы ракеты')
    plot4.set_xlabel('Время в секундах')
    plot4.set_ylabel('Масса в килограммах')
    plot4.plot(time_list, mass_list, label='Рассчетные данные')
    plot4.plot(ksp_time_list, ksp_mass_list, label='ksp')
    plot4.legend('Расчетные данные', 'ksp')
    plot4.grid(True)

    tmp, plot_gamma = plt.subplots()
    plot_gamma.set_title('Изменение gamma ракеты')
    plot_gamma.set_xlabel('Время в секундах')
    plot_gamma.set_ylabel('Скорость в радианах')
    plot_gamma.plot(time_list, gamma_list)
    plot_gamma.grid()

    tmp, plot_teta = plt.subplots()
    plot_teta.set_title('Изменение teta ракеты')
    plot_teta.set_xlabel('Время в секундах')
    plot_teta.set_ylabel('Скорость в радианах')
    plot_teta.plot(time_list, teta_list)
    plot_teta.grid()

    tmp, plot_a_x = plt.subplots()
    plot_a_x.set_title('Изменение ускорения по оси х')
    plot_a_x.set_xlabel('Время в секундах')
    plot_a_x.set_ylabel('Ускорение в м/с^2')
    plot_a_x.plot(time_list, a_x_list)
    plot_a_x.grid()

    tmp, plot_a_y = plt.subplots()
    plot_a_y.set_title('Изменение ускорения по оси y')
    plot_a_y.set_xlabel('Время в секундах')
    plot_a_y.set_ylabel('Ускорение в м/с^2')
    plot_a_y.plot(time_list, a_y_list)
    plot_a_y.grid()


def ksp_data():
    global ksp_mass_list, ksp_time_list, ksp_x_coor_list, ksp_y_coor_list, ksp_h_list, ksp_velocity_list

    with open('data.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            lst = line.split('; ')
            ksp_mass_list.append(float(lst[3]))
            ksp_time_list.append(float(lst[0]))
            ksp_x_coor_list.append(float(lst[4]))
            ksp_y_coor_list.append(float(lst[6]))
            ksp_h_list.append(float(lst[1]))
            ksp_velocity_list.append(float(lst[2]))
        # print('ksp_x_coor_list', ksp_x_coor_list)


# константы атмосферы Кербина
p0 = 1.2255  # плотность воздуха на уровне моря, кг/м^3
t0 = 250.15  # стандартная температурана экваторе в точке старта, Kельвины
m = 0.029  # молярная масса сухого воздуха, кг/моль
R = 8.31  # универсальная газовая постоянная, Дж/(моль * Кельвины)

# физические константы Кербина
G = 6.6743015e-11  # гравитационная постоянная, (м^3) / (c^2 * кг^2)
r = 600_000  # радиус Кербина, м
M = 5.2915158e22  # масса Кербина, кг

# время
N = 10_000  # количество пересчетов
t = 700  # общее время полета в секундах
delta_t = t / N
t1 = 93.07999791949987  # время работы первой ступени
t2 = 141.07999684661627  # время работы второй ступени
t3 = 218.34003578871489  # время работы третий ступени
t4 = 226.02003704756498  # отключение двигателей
t5 = 689.0509350001812  # включение двигателей
t6 = 697.2309363409877  # отключение двгиателей
time_list = create_time_list(N)

# параметры ракеты-носителя и Венеры-4
initial_mass = 211376.171875  # масса ракеты-носителя и полезного груза в начальный момент времени
m_without_fuel = 75360.171875  # масса ракеты-носителя и полезного груза без топлипа
m_stage1 = initial_mass - 61957  # масса первой ступени, кг 61957
m_stage2 = initial_mass - m_stage1 - 30550  # масса второй ступени, кг 30550
m_stage3 = initial_mass - m_stage1 - m_stage2 - 7700 - 12500  # масса третий ступени, кг 21090
m_stage4 = 7708 / 100  # масса четвертой ступени, кг
m_stage5 = 3074 / 100  # масса посадочного модуля, кг
Cd = 0.5  # обтекание конуса
d = 1.5  # диаметр корпуса ракеты
S = 2.25 * pi  # 0.008 * m_without_fuel  # ToDO: масса с топливом или без? Статья про площадь

# параметры двигателей
fuel_consumption2 = (61957 - 46000) / (t2 - t1)  # расход массы второй ступени
fuel_consumption1 = (initial_mass - 110449) / t1 - fuel_consumption2  # расход массы первой ступени
fuel_consumption3 = (30550 - 22000) / (t3 - t2)  # расход массы третий ступени
fuel_consumption4 = 10  # расход массы четвертой ступени

ogr1 = 0.64
ogr2 = 0.7  # 0.7
chill1 = 1  # 1.2
specific_impulse1 = [167969 * 16 * ogr1, 215000 * 16 * ogr1]
specific_impulse2 = [205161 * 4 * chill1, 240000 * 4 * chill1]
specific_impulse3 = [108197 * 4 * ogr2, 120000 * 4 * ogr2]
specific_impulse4 = [16563, 20000]

# израсходованная масса
stage1_mass = t1 * (fuel_consumption1 + fuel_consumption2) + m_stage1  # после первого этапа
stage12_mass = stage1_mass + (t2 - t1) * fuel_consumption2 + m_stage2  # после второго этапа
stage123_mass = stage12_mass + (t3 - t2) * fuel_consumption3 + m_stage3  # после третьего этапа
stage1234_mass = stage123_mass + (t5 - t4) * fuel_consumption4 + m_stage4  # после четвертого этапа
real_mass = initial_mass

# print('mass:', stage1_mass + stage12_mass)
# print('initial_mass:', initial_mass)

# списки данных из ksp
ksp_time_list = []
ksp_mass_list = []  # список масс космического аппарата
ksp_velocity_list = []  # список скоростей космического аппарата
ksp_x_coor_list = []  # список х координат космического аппарата
ksp_y_coor_list = []  # список у координат космического аппарата
ksp_h_list = []  # список высот космического аппарата

ksp_data()

# исходные данные о полете
x0 = -576069.9323614247  # начальная х координата
y0 = 168061.02324176938  # начальная у координата
h0 = 85  # начальная высота
turn_start_altitude = 250  # высота начала вращения
turn_end_altitude = 45_000  # высота, на которой кончается поворот ракеты-носителя
v0 = 0  # начальная скорость

# глобальные переменные: x, y, h, v, a
x = x0
y = y0
h = h0
v = v0
v_x = v * x_coef_fi()
v_y = v * y_coef_fi()
t_now = 0
mass_now = initial_mass
a = acceleration(0)
a_x = acceleration_x(0)
a_y = acceleration_y(0)
# print('g:', acceleration_of_gravity())

plots()

# angle_plot()
# print(ksp_mass_list)

plt.show()

# P0 = 845000  # расчетная тяга двигателя, когда давление на выходе сопла совпадает с давлением газа окружающей среды, Н
# ve = 3070  # выходная скорость
# pe = 1.5  # выходное давление
# ae = 6.91  # площадь сопла
# imp1 = 283  # удельный импульс двигателя первой ступени на уровне моря
# imp2 = 348  # удельный импульс двигателя второй ступени для вакуума
