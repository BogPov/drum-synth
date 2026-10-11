import numpy as np
import argparse
import wave

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Физический синтез звука удара по барабану.\n"
            "Решается волновое уравнение мембраны с затуханием"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-n", "--name", type=str , default="sound",
        help="Имя выходного файла (указывать без расширения)"
    )

    p.add_argument(
        "-R", "--radius", type=float, default=0.15,
        help="Радиус барабана (мембраны), метры",
    )
    p.add_argument(
        "-N", "--grid", type=int, default=140,
        help="Число точек сетки вдоль каждой оси (N×N)",
    )
    p.add_argument(
        "-T", "--tension", type=float, default=2000.0,
        help="Сила натяжения мембраны T, Н/м",
    )
    p.add_argument(
        "-rho", "--density", type=float, default=0.2,
        help="Поверхностная плотность мембраны rho, кг/м²",
    )
    p.add_argument(
        "-g", "--gamma", type=float, default=22.0,
        help="Коэффициент равномерного затухания, 1/с",
    )
    p.add_argument(
        "-t", "--duration", type=float, default=0.6,
        help="Длительность синтеза, секунды.",
    )

    return p.parse_args()


args = parse_args()
filename = args.name + ".wav"

# радиус барабана (мембраны)
R = args.radius

# число точек сетки вдоль каждой оси (N×N)
N = args.grid

# сила натяжения мембраны
T_tens = args.tension     

# поверхностная плотность мембраны
rho = args.density        

# коэффициент равномерного затухания
gamma = args.gamma

# длительность синтеза
T_total = args.duration

# скорость распространения волны по мембране
c = float(np.sqrt(T_tens / rho))

# Пространственная сетка
x = np.linspace(-R, R, N)
y = np.linspace(-R, R, N)
X, Y = np.meshgrid(x, y, indexing='ij')
dx = x[1] - x[0]
dy = y[1] - y[0]

# Маска: внутри круга
r2 = X**2 + Y**2
mask = r2 <= R**2
edge = r2 > (R - dx)**2

# Шаг по времени
dt = 0.9 / (c * np.sqrt(1.0/dx**2 + 1.0/dy**2))
nt = int(T_total / dt)
sr_sim = 1.0 / dt

print(f"Симуляция: dt = {dt:.3e} с, шагов = {nt}, частота ≈ {sr_sim:.0f} Гц")


# Начальные условия: удар

u0 = np.zeros((N, N))
#точка удара
x0, y0 = 0.45 * R, 0.0
# ширина импульса        
sigma_s = 0.05 * R            
v0 = np.zeros((N, N))
v0[mask] = np.exp(-((X[mask] - x0)**2 + (Y[mask] - y0)**2) / (2 * sigma_s**2))

u1 = u0 + dt * v0
u1[~mask] = 0
u1[edge] = 0

# Точка записи
x_rec, y_rec = 0.6 * R, 0.0
i_rec = np.argmin(np.abs(x - x_rec))
j_rec = np.argmin(np.abs(y - y_rec))

# Основной цикл

audio = np.zeros(nt)

for n in range(nt):
    lap = np.zeros_like(u1)
    lap[1:-1, 1:-1] = (
        (u1[2:, 1:-1] - 2*u1[1:-1, 1:-1] + u1[:-2, 1:-1]) / dx**2
        + (u1[1:-1, 2:] - 2*u1[1:-1, 1:-1] + u1[1:-1, :-2]) / dy**2
    )

    u2 = (2*u1 - (1 - gamma*dt)*u0 + (c*dt)**2 * lap) / (1 + gamma*dt)
    u2[~mask] = 0
    u2[edge] = 0

    audio[n] = u2[i_rec, j_rec]
    u0, u1 = u1, u2

# Убираем DC, нормируем
audio -= np.mean(audio)
audio /= (np.max(np.abs(audio)) + 1e-12)

# Мягкое затухание в конце, чтобы не было щелчка
fade = int(0.02 * sr_sim)
audio[-fade:] *= np.linspace(1, 0, fade)


# 24-битный WAV (3 байта / сэмпл)
peak = 2**23 - 1
audio_24 = np.clip(audio * peak, -peak, peak).astype(np.int32)

# int32 -> 3 байта
buf = audio_24.astype('<i4').tobytes()
data_bytes = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 4)[:, :3].tobytes()

with wave.open(filename, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(3)         
    wf.setframerate(sr_sim)  
    wf.writeframes(data_bytes)

print(f"Сохранено: {filename} ({T_total} с, 24 бита, {sr_sim/1000:.0f} кГц)")