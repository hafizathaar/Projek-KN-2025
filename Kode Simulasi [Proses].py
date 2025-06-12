import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import messagebox

def proyektil_tanpa_drag(v0, sudut, dt=0.01, t_max=10):
    g = 9.81
    sudut_rad = np.radians(sudut)
    vx0 = v0 * np.cos(sudut_rad)
    vy0 = v0 * np.sin(sudut_rad)

    x, y, t, vx_list, vy_list = [0], [0], [0], [vx0], [vy0]
    X = np.array([0, 0, vx0, vy0])  # [x, y, vx, vy]

    while X[1] >= 0 and t[-1] < t_max:
        def f(X):
            return np.array([
                X[2],       # dx/dt = vx
                X[3],       # dy/dt = vy
                0,          # dvx/dt = 0
                -g          # dvy/dt = -g
            ])
        k1 = f(X)
        k2 = f(X + 0.5 * dt * k1)
        k3 = f(X + 0.5 * dt * k2)
        k4 = f(X + dt * k3)
        X_new = X + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        x.append(X_new[0])
        y.append(X_new[1])
        t.append(t[-1] + dt)
        vx_list.append(X_new[2])
        vy_list.append(X_new[3])
        X = X_new

    return np.array(x), np.array(y), np.array(t), np.array(vx_list), np.array(vy_list)

def proyektil_dengan_drag(v0, sudut, m, A, Cd, rho, dt=0.01, t_max=10):
    g = 9.81
    sudut_rad = np.radians(sudut)
    vx0 = v0 * np.cos(sudut_rad)
    vy0 = v0 * np.sin(sudut_rad)

    x, y, t, vx_list, vy_list = [0], [0], [0], [vx0], [vy0]
    X = np.array([0, 0, vx0, vy0])  # [x, y, vx, vy]

    while X[1] >= 0 and t[-1] < t_max:
        def f(X):
            vx, vy = X[2], X[3]
            v = np.sqrt(vx**2 + vy**2)
            Fd = 0.5 * Cd * rho * A * v**2 if v > 0 else 0
            ax = -Fd * (vx / v) / m if v > 0 else 0
            ay = -g - (Fd * (vy / v) / m if v > 0 else 0)
            return np.array([
                vx,
                vy,
                ax,
                ay
            ])
        k1 = f(X)
        k2 = f(X + 0.5 * dt * k1)
        k3 = f(X + 0.5 * dt * k2)
        k4 = f(X + dt * k3)
        X_new = X + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        x.append(X_new[0])
        y.append(X_new[1])
        t.append(t[-1] + dt)
        vx_list.append(X_new[2])
        vy_list.append(X_new[3])
        X = X_new

    return np.array(x), np.array(y), np.array(t), np.array(vx_list), np.array(vy_list)

def simpson_integral(t, vx, vy):
    n = len(t) - 1
    if n % 2 != 0:
        n -= 1  # Pastikan jumlah titik genap untuk Simpson
    h = t[1] - t[0]
    v = np.sqrt(vx[:n+1]**2 + vy[:n+1]**2)
    integral = v[0] + v[n]
    for i in range(1, n):
        integral += 4 * v[i] if i % 2 == 1 else 2 * v[i]
    return (h / 3) * integral

class SimulasiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulasi Gerak Parabola")
        self.root.geometry("540x380")

        self.params = {
            'v0': tk.StringVar(value="50"),
            'sudut': tk.StringVar(value="45"),
            'm': tk.StringVar(value="1.0"),
            'A': tk.StringVar(value="0.01"),
            'Cd': tk.StringVar(value="0.47"),
            'rho': tk.StringVar(value="1.225"),
            'simtime': tk.StringVar(value="10")
        }

        # Label, Entry, dan keterangan batas nilai input
        tk.Label(root, text="Kecepatan Awal (m/s):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['v0']).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(root, text="( > 0 )").grid(row=0, column=2, padx=2, sticky="w")

        tk.Label(root, text="Sudut (°):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['sudut']).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(root, text="( 0 – 90 )").grid(row=1, column=2, padx=2, sticky="w")

        tk.Label(root, text="Massa (kg):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['m']).grid(row=2, column=1, padx=5, pady=5)
        tk.Label(root, text="( > 0 )").grid(row=2, column=2, padx=2, sticky="w")

        tk.Label(root, text="Luas Penampang (m²):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['A']).grid(row=3, column=1, padx=5, pady=5)
        tk.Label(root, text="( > 0 )").grid(row=3, column=2, padx=2, sticky="w")

        tk.Label(root, text="Koefisien Drag:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['Cd']).grid(row=4, column=1, padx=5, pady=5)
        tk.Label(root, text="( > 0 )").grid(row=4, column=2, padx=2, sticky="w")

        tk.Label(root, text="Densitas Udara (kg/m³):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['rho']).grid(row=5, column=1, padx=5, pady=5)
        tk.Label(root, text="( > 0 )").grid(row=5, column=2, padx=2, sticky="w")

        tk.Label(root, text="Batas Waktu Simulasi (s):").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.params['simtime']).grid(row=6, column=1, padx=5, pady=5)
        tk.Label(root, text="( > 0 )").grid(row=6, column=2, padx=2, sticky="w")

        # Frame untuk tombol
        button_frame = tk.Frame(root)
        button_frame.grid(row=7, column=0, columnspan=3, pady=12)

        tk.Button(button_frame, text="Jalankan Simulasi", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Tutup Grafik & Ulangi", command=self.kill_graph).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Keluar", command=self.exit_app).pack(side=tk.LEFT, padx=5)

        # State untuk grafik
        self.fig = None
        self.ax = None

    def run_simulation(self):
        try:
            v0 = float(self.params['v0'].get())
            sudut = float(self.params['sudut'].get())
            m = float(self.params['m'].get())
            A = float(self.params['A'].get())
            Cd = float(self.params['Cd'].get())
            rho = float(self.params['rho'].get())
            t_max = float(self.params['simtime'].get())

            if (v0 <= 0 or m <= 0 or A <= 0 or Cd <= 0 or rho <= 0 or
                sudut < 0 or sudut > 90 or t_max <= 0):
                messagebox.showerror("Error", "Masukkan nilai yang valid (positif, sudut 0-90°)!")
                return

            # Kill grafik sebelumnya jika masih terbuka (agar tidak ada multi-window/bug)
            if self.fig is not None:
                try:
                    plt.close(self.fig)
                except Exception:
                    pass
                self.fig = None
                self.ax = None

            # Buat plot baru setiap kali simulasi dijalankan
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel('Jarak (m)')
            self.ax.set_ylabel('Tinggi (m)')
            self.ax.set_title('Simulasi Gerak Proyektil dengan & tanpa Hambatan Udara')
            line_nodrag, = self.ax.plot([], [], 'b-', label='Tanpa Hambatan')
            line_drag, = self.ax.plot([], [], 'r-', label='Dengan Hambatan')
            point_nodrag, = self.ax.plot([], [], 'bo')
            point_drag, = self.ax.plot([], [], 'ro')
            self.ax.legend()

            x_nodrag, y_nodrag, t_nodrag, vx_nodrag, vy_nodrag = proyektil_tanpa_drag(v0, sudut, t_max=t_max)
            x_drag, y_drag, t_drag, vx_drag, vy_drag = proyektil_dengan_drag(v0, sudut, m, A, Cd, rho, t_max=t_max)

            s_nodrag = simpson_integral(t_nodrag, vx_nodrag, vy_nodrag)
            s_drag = simpson_integral(t_drag, vx_drag, vy_drag)

            self.ax.set_xlim(0, max(np.max(x_nodrag), np.max(x_drag)) * 1.1)
            self.ax.set_ylim(0, max(np.max(y_nodrag), np.max(y_drag)) * 1.1)

            def update(frame):
                if frame < len(x_nodrag):
                    line_nodrag.set_data(x_nodrag[:frame], y_nodrag[:frame])
                    point_nodrag.set_data([x_nodrag[frame]], [y_nodrag[frame]])
                if frame < len(x_drag):
                    line_drag.set_data(x_drag[:frame], y_drag[:frame])
                    point_drag.set_data([x_drag[frame]], [y_drag[frame]])
                return line_nodrag, line_drag, point_nodrag, point_drag

            def on_animation_end(event):
                messagebox.showinfo(
                    "Hasil Simulasi",
                    f'Tanpa Hambatan:\n'
                    f'Jarak = {np.max(x_nodrag):.2f} m, Waktu = {t_nodrag[-1]:.2f} s\n'
                    f'Panjang Lintasan = {s_nodrag:.2f} m\n\n'
                    f'Dengan Hambatan:\n'
                    f'Jarak = {np.max(x_drag):.2f} m, Waktu = {t_drag[-1]:.2f} s\n'
                    f'Panjang Lintasan = {s_drag:.2f} m'
                )

            ani = FuncAnimation(
                self.fig, update, frames=range(max(len(x_nodrag), len(x_drag))),
                interval=20, blit=True
            )
            self.fig.canvas.mpl_connect('close_event', on_animation_end)
            plt.show()

        except ValueError:
            messagebox.showerror("Error", "Masukkan nilai numerik yang valid!")

    def kill_graph(self):
        # Tutup grafik jika ada
        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None
            self.ax = None

    def exit_app(self):
        self.kill_graph()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulasiApp(root)
    root.mainloop()
