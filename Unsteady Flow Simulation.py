import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PARAMETERS
U_inf = 1.0 # freestream velocity
rho = 1.0 # density
chord = 1.0 # chord length 
N_panels = 5 # panels 
dt = 0.05 # time step
num_steps = 600 # total steps
pitch_amplitude = 10  # degrees
pitch_frequency = 0.2 # Hz
vortex_core_size = 0.05 # vortex core size

# INFLUENCE COEFFICENT MATRIX
panel_length = chord / N_panels
panel_center = np.linspace(panel_length/2, chord - panel_length/2, N_panels)

dx_matrix = panel_center[:, None] - panel_center[None, :]
A = (panel_length / (dx_matrix + 1e-20)) / (2 * np.pi)
np.fill_diagonal(A, 0.5)

time_list, lift_list, moment_list = [], [], []
wake_x, wake_y, wake_gamma = [], [], []
gamma_total_prev = 0

all_plate_x, all_plate_y = [], []
all_singularity_x, all_singularity_y = [], []
all_wake_x, all_wake_y = [], []

# TIME LOOP
for step in range(num_steps):
    t = step * dt
    time_list.append(t)

    alpha_deg = pitch_amplitude * np.sin(2 * np.pi * pitch_frequency * t)
    alpha_rad = np.radians(alpha_deg)

    RHS = -U_inf * np.sin(alpha_rad) * np.ones(N_panels)

# WAKE INDUCED VELOCITY 
    if wake_x:
        wake_x_arr = np.array(wake_x)
        wake_y_arr = np.array(wake_y)
        dx_w = panel_center[:, None] - wake_x_arr[None, :]
        dy_w = -wake_y_arr[None, :]
        r2 = dx_w**2 + dy_w**2 + vortex_core_size**2
        induced = np.sum((wake_gamma * dy_w) / (2 * np.pi * r2), axis=1)
        RHS -= induced

    gamma_panel = np.linalg.solve(A, RHS)
    gamma_total = np.sum(gamma_panel)

# SHED VORTEX AND WAKE 
    gamma_shed = gamma_total - gamma_total_prev
    wake_x.append(chord)
    wake_y.append(0.0)
    wake_gamma.append(gamma_shed)
    gamma_total_prev = gamma_total

    if len(wake_x) > 1:
        wake_x_arr = np.array(wake_x)
        wake_y_arr = np.array(wake_y)
        vx_total = U_inf * np.ones_like(wake_x_arr)
        vy_total = np.zeros_like(wake_y_arr)
        for j in range(len(wake_x)):
            dx = wake_x_arr - wake_x_arr[j]
            dy = wake_y_arr - wake_y_arr[j]
            r2 = dx**2 + dy**2 + vortex_core_size**2
            r2[j] = 1e10
            vx_total += -wake_gamma[j] * dy / (2 * np.pi * r2)
            vy_total +=  wake_gamma[j] * dx / (2 * np.pi * r2)

        vx_total += np.random.normal(0, 0.01, size=vx_total.shape)
        vy_total += np.random.normal(0, 0.01, size=vy_total.shape)
        wake_x_arr += vx_total * dt
        wake_y_arr += vy_total * dt

        wake_x = list(wake_x_arr)
        wake_y = list(wake_y_arr)

# LIFT AND MOMENT
    lift = rho * U_inf * gamma_total
    moment = -rho * U_inf * np.sum(gamma_panel * (panel_center - 0.0))
    lift_list.append(lift + np.random.normal(0, 0.02 * abs(lift)))
    moment_list.append(moment + np.random.normal(0, 0.02 * abs(moment)))

# DATA FOR SIMULATION
    plate_x, plate_y = [], []
    singularity_x, singularity_y = [], []
    for i in range(N_panels + 1):
        x_local = i * panel_length
        y_local = 0
        x_rot = x_local * np.cos(alpha_rad) - y_local * np.sin(alpha_rad)
        y_rot = x_local * np.sin(alpha_rad) + y_local * np.cos(alpha_rad)
        plate_x.append(x_rot)
        plate_y.append(y_rot)

    for i in range(N_panels):
        x_local = panel_center[i]
        y_local = 0
        x_rot = x_local * np.cos(alpha_rad) - y_local * np.sin(alpha_rad)
        y_rot = x_local * np.sin(alpha_rad) + y_local * np.cos(alpha_rad)
        singularity_x.append(x_rot)
        singularity_y.append(y_rot)

    all_plate_x.append(plate_x)
    all_plate_y.append(plate_y)
    all_singularity_x.append(singularity_x)
    all_singularity_y.append(singularity_y)
    all_wake_x.append(wake_x.copy())
    all_wake_y.append(wake_y.copy())

# PLOT LIFT & MOMENT
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time_list, lift_list)
plt.title("Lift vs Time")
plt.xlabel("Time")
plt.ylabel("Lift")

plt.subplot(1, 2, 2)
plt.plot(time_list, moment_list)
plt.title("Moment vs Time")
plt.xlabel("Time")
plt.ylabel("Moment")
plt.tight_layout()
plt.show()

# SIMULATION
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-1, 20)
ax.set_ylim(-2, 2)
plate_line, = ax.plot([], [], 'k-', lw=2)
wake_points, = ax.plot([], [], 'ro', markersize=3)
singularity_points, = ax.plot([], [], 'bo', markersize=5)

def init():
    plate_line.set_data([], [])
    wake_points.set_data([], [])
    singularity_points.set_data([], [])
    return plate_line, wake_points, singularity_points

def animate(frame):
    plate_line.set_data(all_plate_x[frame], all_plate_y[frame])
    wake_points.set_data(all_wake_x[frame], all_wake_y[frame])
    singularity_points.set_data(all_singularity_x[frame], all_singularity_y[frame])
    return plate_line, wake_points, singularity_points

ani = animation.FuncAnimation(fig, animate, frames=num_steps, init_func=init, blit=True, interval=40)
plt.show()
