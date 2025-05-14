import numpy as np
from scipy.integrate import solve_ivp
import pygame
from datetime import datetime

class StringPhysics:
    """Handles the physical model of the vibrating string."""
    def __init__(self, N=50, L=1.0, Mtot=0.12, T=10.0, b=0.05, F0=1.0, f=None):
        self.N = N
        self.L = L
        self.l = L / N
        self.m = Mtot/N
        self.T = T
        self.b = b
        self.F0 = F0
        self.F0_target = F0
        self.F0_previous = F0
        self.F0_transition_time = 0.5
        self.M_total = Mtot
        self.mu = self.M_total / L
        self.f_0 = (1 / (2 * L)) * np.sqrt(T / self.mu)
        self.f = f if f is not None else self.f_0
        self.w = 2 * np.pi * self.f
        self.ramp_time = 1.0
        self.state = np.zeros(2 * (N - 1))
        self.resetting = False
    
    def ode(self, t, state, damping_coeff, force_amplitude):
        y = state[:self.N-1]
        v = state[self.N-1:]
        dydt = v
        dvdt = np.zeros(self.N-1)
        k = self.T / (self.m * self.l)
        damping = damping_coeff / self.m
        for i in range(self.N-1):
            y_prev = 0 if i == 0 else y[i-1]
            y_next = force_amplitude * np.cos(self.w * t) / self.T if i == self.N-2 else y[i+1]
            dvdt[i] = k * (y_next - 2 * y[i] + y_prev) - damping * v[i]
        return np.concatenate([dydt, dvdt])

    def ode_wrapper(self, t, state):
        force_amplitude = self.F0 * min(t / self.ramp_time, 1.0) if not self.resetting else 0.0
        return self.ode(t, state, self.b, force_amplitude)

    def set_normal_mode(self, mode):
        self.f = mode * self.f_0
        self.w = 2 * np.pi * self.f

    def update_F0(self, dt):
        if self.F0 != self.F0_target:
            delta_F0 = (self.F0_target - self.F0) * (dt / self.F0_transition_time)
            if abs(self.F0_target - self.F0) < abs(delta_F0):
                self.F0 = self.F0_target
            else:
                self.F0 += delta_F0

class AxesVisualizer:
    """Handles drawing of axes with ticks and labels."""
    def __init__(self, width, height, left_margin, right_margin, top_margin=100, bottom_margin=100):
        self.width = width
        self.height = height
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.effective_width = width - left_margin - right_margin
        self.effective_height = height - top_margin - bottom_margin
        self.axes_color = (100, 100, 100)
        self.tick_length = 10
        self.tick_interval_x = 0.2
        self.tick_interval_y = 1.0
        self.axes_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.show_axes = False

    def render(self, screen, physics):
        if not self.show_axes:
            return
        x_scale = self.effective_width / physics.L
        y_scale = self.effective_height / 10.0
        center_y = self.height // 2
        
        pygame.draw.line(screen, self.axes_color, (self.left_margin, center_y), (self.width - self.right_margin, center_y), 2)
        pygame.draw.line(screen, self.axes_color, (self.left_margin, self.top_margin), (self.left_margin, self.height - self.bottom_margin), 2)
        
        for x in np.arange(0, physics.L + self.tick_interval_x, self.tick_interval_x):
            x_pixel = self.left_margin + x * x_scale
            pygame.draw.line(screen, self.axes_color, (x_pixel, center_y - self.tick_length // 2), (x_pixel, center_y + self.tick_length // 2), 2)
            label = self.axes_font.render(f"{x:.1f}", True, self.axes_color)
            screen.blit(label, (x_pixel - 10, center_y + 5))
        
        y_range = 5.0
        for y in np.arange(-y_range, y_range + self.tick_interval_y, self.tick_interval_y):
            y_pixel = center_y - y * y_scale
            pygame.draw.line(screen, self.axes_color, (self.left_margin - self.tick_length // 2, y_pixel), (self.left_margin + self.tick_length // 2, y_pixel), 2)
            label = self.axes_font.render(f"{y:.0f}", True, self.axes_color)
            screen.blit(label, (self.left_margin - 40, y_pixel - 8))

class StringVisualizer:
    """Handles visualization of the string simulation using Pygame, excluding axes."""
    def __init__(self, width=800, height=600, left_margin=50, right_margin=50, top_margin=100, bottom_margin=100):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Vibrating String Simulation")
        self.clock = pygame.time.Clock()
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.effective_width = width - left_margin - right_margin
        self.effective_height = height - top_margin - bottom_margin
        self.BLACK = (0, 0, 0)
        self.BLUE = (50, 150, 255)
        self.DARK_BLUE = (20, 80, 150)
        self.GRAY = (150, 150, 150)
        self.LIGHT_GRAY = (200, 200, 200, 180)
        self.GREEN = (0, 200, 0)
        self.RED = (200, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.help_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.string_thickness = 4
        self.sphere_radius = 6
        self.sphere_color = (255, 165, 0)
        self.show_controls = True
        self.show_sticks = True
        self.show_spheres = True
        self.show_shadow = False
        self.show_fixed_params = True  # New flag for second line

    def draw_gradient(self):
        for y in range(self.height):
            r = int(240 + (103 - 240) * y / self.height)
            g = int(248 + (216 - 248) * y / self.height)
            b = int(255 + (230 - 255) * y / self.height)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))

    def render(self, physics, t, dt, paused, resetting, M, sphere_indices):
        self.draw_gradient()
        x_scale = self.effective_width / physics.L
        y_scale = self.effective_height / 10.0
        center_y = self.height // 2
        
        # Extract positions
        y = physics.state[:physics.N-1]
        positions = [(self.left_margin + i * physics.l * x_scale, center_y - (0 if i == 0 else y[i-1]) * y_scale) for i in range(physics.N)]
        force_amplitude = physics.F0 * min(t / physics.ramp_time, 1.0) if not resetting else 0.0
        driven_y = center_y - (force_amplitude * np.cos(physics.w * t) / physics.T) * y_scale
        positions.append((self.left_margin + physics.L * x_scale, driven_y))
        
        if self.show_sticks:
            stick_height = 100
            pygame.draw.line(self.screen, self.GREEN, (self.left_margin, center_y), (self.left_margin, center_y + stick_height), 6)
            pygame.draw.line(self.screen, self.RED, (self.left_margin + physics.L * x_scale, center_y), (self.left_margin + physics.L * x_scale, center_y + stick_height), 6)
            ring_pos = (int(self.left_margin + physics.L * x_scale), int(driven_y))
            pygame.draw.circle(self.screen, self.YELLOW, ring_pos, 10, 2)
        
        shadow_positions = [(x + 2, y + 2) for x, y in positions]
        if self.show_shadow:
            pygame.draw.lines(self.screen, self.GRAY, False, shadow_positions, self.string_thickness)
        pygame.draw.lines(self.screen, self.BLUE, False, positions, self.string_thickness)
        
        if self.show_spheres:
            for idx in sphere_indices:
                x, y = positions[idx]
                pygame.draw.circle(self.screen, self.sphere_color, (int(x), int(y)), self.sphere_radius)
        
#        for pos in positions[:-1]:
#            pygame.draw.circle(self.screen, self.DARK_BLUE, (int(pos[0]), int(pos[1])), 5)
        
        # Info box (two lines, second line toggleable)
        status = "Paused" if paused else "Running" if not resetting else "Resetting"
        info_text1 = f"f = {physics.f:.1f} Hz  |  f_0 = {physics.f_0:.1f} Hz  |  dt = {dt:.3f} s  |  b = {physics.b:.2f} kg/s  |  F0 = {physics.F0:.1f} N  |  {status}"
        info_surface1 = self.font.render(info_text1, True, self.BLACK)
        info_rect1 = info_surface1.get_rect(topleft=(10, 10))
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, info_rect1.inflate(20, 10))
        self.screen.blit(info_surface1, (20, 15))
        
        if self.show_fixed_params:
            info_text2 = f"T = {physics.T} N  |  L = {physics.L} m  |  Mtotal = {physics.M_total:.3f} kg  |  m = {physics.m} kg  |  mu = {physics.mu:.3f} kg/m"
            info_surface2 = self.font.render(info_text2, True, self.BLACK)
            info_rect2 = info_surface2.get_rect(topleft=(10, 40))
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, info_rect2.inflate(20, 10))
            self.screen.blit(info_surface2, (20, 45))
        
        if self.show_controls:
            controls = [
                "Controls:",
                "UP: Increase f",
                "DOWN: Decrease f",
                "LEFT: Decrease dt",
                "RIGHT: Increase dt",
                "Z: Decrease b",
                "X: Increase b",
                "K: Decrease F0",
                "L: Increase F0",
                "H: Toggle Help",
                "S: Toggle Sticks",
                "J: Toggle Shadow",
                "SPACE: Pause/Resume",
                "R: Reset",
                "F: Toggle F0 On/Off",
                "D: Toggle Spheres",
                "NUMPAD +/-: Inc/Dec Spheres",
                "P: Print Log",
                "G: Toggle Axes",
                "1-9: Set f = n*f_0",
                "Y: Toggle Fixed Params"
            ]
            mid = (len(controls) + 1) // 2
            col1, col2 = controls[:mid], controls[mid:]
            start_y = self.height - self.bottom_margin - 180  # Moved up
            for i, line in enumerate(col1):
                text = self.help_font.render(line, True, self.DARK_BLUE)
                self.screen.blit(text, (10, start_y + i * 20))
            for i, line in enumerate(col2):
                text = self.help_font.render(line, True, self.DARK_BLUE)
                self.screen.blit(text, (200, start_y + i * 20))

    def flip(self):
        pygame.display.flip()
        self.clock.tick(60)

class StringSimulation:
    """Main simulation class integrating physics, visualization, and axes."""
    def __init__(self, N=50, L=1.0, Mtot=0.1, T=10.0, b=0.05, F0=1.0, f=None, width=800, height=600, left_margin=50, right_margin=50, top_margin=100, bottom_margin=100):
        self.physics = StringPhysics(N, L, Mtot, T, b, F0, f)
        self.visualizer = StringVisualizer(width, height, left_margin, right_margin, top_margin, bottom_margin)
        self.axes = AxesVisualizer(width, height, left_margin, right_margin, top_margin, bottom_margin)
        self.t = 0.0
        self.dt = 0.01
        self.paused = False
        self.resetting = False
        self.reset_start_time = 0.0
        self.b_initial = b
        self.k_damping = 2.0
        self.b_max = 1.0
        self.M = min(10, N)
        self.sphere_indices = np.linspace(0, N, self.M, dtype=int)  # Include endpoint

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.physics.f += 0.5
                        self.physics.w = 2 * np.pi * self.physics.f
                    elif event.key == pygame.K_DOWN:
                        self.physics.f = max(0.5, self.physics.f - 0.5)
                        self.physics.w = 2 * np.pi * self.physics.f
                    elif event.key == pygame.K_LEFT:
                        self.dt = max(0.001, self.dt - 0.001)
                    elif event.key == pygame.K_RIGHT:
                        self.dt = min(0.05, self.dt + 0.001)
                    elif event.key == pygame.K_z:
                        self.physics.b = max(0.0, self.physics.b - 0.01)
                        self.b_initial = self.physics.b
                    elif event.key == pygame.K_x:
                        self.physics.b = min(0.5, self.physics.b + 0.01)
                        self.b_initial = self.physics.b
                    elif event.key == pygame.K_h:
                        self.visualizer.show_controls = not self.visualizer.show_controls
                    elif event.key == pygame.K_s:
                        self.visualizer.show_sticks = not self.visualizer.show_sticks
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        if not self.paused and not self.resetting:
                            self.physics.F0_target = self.physics.F0
                    elif event.key == pygame.K_r and not self.resetting:
                        self.resetting = True
                        self.physics.resetting = True
                        self.reset_start_time = self.t
                        self.physics.F0_target = 0.0
                    elif event.key == pygame.K_d:
                        self.visualizer.show_spheres = not self.visualizer.show_spheres
                    elif event.key == pygame.K_k:
                        self.physics.F0_target = max(0.0, self.physics.F0_target - 0.1)
                        self.physics.F0_initial = self.physics.F0_target
                    elif event.key == pygame.K_l:
                        self.physics.F0_target = min(5.0, self.physics.F0_target + 0.1)
                        self.physics.F0_initial = self.physics.F0_target
                    elif event.key == pygame.K_j:
                        self.visualizer.show_shadow = not self.visualizer.show_shadow
                    elif event.key == pygame.K_KP_PLUS:
                        self.M = min(100, self.M + 1)
                        self.sphere_indices = np.linspace(0, self.physics.N, self.M, dtype=int)
                    elif event.key == pygame.K_KP_MINUS:
                        self.M = max(1, self.M - 1)
                        self.sphere_indices = np.linspace(0, self.physics.N, self.M, dtype=int)
                    elif event.key == pygame.K_f:
                        if self.physics.F0_target != 0.0:
                            self.physics.F0_previous = self.physics.F0_target
                            self.physics.F0_target = 0.0
                        else:
                            self.physics.F0_target = self.physics.F0_previous
                        self.physics.F0_initial = self.physics.F0_target
                    elif event.key == pygame.K_p:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"log_{timestamp}.txt"
                        with open(filename, 'w') as log_file:
                            log_file.write("Simulation Parameters:\n")
                            log_file.write(f"Simulation time: {self.t:.3f} s\n")
                            log_file.write(f"T (tension): {self.physics.T} N\n")
                            log_file.write(f"f (frequency): {self.physics.f} Hz\n")
                            log_file.write(f"m (mass per segment): {self.physics.m} kg\n")
                            log_file.write(f"l (segment length): {self.physics.l} m\n")
                            log_file.write(f"L (total length): {self.physics.L} m\n")
                            log_file.write(f"N (number of segments): {self.physics.N}\n")
                            log_file.write(f"Mtotal (total mass): {self.physics.M_total} kg\n")
                            log_file.write(f"F0 (driving amplitude): {self.physics.F0} N\n")
                            log_file.write(f"f0 (fundamental frequency): {self.physics.f_0:.3f} Hz\n")
                            log_file.write(f"b (damping coefficient): {self.physics.b} kg/s\n")
                            log_file.write(f"mu (linear mass density): {self.physics.mu} kg/m\n")
                            log_file.write("\n")
                            log_file.write("Positions:\n")
                            log_file.write("Index | x (m) | y (m)\n")
                            for i in range(self.physics.N + 1):
                                if i == 0:
                                    x = 0.0
                                    y = 0.0
                                elif i < self.physics.N:
                                    x = i * self.physics.l
                                    y = self.physics.state[i - 1]
                                else:
                                    x = self.physics.L
                                    y = (self.physics.F0 * min(self.t / self.physics.ramp_time, 1.0) * np.cos(self.physics.w * self.t) / self.physics.T) if not self.resetting else 0.0
                                log_file.write(f"{i} | {x:.6f} | {y:.6f}\n")
                    elif event.key == pygame.K_g:
                        self.axes.show_axes = not self.axes.show_axes
                    elif event.key == pygame.K_y:
                        self.visualizer.show_fixed_params = not self.visualizer.show_fixed_params  # Toggle second line
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        mode = event.key - pygame.K_0
                        self.physics.set_normal_mode(mode)
            
            if not self.paused:
                self.physics.update_F0(self.dt)
            
            if self.resetting:
                elapsed_reset_time = self.t - self.reset_start_time
                self.physics.b = self.b_initial * np.exp(self.k_damping * elapsed_reset_time)
                if self.physics.b >= self.b_max:
                    self.physics.state = np.zeros(2 * (self.physics.N - 1))
                    self.physics.b = self.b_initial
                    self.resetting = False
                    self.physics.resetting = False
                    self.paused = True
            
            if not self.paused:
                t_span = (self.t, self.t + self.dt)
                sol = solve_ivp(self.physics.ode_wrapper, t_span, self.physics.state, method='RK45', rtol=1e-5, atol=1e-7)
                self.physics.state = sol.y[:, -1]
                self.t = sol.t[-1]
            
            self.visualizer.render(self.physics, self.t, self.dt, self.paused, self.resetting, self.M, self.sphere_indices)
            self.axes.render(self.visualizer.screen, self.physics)
            self.visualizer.flip()
        
        pygame.quit()

if __name__ == "__main__":
    sim = StringSimulation(
        N=100, L=1.0, Mtot=0.8, T=5.0, b=0.02, F0=1.0, f=None,
        width=800, height=600, left_margin=50, right_margin=50, top_margin=100, bottom_margin=100
    )
    sim.run()
