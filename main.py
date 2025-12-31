"""
RHEO SNN - Brain & Survival Lab
Main Application with Settings Menu, Multi-Map, and Persistent Memory
"""
import numpy as np
import pygame
import time
import os

from src.environment.simulation import Environment
from src.encoding import Encoder
from src.decoding import Decoder
from src.network import Network
from src.monitor import Monitor
from src.utils.file_manager import setup_project_structure, get_experiments_path, get_brain_weights_path

# Initialize project structure at import time
setup_project_structure()

# ============== DISPLAY CONSTANTS ==============
SIM_WIDTH = 900
SIM_HEIGHT = 700
LOG_PANEL_WIDTH = 350
WINDOW_WIDTH = SIM_WIDTH + LOG_PANEL_WIDTH  # 1250

# Default agent configuration
DEFAULT_CONFIG = {
    'num_sensors': 10,
    'sensor_range': 180,
    'num_hidden': 50,
}
DT = 1.0
NUM_MOTORS = 2
PANIC_THRESHOLD = 0.75

# Speed settings
MIN_SPEED = 1
MAX_SPEED = 100
DEFAULT_SPEED = 1

# Brain map display limit
MAX_DISPLAY_NEURONS = 100

# Weights file
WEIGHTS_FILE = "brain_weights.npz"

# ============== COLORS ==============
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
COLOR_LOG_BG = (20, 20, 25)
COLOR_TEXT = (200, 200, 200)
COLOR_TURBO = (255, 100, 100)
COLOR_ACCENT = (100, 180, 255)
COLOR_SUCCESS = (100, 255, 150)
COLOR_WARNING = (255, 200, 100)

# Brain map colors
COLOR_INPUT_OFF = (30, 60, 120)
COLOR_INPUT_ON = (80, 180, 255)
COLOR_HIDDEN_OFF = (50, 50, 60)
COLOR_HIDDEN_ON = (255, 255, 255)
COLOR_MOTOR_OFF = (100, 30, 30)
COLOR_MOTOR_ON = (255, 80, 80)

COLOR_ENERGY_BG = (60, 20, 20)
COLOR_ENERGY_FG = (50, 200, 50)


class RheoApp:
    """Main application class with multi-state menu system."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, SIM_HEIGHT))
        pygame.display.set_caption("RHEO SNN - Brain & Survival Lab")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.SysFont("Consolas", 14)
        self.font_bold = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 12)
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)

        # Configuration (user-adjustable)
        self.config = DEFAULT_CONFIG.copy()
        
        # State machine: MENU, SETTINGS, SIMULATION
        self.state = "MENU"
        self.running = True
        
        # Menu state
        self.selected_map = 'labyrinth'
        self.available_maps = []
        
        # Initialize with default config
        self._initialize_simulation()
        
        # Logs
        self.logs = []
        
        # Simulation state
        self.epoch = 1
        self.epoch_start_time = 0
        self.epoch_duration = 120
        
        # Speed control
        self.simulation_speed = DEFAULT_SPEED
        self.turbo_mode = False
        
        # Performance metrics
        self.steps_this_second = 0
        self.sps = 0
        self.last_sps_update = time.time()
        self.total_steps = 0
        
        # Stats
        self.wall_hits = 0
        self.energy_collected = 0
        self.best_goal_time = float('inf')
        self.goals_reached = 0
        
        # Performance Monitor
        self.monitor = Monitor(record_interval=10)
        
        # UI state
        self.hidden_page = 0  # For paginated hidden neurons
        self.current_tooltip = None  # Current tooltip to display
        
        # Tooltip definitions (English)
        self.tooltips = {
            # HUD tooltips
            'da': "DA (Dopamine): Reward signal. Positive = goal/food found. Negative = penalty received.",
            'ach': "ACh (Acetylcholine): Attention level. Increases when approaching walls.",
            'panic': "Panic: Emergency level. Triggered when very close to walls.",
            '5ht': "5HT (Serotonin): Stress level. Spikes when epoch fails (timeout).",
            'explore': "Explore: Noise multiplier. Increases after failed epoch to try new paths.",
            'fire': "Fire: Hidden layer average activation percentage.",
            'brain_e': "Brain Energy: Average metabolic energy of hidden neurons.",
            'energy': "Agent Energy: Remaining energy. Enters sleep mode if depleted.",
            'epoch': "Epoch: Current trial number. Increases when goal reached or time runs out.",
            'time': "Time: Elapsed time this epoch. Penalty applied when time runs out.",
            'best': "Best: Shortest time to reach goal (seconds).",
            'speed': "Speed: Simulation speed multiplier. Change with +/- keys.",
            'sps': "SPS (Steps/Second): Number of simulation steps processed per second.",
            'map': "Map: Current environment layout.",
            'neural_map': "Neural Map: Live neuron activity. Blue=Input, White=Hidden, Red=Motor.",
            'goal': "Gold Target: Reaching this gives a big reward (+500)!",
            'food': "Food: Restores energy (+15 reward).",
            'wall': "Wall: Collision penalty (-5).",
            
            # Analysis screen - Layer Statistics
            'layer_stats': "Layer Statistics: Overview of each neural layer's current state and parameters.",
            'input_layer': "Input Layer: Receives sensor data. High recovery rate - sensors don't tire easily.",
            'hidden_layer': "Hidden Layer: Internal processing with recurrent connections. Moderate fatigue.",
            'output_layer': "Output Layer: Generates motor commands. High energy cost per spike.",
            'neurons': "Neurons: Number of neurons in this layer.",
            'avg_energy': "Avg Energy: Mean metabolic energy. Low = fatigued neurons, fewer spikes.",
            'avg_threshold': "Avg Threshold: Mean firing threshold. Higher = harder to activate.",
            'energy_cost': "Energy Cost: Energy consumed per spike. Higher = faster fatigue.",
            'recovery_rate': "Recovery Rate: Energy recovered per timestep. Higher = faster recovery.",
            
            # Analysis screen - Synapse Statistics
            'synapse_stats': "Synapse Statistics: Weight matrix properties for each connection type.",
            'w_in_hidden': "Input→Hidden: Feedforward connections carrying sensor signals to hidden layer.",
            'w_hidden_hidden': "Hidden→Hidden: Recurrent connections for temporal processing and memory.",
            'w_hidden_out': "Hidden→Output: LEARNABLE connections modified by R-STDP reward learning.",
            'shape': "Shape: Matrix dimensions (post, pre). Total connections = post × pre.",
            'mean_std': "Mean/Std: Average weight and spread. Mean near 0 = balanced. High std = diverse.",
            'range': "Range: [min, max] weight values. Clamped to prevent runaway plasticity.",
            'abs_mean': "Abs Mean: Average magnitude. Increasing = stronger connections forming.",
            
            # Analysis screen - Performance Summary
            'perf_summary': "Performance Summary: Overall session statistics and learning progress.",
            'total_steps': "Total Steps: Cumulative simulation steps executed across all epochs.",
            'epochs_completed': "Epochs: Number of completed trials (success or timeout).",
            'goals_reached': "Goals Reached: Times agent successfully reached the gold target.",
            'wall_hits': "Wall Hits: Total collision penalties received. Lower = better navigation.",
            'food_collected': "Food Collected: Energy pickups gathered. Shows exploration efficiency.",
            'learning': "Learning: Time improvement between first and latest goal. Shows if R-STDP is working!",
        }
        
        # Tooltip hover areas (will be populated during draw)
        self.tooltip_areas = {}
        
    def _initialize_simulation(self):
        """Initialize or reinitialize simulation components with current config."""
        num_sensors = self.config['num_sensors']
        num_hidden = self.config['num_hidden']
        sensor_range = self.config['sensor_range']
        total_neurons = num_sensors + num_hidden + NUM_MOTORS
        
        # Create environment first (needs screen)
        self.env = Environment(
            self.screen, 
            width=SIM_WIDTH, 
            height=SIM_HEIGHT, 
            num_sensors=num_sensors,
            sensor_range=sensor_range
        )
        self.available_maps = self.env.get_available_maps()
        
        # Create brain with NEW layer-based architecture
        # Network(num_inputs, num_hidden, num_outputs, dt)
        self.brain = Network(num_sensors, num_hidden, NUM_MOTORS, DT)
        
        # Encoder/Decoder
        self.encoder = Encoder(0.0, 1.0, 100.0, DT)
        self.decoder = Decoder(NUM_MOTORS, 5.0, DT)
        
        # Precompute neuron positions
        self._compute_neuron_positions()
        
        print(f"Simulation initialized: {total_neurons} neurons, {num_sensors} sensors")

    def _compute_neuron_positions(self):
        """Precompute fixed positions for neurons in layered layout."""
        num_sensors = self.config['num_sensors']
        num_hidden = self.config['num_hidden']
        total = num_sensors + num_hidden + NUM_MOTORS
        
        self.neuron_pos = []
        self.neuron_types = []  # 0=input, 1=hidden, 2=motor
        
        base_x = SIM_WIDTH + 30
        base_y = 420
        
        # Input layer (left column)
        input_x = base_x + 15
        for i in range(min(num_sensors, 15)):  # Max 15 inputs displayed
            y = base_y + i * 14
            self.neuron_pos.append((input_x, y))
            self.neuron_types.append(0)
        
        # Hidden layer (center grid, paginated if > 50)
        hidden_x = base_x + 60
        display_hidden = min(num_hidden, 60)  # Max 60 hidden displayed
        hidden_cols = 10
        for i in range(display_hidden):
            col = i % hidden_cols
            row = i // hidden_cols
            x = hidden_x + col * 15
            y = base_y + row * 15
            self.neuron_pos.append((x, y))
            self.neuron_types.append(1)
        
        # Motor layer (right column)
        motor_x = base_x + 230
        motor_y = base_y + 50
        for i in range(NUM_MOTORS):
            y = motor_y + i * 35
            self.neuron_pos.append((motor_x, y))
            self.neuron_types.append(2)
        
        self.neuron_radius = 5

    def add_log(self, message):
        """Add a timestamped log message."""
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 6:
            self.logs.pop(0)

    def register_tooltip_area(self, key: str, rect: pygame.Rect):
        """Register a rectangular area for tooltip display."""
        self.tooltip_areas[key] = rect

    def check_tooltips(self, mx: int, my: int):
        """Check if mouse is over any tooltip area and set current tooltip."""
        self.current_tooltip = None
        for key, rect in self.tooltip_areas.items():
            if rect.collidepoint(mx, my):
                if key in self.tooltips:
                    self.current_tooltip = self.tooltips[key]
                break

    def draw_tooltip(self, mx: int, my: int):
        """Draw the current tooltip near mouse position."""
        if not self.current_tooltip:
            return
        
        # Create tooltip surface
        padding = 8
        max_width = 350
        
        # Word wrap if needed
        words = self.current_tooltip.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            test_surface = self.font_small.render(test_line, True, WHITE)
            if test_surface.get_width() > max_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        # Calculate tooltip size
        line_height = 16
        tooltip_width = max(self.font_small.render(line, True, WHITE).get_width() for line in lines) + padding * 2
        tooltip_height = len(lines) * line_height + padding * 2
        
        # Position tooltip (avoid going off screen)
        tooltip_x = min(mx + 15, WINDOW_WIDTH - tooltip_width - 10)
        tooltip_y = max(my - tooltip_height - 5, 10)
        
        # Draw background with border
        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, (30, 30, 40), tooltip_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_ACCENT, tooltip_rect, 1, border_radius=6)
        
        # Draw text
        for i, line in enumerate(lines):
            line_surface = self.font_small.render(line, True, (220, 220, 220))
            self.screen.blit(line_surface, (tooltip_x + padding, tooltip_y + padding + i * line_height))

    # ==================== MENU SCREENS ====================

    def draw_menu(self, mx, my):
        """Draw the main menu."""
        self.screen.fill((25, 25, 35))
        
        # Title
        title = self.font_title.render("RHEO SNN", True, COLOR_ACCENT)
        subtitle = self.font.render("Brain & Survival Lab", True, COLOR_TEXT)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 80))
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 120))
        
        # Map selection
        map_label = self.font_bold.render("SELECT MAP:", True, WHITE)
        self.screen.blit(map_label, (WINDOW_WIDTH // 2 - 150, 180))
        
        y_offset = 210
        self.map_buttons = []
        for map_id, map_name in self.available_maps:
            btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 120, y_offset, 240, 35)
            self.map_buttons.append((btn_rect, map_id))
            
            is_selected = map_id == self.selected_map
            is_hover = btn_rect.collidepoint(mx, my)
            
            if is_selected:
                color = COLOR_SUCCESS
            elif is_hover:
                color = (70, 130, 230)
            else:
                color = (50, 50, 70)
            
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=6)
            pygame.draw.rect(self.screen, GRAY, btn_rect, 1, border_radius=6)
            
            txt = self.font.render(map_name, True, WHITE)
            self.screen.blit(txt, (btn_rect.centerx - txt.get_width() // 2, btn_rect.centery - 7))
            y_offset += 45
        
        # Buttons row 1: SETTINGS, START, LOAD
        btn_y = 440
        
        # Settings button
        self.settings_btn = pygame.Rect(WINDOW_WIDTH // 2 - 200, btn_y, 120, 36)
        btn_color = (70, 70, 100) if self.settings_btn.collidepoint(mx, my) else (50, 50, 70)
        pygame.draw.rect(self.screen, btn_color, self.settings_btn, border_radius=8)
        pygame.draw.rect(self.screen, GRAY, self.settings_btn, 1, border_radius=8)
        txt = self.font_bold.render("SETTINGS", True, WHITE)
        self.screen.blit(txt, (self.settings_btn.centerx - txt.get_width() // 2, self.settings_btn.centery - 8))
        
        # Start button (center, larger)
        self.start_btn = pygame.Rect(WINDOW_WIDTH // 2 - 60, btn_y, 120, 36)
        btn_color = (70, 180, 100) if self.start_btn.collidepoint(mx, my) else (50, 150, 80)
        pygame.draw.rect(self.screen, btn_color, self.start_btn, border_radius=8)
        txt = self.font_bold.render("START", True, WHITE)
        self.screen.blit(txt, (self.start_btn.centerx - txt.get_width() // 2, self.start_btn.centery - 8))
        
        # Load brain button
        self.load_btn = pygame.Rect(WINDOW_WIDTH // 2 + 80, btn_y, 120, 36)
        has_save = (get_brain_weights_path() / "brain_weights.npz").exists()
        btn_color = (100, 100, 50) if has_save and self.load_btn.collidepoint(mx, my) else (70, 70, 40) if has_save else (40, 40, 40)
        pygame.draw.rect(self.screen, btn_color, self.load_btn, border_radius=8)
        pygame.draw.rect(self.screen, GRAY, self.load_btn, 1, border_radius=8)
        txt = self.font_bold.render("LOAD", True, WHITE if has_save else GRAY)
        self.screen.blit(txt, (self.load_btn.centerx - txt.get_width() // 2, self.load_btn.centery - 8))
        
        # Buttons row 2: ANALYSIS, MAP EDITOR
        btn_y2 = 485
        
        # Analysis button
        self.analysis_btn = pygame.Rect(WINDOW_WIDTH // 2 - 145, btn_y2, 130, 36)
        btn_color = (100, 70, 130) if self.analysis_btn.collidepoint(mx, my) else (70, 50, 100)
        pygame.draw.rect(self.screen, btn_color, self.analysis_btn, border_radius=8)
        txt = self.font_bold.render("ANALYSIS", True, WHITE)
        self.screen.blit(txt, (self.analysis_btn.centerx - txt.get_width() // 2, self.analysis_btn.centery - 8))
        
        # Map Editor button
        self.editor_btn = pygame.Rect(WINDOW_WIDTH // 2 + 15, btn_y2, 130, 36)
        btn_color = (130, 100, 70) if self.editor_btn.collidepoint(mx, my) else (100, 70, 50)
        pygame.draw.rect(self.screen, btn_color, self.editor_btn, border_radius=8)
        txt = self.font_bold.render("MAP EDITOR", True, WHITE)
        self.screen.blit(txt, (self.editor_btn.centerx - txt.get_width() // 2, self.editor_btn.centery - 8))
        
        # Buttons row 3: EXPERIMENTS
        btn_y3 = 530
        
        # View Experiments button
        self.experiments_btn = pygame.Rect(WINDOW_WIDTH // 2 - 80, btn_y3, 160, 36)
        has_experiments = list(get_experiments_path().glob("*.json"))
        btn_color = (70, 120, 130) if has_experiments and self.experiments_btn.collidepoint(mx, my) else (50, 90, 100) if has_experiments else (40, 50, 55)
        pygame.draw.rect(self.screen, btn_color, self.experiments_btn, border_radius=8)
        txt = self.font_bold.render("EXPERIMENTS", True, WHITE if has_experiments else GRAY)
        self.screen.blit(txt, (self.experiments_btn.centerx - txt.get_width() // 2, self.experiments_btn.centery - 8))
        
        # Current config display
        cfg_y = 580
        cfg_text = f"Sensors: {self.config['num_sensors']} | Range: {self.config['sensor_range']} | Hidden: {self.config['num_hidden']}"
        cfg_surface = self.font_small.render(cfg_text, True, GRAY)
        self.screen.blit(cfg_surface, (WINDOW_WIDTH // 2 - cfg_surface.get_width() // 2, cfg_y))
        
        # Best time
        if self.best_goal_time < float('inf'):
            best_txt = f"Best Goal Time: {self.best_goal_time:.1f}s"
        else:
            best_txt = "Best Goal Time: --"
        best_surface = self.font_small.render(best_txt, True, COLOR_WARNING)
        self.screen.blit(best_surface, (WINDOW_WIDTH // 2 - best_surface.get_width() // 2, cfg_y + 20))
        
        # Controls hint
        hint = self.font_small.render("Press ESC during simulation to return to menu", True, (80, 80, 80))
        self.screen.blit(hint, (WINDOW_WIDTH // 2 - hint.get_width() // 2, SIM_HEIGHT - 40))

    def draw_settings(self, mx, my):
        """Draw the settings screen."""
        self.screen.fill((25, 25, 35))
        
        title = self.font_title.render("SETTINGS", True, COLOR_ACCENT)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 60))
        
        # Sliders
        settings = [
            ('num_sensors', 'Sensors (Vision Count)', 4, 20, 2),
            ('sensor_range', 'Sensor Range (px)', 100, 300, 20),
            ('num_hidden', 'Hidden Neurons (Brain Size)', 20, 200, 10),
        ]
        
        self.setting_sliders = []
        y = 150
        
        for key, label, min_val, max_val, step in settings:
            current_val = self.config[key]
            
            # Label
            lbl = self.font_bold.render(label, True, WHITE)
            self.screen.blit(lbl, (WINDOW_WIDTH // 2 - 200, y))
            
            # Slider bar
            slider_rect = pygame.Rect(WINDOW_WIDTH // 2 - 200, y + 30, 300, 20)
            pygame.draw.rect(self.screen, (50, 50, 60), slider_rect, border_radius=4)
            
            # Filled portion
            pct = (current_val - min_val) / (max_val - min_val)
            filled_width = int(slider_rect.width * pct)
            pygame.draw.rect(self.screen, COLOR_ACCENT, 
                           (slider_rect.x, slider_rect.y, filled_width, slider_rect.height), 
                           border_radius=4)
            
            # Value display
            val_txt = self.font.render(str(current_val), True, WHITE)
            self.screen.blit(val_txt, (slider_rect.right + 15, y + 32))
            
            # +/- buttons
            minus_btn = pygame.Rect(slider_rect.right + 60, y + 28, 30, 25)
            plus_btn = pygame.Rect(slider_rect.right + 95, y + 28, 30, 25)
            
            pygame.draw.rect(self.screen, (80, 50, 50), minus_btn, border_radius=4)
            pygame.draw.rect(self.screen, (50, 80, 50), plus_btn, border_radius=4)
            
            minus_txt = self.font_bold.render("-", True, WHITE)
            plus_txt = self.font_bold.render("+", True, WHITE)
            self.screen.blit(minus_txt, (minus_btn.centerx - 4, minus_btn.centery - 8))
            self.screen.blit(plus_txt, (plus_btn.centerx - 4, plus_btn.centery - 8))
            
            self.setting_sliders.append((key, minus_btn, plus_btn, min_val, max_val, step))
            y += 90
        
        # Back button
        self.back_btn = pygame.Rect(WINDOW_WIDTH // 2 - 80, y + 30, 160, 45)
        btn_color = (70, 130, 230) if self.back_btn.collidepoint(mx, my) else (50, 100, 200)
        pygame.draw.rect(self.screen, btn_color, self.back_btn, border_radius=8)
        txt = self.font_bold.render("BACK", True, WHITE)
        self.screen.blit(txt, (self.back_btn.centerx - txt.get_width() // 2, self.back_btn.centery - 8))
        
        # Warning
        warn = self.font_small.render("Note: Changing settings will reset the brain!", True, COLOR_WARNING)
        self.screen.blit(warn, (WINDOW_WIDTH // 2 - warn.get_width() // 2, y + 100))

    def draw_analysis(self, mx, my):
        """Draw the network analysis screen with detailed statistics."""
        self.screen.fill((20, 20, 30))
        
        # Title
        title = self.font_title.render("NETWORK ANALYSIS", True, (150, 100, 255))
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 30))
        
        x_left = 80
        x_right = WINDOW_WIDTH // 2 + 40
        y = 90
        line_h = 24
        
        # ===== LEFT COLUMN: Layer Statistics =====
        section_title = self.font_bold.render("LAYER STATISTICS", True, COLOR_ACCENT)
        self.screen.blit(section_title, (x_left, y))
        self.register_tooltip_area('layer_stats', pygame.Rect(x_left, y, 200, 20))
        y += line_h + 5
        
        layers_info = [
            ("Input Layer", self.brain.in_layer, COLOR_INPUT_ON, 'input_layer'),
            ("Hidden Layer", self.brain.hidden_layer, COLOR_HIDDEN_ON, 'hidden_layer'),
            ("Output Layer", self.brain.out_layer, COLOR_MOTOR_ON, 'output_layer'),
        ]
        
        for name, layer, color, tooltip_key in layers_info:
            # Layer name
            name_txt = self.font_bold.render(name, True, color)
            self.screen.blit(name_txt, (x_left, y))
            self.register_tooltip_area(tooltip_key, pygame.Rect(x_left, y, 150, 18))
            y += line_h
            
            # Stats
            avg_energy = np.mean(layer.energies)
            avg_threshold = np.mean(layer.thresholds)
            
            stats = [
                (f"  Neurons: {layer.N}", 'neurons'),
                (f"  Avg Energy: {avg_energy:.1f} / {layer.energy_max}", 'avg_energy'),
                (f"  Avg Threshold: {avg_threshold:.2f}", 'avg_threshold'),
                (f"  Energy Cost: {layer.energy_cost}", 'energy_cost'),
                (f"  Recovery Rate: {layer.recovery_rate}", 'recovery_rate'),
            ]
            
            for stat_text, stat_key in stats:
                stat_txt = self.font_small.render(stat_text, True, COLOR_TEXT)
                self.screen.blit(stat_txt, (x_left, y))
                self.register_tooltip_area(f"{stat_key}_{tooltip_key}", pygame.Rect(x_left, y, 220, 14))
                y += line_h - 6
            
            y += 10
        
        # ===== RIGHT COLUMN: Synapse Statistics =====
        y = 90
        section_title = self.font_bold.render("SYNAPSE STATISTICS", True, COLOR_WARNING)
        self.screen.blit(section_title, (x_right, y))
        self.register_tooltip_area('synapse_stats', pygame.Rect(x_right, y, 200, 20))
        y += line_h + 5
        
        synapses_info = [
            ("Input → Hidden", self.brain.w_in_hidden, 'w_in_hidden'),
            ("Hidden → Hidden (Recurrent)", self.brain.w_hidden_hidden, 'w_hidden_hidden'),
            ("Hidden → Output (Learnable)", self.brain.w_hidden_out, 'w_hidden_out'),
        ]
        
        for name, weights, tooltip_key in synapses_info:
            name_txt = self.font_bold.render(name, True, WHITE)
            self.screen.blit(name_txt, (x_right, y))
            self.register_tooltip_area(tooltip_key, pygame.Rect(x_right, y, 280, 18))
            y += line_h
            
            w_mean = np.mean(weights)
            w_std = np.std(weights)
            w_max = np.max(weights)
            w_min = np.min(weights)
            w_abs_mean = np.mean(np.abs(weights))
            
            stats = [
                (f"  Shape: {weights.shape}", 'shape'),
                (f"  Mean: {w_mean:.3f}  Std: {w_std:.3f}", 'mean_std'),
                (f"  Range: [{w_min:.2f}, {w_max:.2f}]", 'range'),
                (f"  Abs Mean: {w_abs_mean:.3f}", 'abs_mean'),
            ]
            
            for stat_text, stat_key in stats:
                stat_txt = self.font_small.render(stat_text, True, COLOR_TEXT)
                self.screen.blit(stat_txt, (x_right, y))
                self.register_tooltip_area(stat_key, pygame.Rect(x_right, y, 250, 14))
                y += line_h - 6
            
            y += 10
        
        # ===== BOTTOM: Performance Summary =====
        y = 480
        section_title = self.font_bold.render("PERFORMANCE SUMMARY", True, COLOR_SUCCESS)
        self.screen.blit(section_title, (x_left, y))
        self.register_tooltip_area('perf_summary', pygame.Rect(x_left, y, 220, 20))
        y += line_h + 5
        
        perf_stats = [
            (f"Total Steps: {self.total_steps:,}", 'total_steps'),
            (f"Epochs Completed: {self.epoch}", 'epochs_completed'),
            (f"Goals Reached: {self.goals_reached}", 'goals_reached'),
            (f"Wall Hits: {self.wall_hits}", 'wall_hits'),
            (f"Food Collected: {self.energy_collected}", 'food_collected'),
            (f"Best Goal Time: {self.best_goal_time:.1f}s" if self.best_goal_time < float('inf') else "Best Goal Time: --", 'best'),
        ]
        
        for i, (stat_text, stat_key) in enumerate(perf_stats):
            col = x_left if i < 3 else x_right
            row_y = y + (i % 3) * (line_h - 4)
            stat_txt = self.font.render(stat_text, True, COLOR_TEXT)
            self.screen.blit(stat_txt, (col, row_y))
            self.register_tooltip_area(stat_key, pygame.Rect(col, row_y, 200, 18))
        
        # Learning curve indicator
        y = 560
        if len(self.monitor.history["success_epochs"]) > 1:
            first_time = self.monitor.history["success_epochs"][0]["time"]
            last_time = self.monitor.history["success_epochs"][-1]["time"]
            improvement = first_time - last_time
            if improvement > 0:
                learning_txt = f"Learning: -{improvement:.1f}s improvement from first goal!"
                learning_color = COLOR_SUCCESS
            else:
                learning_txt = "Learning: No improvement yet..."
                learning_color = COLOR_WARNING
        else:
            learning_txt = "Learning: Not enough data (reach goal 2+ times)"
            learning_color = GRAY
        
        learning_surface = self.font_bold.render(learning_txt, True, learning_color)
        self.screen.blit(learning_surface, (x_left, y))
        self.register_tooltip_area('learning', pygame.Rect(x_left, y, 450, 20))
        
        # Back button
        self.back_btn = pygame.Rect(WINDOW_WIDTH // 2 - 80, 620, 160, 45)
        btn_color = (70, 130, 230) if self.back_btn.collidepoint(mx, my) else (50, 100, 200)
        pygame.draw.rect(self.screen, btn_color, self.back_btn, border_radius=8)
        txt = self.font_bold.render("BACK", True, WHITE)
        self.screen.blit(txt, (self.back_btn.centerx - txt.get_width() // 2, self.back_btn.centery - 8))
        
        # Check and draw tooltips
        self.check_tooltips(mx, my)
        self.draw_tooltip(mx, my)

    # ==================== SIMULATION RENDERING ====================

    def draw_brain_map(self, spikes):
        """Visualize neural network with display limits."""
        # Clear area
        brain_area = pygame.Rect(SIM_WIDTH + 15, 400, LOG_PANEL_WIDTH - 25, 290)
        pygame.draw.rect(self.screen, COLOR_LOG_BG, brain_area)
        
        # Header
        header = self.font_bold.render("NEURAL MAP", True, COLOR_SUCCESS)
        self.screen.blit(header, (SIM_WIDTH + 20, 405))
        
        # Layer labels
        self.screen.blit(self.font_small.render("IN", True, COLOR_INPUT_ON), (SIM_WIDTH + 25, 600))
        self.screen.blit(self.font_small.render("HIDDEN", True, COLOR_HIDDEN_ON), (SIM_WIDTH + 100, 600))
        self.screen.blit(self.font_small.render("OUT", True, COLOR_MOTOR_ON), (SIM_WIDTH + 230, 600))
        
        # Draw synapses (behind neurons)
        self._draw_synapses(spikes)
        
        # Draw neurons (limited to positions computed)
        num_display = min(len(self.neuron_pos), MAX_DISPLAY_NEURONS)
        for i in range(num_display):
            x, y = self.neuron_pos[i]
            ntype = self.neuron_types[i]
            is_spiking = spikes[i] if i < len(spikes) else False
            
            if ntype == 0:
                color = COLOR_INPUT_ON if is_spiking else COLOR_INPUT_OFF
            elif ntype == 1:
                color = COLOR_HIDDEN_ON if is_spiking else COLOR_HIDDEN_OFF
            else:
                color = COLOR_MOTOR_ON if is_spiking else COLOR_MOTOR_OFF
            
            pygame.draw.circle(self.screen, color, (x, y), self.neuron_radius)
            if is_spiking:
                pygame.draw.circle(self.screen, (255, 255, 200), (x, y), self.neuron_radius + 2, 1)

    def _draw_synapses(self, spikes):
        """Draw synaptic connections from spiking neurons."""
        spiking_indices = np.where(spikes)[0][:8]  # Max 8
        
        for pre_idx in spiking_indices:
            if pre_idx >= len(self.neuron_pos):
                continue
            synapses = self.brain.get_top_synapses(pre_idx, top_k=3)
            pre_pos = self.neuron_pos[pre_idx]
            
            for post_idx, weight in synapses:
                if post_idx < len(self.neuron_pos):
                    post_pos = self.neuron_pos[post_idx]
                    intensity = min(255, int(80 + abs(weight) * 6))
                    color = (50, intensity, 50) if weight > 0 else (intensity, 50, 50)
                    pygame.draw.aaline(self.screen, color, pre_pos, post_pos)

    def draw_speed_indicator(self):
        """Draw speed and SPS."""
        x, y = SIM_WIDTH + 20, 195
        
        if self.turbo_mode:
            txt = self.font_bold.render("TURBO", True, COLOR_TURBO)
        else:
            txt = self.font_bold.render(f"Speed: {self.simulation_speed}x", True, COLOR_ACCENT)
        self.screen.blit(txt, (x, y))
        
        sps_color = COLOR_TURBO if self.sps > 500 else COLOR_SUCCESS
        sps_txt = self.font.render(f"SPS: {self.sps:,}", True, sps_color)
        self.screen.blit(sps_txt, (x + 130, y + 2))
        
        hint = self.font_small.render("[+/-] Speed  [0] Turbo  [S] Save", True, (80, 80, 80))
        self.screen.blit(hint, (x, y + 22))

    def draw_hud(self, elapsed_time):
        """Draw HUD with map name, stats, etc."""
        x, y = SIM_WIDTH + 20, 250
        line_h = 18
        
        # Map name
        map_txt = self.font_bold.render(f"Map: {self.env.current_map_name}", True, COLOR_WARNING)
        self.screen.blit(map_txt, (x, y))
        
        # Epoch and timer
        epoch_txt = self.font.render(f"Epoch: {self.epoch}", True, COLOR_TEXT)
        self.screen.blit(epoch_txt, (x + 180, y + 2))
        
        mins, secs = int(elapsed_time // 60), int(elapsed_time % 60)
        time_txt = self.font.render(f"Time: {mins:02d}:{secs:02d}", True, COLOR_TEXT)
        self.screen.blit(time_txt, (x, y + line_h))
        
        # Best time
        if self.best_goal_time < float('inf'):
            best_txt = self.font.render(f"Best: {self.best_goal_time:.1f}s", True, COLOR_SUCCESS)
        else:
            best_txt = self.font.render("Best: --", True, GRAY)
        self.screen.blit(best_txt, (x + 130, y + line_h))
        self.register_tooltip_area('best', pygame.Rect(x + 130, y + line_h, 100, 16))
        
        # Energy bar
        bar_y = y + line_h * 2 + 5
        energy_pct = self.env.agent_energy / self.env.max_agent_energy
        bar_w = 160
        pygame.draw.rect(self.screen, COLOR_ENERGY_BG, (x, bar_y, bar_w, 12))
        pygame.draw.rect(self.screen, COLOR_ENERGY_FG, (x, bar_y, int(bar_w * energy_pct), 12))
        pygame.draw.rect(self.screen, GRAY, (x, bar_y, bar_w, 12), 1)
        self.register_tooltip_area('energy', pygame.Rect(x, bar_y, bar_w + 40, 14))
        
        nrg_txt = self.font_small.render(f"{int(self.env.agent_energy)}", True, WHITE)
        self.screen.blit(nrg_txt, (x + bar_w + 8, bar_y - 1))
        
        # Neuromodulators Row 1: DA, ACh, Panic
        nm_y = bar_y + 20
        da_color = COLOR_SUCCESS if self.brain.dopamine > 10 else COLOR_TURBO if self.brain.dopamine < -10 else COLOR_WARNING
        da_txt = self.font_small.render(f"DA:{self.brain.cumulative_reward:+.1f}", True, da_color)
        ach_txt = self.font_small.render(f"ACh:{self.brain.acetylcholine:.1f}", True, COLOR_ACCENT)
        panic_txt = self.font_small.render(f"Panic:{self.brain.panic_level:.1f}", True, 
                                          COLOR_TURBO if self.brain.panic_level > 0.5 else GRAY)
        self.screen.blit(da_txt, (x, nm_y))
        self.screen.blit(ach_txt, (x + 85, nm_y))
        self.screen.blit(panic_txt, (x + 160, nm_y))
        
        # Register tooltip areas for neuromodulators row 1
        self.register_tooltip_area('da', pygame.Rect(x, nm_y, 80, 14))
        self.register_tooltip_area('ach', pygame.Rect(x + 85, nm_y, 70, 14))
        self.register_tooltip_area('panic', pygame.Rect(x + 160, nm_y, 80, 14))
        
        # Neuromodulators Row 2: 5HT (Serotonin), Exploration Noise
        nm_y2 = nm_y + 16
        sero_color = COLOR_TURBO if self.brain.serotonin > 2.0 else COLOR_WARNING if self.brain.serotonin > 0.5 else GRAY
        sero_txt = self.font_small.render(f"5HT:{self.brain.serotonin:.1f}", True, sero_color)
        
        noise_color = COLOR_SUCCESS if self.brain.exploration_noise > 1.5 else GRAY
        noise_txt = self.font_small.render(f"Explore:{self.brain.exploration_noise:.1f}x", True, noise_color)
        
        self.screen.blit(sero_txt, (x, nm_y2))
        self.screen.blit(noise_txt, (x + 85, nm_y2))
        
        # Register tooltip areas for neuromodulators row 2
        self.register_tooltip_area('5ht', pygame.Rect(x, nm_y2, 80, 14))
        self.register_tooltip_area('explore', pygame.Rect(x + 85, nm_y2, 100, 14))
        
        # Layer Stats (from Monitor)
        stats_y = nm_y2 + 18
        firing_rate = self.monitor.get_avg_firing_rate("hidden") * 100
        hidden_energy = self.monitor.get_avg_energy("hidden")
        
        fr_color = COLOR_TURBO if firing_rate == 0 else COLOR_SUCCESS
        fr_txt = self.font_small.render(f"Fire:{firing_rate:.0f}%", True, fr_color)
        nrg_layer_txt = self.font_small.render(f"BrainE:{hidden_energy:.0f}", True, COLOR_ACCENT)
        
        self.screen.blit(fr_txt, (x, stats_y))
        self.screen.blit(nrg_layer_txt, (x + 80, stats_y))
        
        # Register tooltip areas for layer stats
        self.register_tooltip_area('fire', pygame.Rect(x, stats_y, 75, 14))
        self.register_tooltip_area('brain_e', pygame.Rect(x + 80, stats_y, 90, 14))

    def draw_log_panel(self, spikes, elapsed_time):
        """Draw complete log panel."""
        pygame.draw.rect(self.screen, COLOR_LOG_BG, (SIM_WIDTH, 0, LOG_PANEL_WIDTH, SIM_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 70), (SIM_WIDTH, 0), (SIM_WIDTH, SIM_HEIGHT), 2)
        
        # Header
        header = self.font_bold.render("SYSTEM LOG", True, COLOR_ACCENT)
        self.screen.blit(header, (SIM_WIDTH + 20, 15))
        
        # Logs
        for i, log in enumerate(reversed(self.logs)):
            y = 38 + i * 18
            if y < 175:
                log_txt = self.font_small.render(log[:38], True, COLOR_TEXT)
                self.screen.blit(log_txt, (SIM_WIDTH + 20, y))
        
        self.draw_speed_indicator()
        self.draw_hud(elapsed_time)
        self.draw_brain_map(spikes)

    # ==================== SIMULATION STEP ====================

    def simulation_step(self, current_reward):
        """Execute one simulation step."""
        # Get combined sensor state (includes food/goal info)
        sensor_data = self.env.get_state()
        
        # Use RAW wall sensors for reactive inhibition and ACh (more accurate)
        raw_wall_sensors = self.env.sensors  # Pure wall detection
        
        # Reactive inhibition - based on actual wall proximity
        self.brain.apply_reactive_inhibition(raw_wall_sensors, threshold=PANIC_THRESHOLD)
        
        # Encode the combined sensor data (includes attraction to food/goal)
        input_currents = self.encoder.step_current(sensor_data, gain=100.0)
        # Apply exploration noise (boosted after failures)
        noise = np.random.uniform(0.0, 3.0 * self.brain.exploration_noise, size=input_currents.shape)
        input_currents += noise
        
        # Decay exploration noise back to normal over time
        self.brain.decay_exploration_noise()
        
        # Neuromodulators - use raw wall sensors for danger detection
        max_wall_signal = np.max(raw_wall_sensors) if len(raw_wall_sensors) > 0 else 0
        self.brain.acetylcholine = max_wall_signal * 3.0
        
        # Stress based on energy
        stress = 1.0 - (self.env.agent_energy / self.env.max_agent_energy)
        self.brain.serotonin = max(self.brain.serotonin, stress * 1.5)
        
        # Brain step
        all_spikes = self.brain.step(external_inputs=input_currents, reward=current_reward)
        
        output_spikes = all_spikes[-NUM_MOTORS:]
        should_brake = np.max(sensor_data) > 0.85
        motor_commands = self.decoder.step(output_spikes, brake=should_brake)
        
        # Differential
        mid = self.config['num_sensors'] // 2
        left_avg = np.mean(sensor_data[:mid])
        right_avg = np.mean(sensor_data[mid:])
        self.decoder.apply_differential(left_avg, right_avg, 0.2)
        motor_commands = self.decoder.motor_values
        
        # Movement
        l_m = motor_commands[0] * 3.0 + 0.1
        r_m = motor_commands[1] * 3.0 + 0.1
        
        _, reward, done = self.env.step(l_m, r_m)
        return all_spikes, reward, done

    # ==================== EVENT HANDLING ====================

    def handle_events(self):
        """Handle pygame events."""
        mx, my = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Menu clicks
            if self.state == "MENU" and event.type == pygame.MOUSEBUTTONDOWN:
                # Map selection
                for btn_rect, map_id in self.map_buttons:
                    if btn_rect.collidepoint(mx, my):
                        self.selected_map = map_id
                
                # Start button
                if self.start_btn.collidepoint(mx, my):
                    self._initialize_simulation()
                    self.env.set_map(self.selected_map)
                    self.env.reset()
                    self.brain.reset_state()
                    self.state = "SIMULATION"
                    self.epoch_start_time = time.time()
                    self.total_steps = 0
                    self.add_log(f"Started: {self.env.current_map_name}")
                
                # Settings button
                if self.settings_btn.collidepoint(mx, my):
                    self.state = "SETTINGS"
                
                # Analysis button
                if self.analysis_btn.collidepoint(mx, my):
                    self.state = "ANALYSIS"
                
                # Map Editor button
                if self.editor_btn.collidepoint(mx, my):
                    from src.environment.editor import run_editor
                    pygame.quit()  # Close main window
                    run_editor()   # Open editor (has its own pygame.init/quit)
                    # Fully reinitialize after editor closes
                    pygame.init()
                    self.screen = pygame.display.set_mode((WINDOW_WIDTH, SIM_HEIGHT))
                    pygame.display.set_caption("RHEO SNN - Brain & Survival Lab")
                    self.clock = pygame.time.Clock()
                    # Reinitialize fonts
                    self.font = pygame.font.SysFont("Consolas", 14)
                    self.font_bold = pygame.font.SysFont("Consolas", 16, bold=True)
                    self.font_small = pygame.font.SysFont("Consolas", 12)
                    self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
                    # Reinitialize simulation
                    self._initialize_simulation()
                    self.state = "MENU"  # Ensure we're in menu state
                
                # Experiments button
                if self.experiments_btn.collidepoint(mx, my):
                    exp_files = list(get_experiments_path().glob("*.json"))
                    if exp_files:
                        # Open experiments folder in file explorer
                        import subprocess
                        subprocess.Popen(f'explorer "{get_experiments_path()}"')
                        self.add_log("Opened experiments folder")
                
                # Load button
                weights_path = get_brain_weights_path() / "brain_weights.npz"
                if self.load_btn.collidepoint(mx, my) and weights_path.exists():
                    self._initialize_simulation()
                    if self.brain.load_weights("brain_weights.npz"):
                        self.add_log("Brain loaded!")
                    else:
                        self.add_log("Load failed!")
            
            # Settings clicks
            elif self.state == "SETTINGS" and event.type == pygame.MOUSEBUTTONDOWN:
                for key, minus_btn, plus_btn, min_val, max_val, step in self.setting_sliders:
                    if minus_btn.collidepoint(mx, my):
                        self.config[key] = max(min_val, self.config[key] - step)
                    if plus_btn.collidepoint(mx, my):
                        self.config[key] = min(max_val, self.config[key] + step)
                
                if self.back_btn.collidepoint(mx, my):
                    self.state = "MENU"
            
            # Analysis clicks
            elif self.state == "ANALYSIS" and event.type == pygame.MOUSEBUTTONDOWN:
                if self.back_btn.collidepoint(mx, my):
                    self.state = "MENU"
            
            # Simulation keys
            elif self.state == "SIMULATION" and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state = "MENU"
                    self.add_log("Paused - Menu")
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.turbo_mode = False
                    self.simulation_speed = min(self.simulation_speed + 1, MAX_SPEED)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.turbo_mode = False
                    self.simulation_speed = max(self.simulation_speed - 1, MIN_SPEED)
                elif event.key in (pygame.K_0, pygame.K_KP0):
                    self.turbo_mode = not self.turbo_mode
                    self.add_log("TURBO " + ("ON" if self.turbo_mode else "OFF"))
                elif event.key == pygame.K_s:
                    if self.brain.save_weights(WEIGHTS_FILE):
                        self.add_log("Brain saved!")
        
        return mx, my

    def update_sps(self):
        """Update SPS metric."""
        now = time.time()
        if now - self.last_sps_update >= 1.0:
            self.sps = self.steps_this_second
            self.steps_this_second = 0
            self.last_sps_update = now

    # ==================== MAIN LOOP ====================

    def run(self):
        """Main application loop."""
        current_reward = 0.0
        total_neurons = self.config['num_sensors'] + self.config['num_hidden'] + NUM_MOTORS
        last_spikes = np.zeros(total_neurons, dtype=bool)
        elapsed_time = 0.0
        goal_start_time = 0.0

        while self.running:
            mx, my = self.handle_events()
            self.update_sps()

            # ========== MENU STATE ==========
            if self.state == "MENU":
                self.draw_menu(mx, my)
                pygame.display.flip()
                self.clock.tick(60)

            # ========== SETTINGS STATE ==========
            elif self.state == "SETTINGS":
                self.draw_settings(mx, my)
                pygame.display.flip()
                self.clock.tick(60)

            # ========== ANALYSIS STATE ==========
            elif self.state == "ANALYSIS":
                self.draw_analysis(mx, my)
                pygame.display.flip()
                self.clock.tick(60)

            # ========== SIMULATION STATE ==========
            elif self.state == "SIMULATION":
                elapsed_time = time.time() - self.epoch_start_time
                
                # Epoch timeout check (FAILURE - didn't reach goal)
                if elapsed_time >= self.epoch_duration:
                    # Apply failure penalty before reset
                    self.brain.apply_epoch_failure(penalty=-50.0)
                    self.add_log(f"Epoch {self.epoch} TIMEOUT! Penalty applied.")
                    
                    self.epoch += 1
                    self.epoch_start_time = time.time()
                    goal_start_time = time.time()
                    self.env.reset()
                    self.brain.reset_state()
                    
                    # Auto-save every 5 epochs
                    if self.epoch % 5 == 0:
                        self.monitor.save_stats("simulation_stats.json")
                
                # Frame-Skipping Logic Loop
                # Instead of running exactly 'speed' steps per frame, we run as many as possible
                # within a time budget (12ms) to leave room for rendering (4ms) -> 60 FPS
                
                frame_budget = 0.012  # 12ms for logic
                logic_start = time.time()
                
                # Turbo mode has no budget limit per frame (runs until 200 steps or just fast)
                # But to keep UI responsive even in Turbo, we limit batch size per frame
                max_steps_per_frame = 200 if self.turbo_mode else self.simulation_speed
                
                steps_done = 0
                while steps_done < max_steps_per_frame:
                    last_spikes, reward, done = self.simulation_step(current_reward)
                    current_reward = reward
                    self.steps_this_second += 1
                    self.total_steps += 1
                    steps_done += 1
                    
                    # Monitor recording (every 10 steps)
                    if self.monitor.should_record(self.total_steps):
                        layers_dict = {
                            "input": self.brain.in_layer,
                            "hidden": self.brain.hidden_layer,
                            "output": self.brain.out_layer,
                        }
                        self.monitor.record(
                            self.total_steps,
                            layers_dict,
                            reward,
                            self.brain.w_hidden_out,
                            self.brain.panic_level
                        )
                        
                        # Neural silence detection
                        if self.monitor.neural_silence_detected:
                            if self.total_steps % 100 == 0:
                                self.add_log("Alert: Neural Silence!")
                    
                    # Events
                    if reward > 5:
                        self.energy_collected += 1
                        self.monitor.log_food_collected()
                    if reward < -4:
                        self.wall_hits += 1
                        self.monitor.log_wall_hit()
                    
                    if done and reward > 50:
                        goal_time = time.time() - goal_start_time
                        self.monitor.log_success(self.epoch, goal_time)
                        
                        if goal_time < self.best_goal_time:
                            self.best_goal_time = goal_time
                            self.add_log(f"NEW BEST: {goal_time:.1f}s!")
                        else:
                            self.add_log(f"Goal: {goal_time:.1f}s")
                        self.goals_reached += 1
                        self.epoch += 1
                        self.epoch_start_time = time.time()
                        goal_start_time = time.time()
                        self.env.reset()
                        self.brain.reset_state()
                        break
                    
                    # Check time budget only if not in Turbo (Turbo ignores lag to go fast)
                    if not self.turbo_mode and (time.time() - logic_start) > frame_budget:
                        break
                
                # Render
                if not self.turbo_mode:
                    self.env.render()
                    self.draw_log_panel(last_spikes, elapsed_time)
                    # Check and draw tooltips
                    self.check_tooltips(mx, my)
                    self.draw_tooltip(mx, my)
                    pygame.display.flip()
                    self.clock.tick(60)
                else:
                    # Turbo render: only every 200ms
                    current_time = time.time()
                    if current_time - self.last_sps_time > 0.2:
                        self.env.render()
                        self.draw_log_panel(last_spikes, elapsed_time)
                        pygame.display.flip()
                    # No tick limit in turbo
                    self.clock.tick(0)

        # Auto-save stats on quit
        self.monitor.save_stats("simulation_stats.json")
        pygame.quit()


if __name__ == "__main__":
    RheoApp().run()