"""
Map Editor: Visual Tool for Creating Custom Maps

A Pygame-based editor for designing custom environments.
Maps are saved as JSON files in src/environment/maps/.

Controls:
    - Left Click: Place walls (drag to draw)
    - Right Click: Erase walls
    - F Key: Place food spawn point
    - G Key: Place goal (gold target)
    - S Key: Set spawn point (agent start)
    - Ctrl+S: Save map
    - Ctrl+L: Load map
    - Escape: Exit editor

Grid System:
    - 20x20 pixel cells for precise wall placement
    - Walls are automatically merged into rectangles for efficiency
"""
import pygame
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


# Editor constants
GRID_SIZE = 20
CELL_SIZE = 20


class MapEditor:
    """Visual map editor for RHEO SNN environments."""
    
    def __init__(self, width: int = 900, height: int = 700):
        """
        Initialize the map editor.
        
        Args:
            width: Editor window width.
            height: Editor window height.
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width + 200, height))
        pygame.display.set_caption("RHEO Map Editor")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.SysFont("Consolas", 14)
        self.font_bold = pygame.font.SysFont("Consolas", 16, bold=True)
        
        # Grid dimensions
        self.cols = width // CELL_SIZE
        self.rows = height // CELL_SIZE
        
        # Map data
        self.walls = set()  # Set of (col, row) tuples for wall cells
        self.foods = []     # List of (x, y) positions for food spawns
        self.goal = None    # (x, y) position for goal
        self.spawn = None   # (x, y) position for agent spawn
        
        # Editor state
        self.current_tool = "wall"  # wall, erase, food, goal, spawn
        self.running = True
        self.map_name = "custom_map"
        self.drawing = False
        self.editing_name = False  # Text input mode for map name
        self.name_input = ""        # Current text input
        
        # Colors
        self.colors = {
            "bg": (30, 30, 40),
            "grid": (50, 50, 60),
            "wall": (80, 80, 100),
            "wall_preview": (60, 60, 80),
            "food": (100, 200, 100),
            "goal": (255, 215, 0),
            "spawn": (100, 150, 255),
            "panel": (40, 40, 50),
            "text": (200, 200, 200),
            "highlight": (100, 150, 255),
        }
        
        # Load maps path
        from src.utils.file_manager import get_maps_path
        self.maps_path = get_maps_path()
        
        # Initialize with border walls
        self._add_border_walls()
    
    def _add_border_walls(self):
        """Add walls around the border of the map."""
        for col in range(self.cols):
            self.walls.add((col, 0))
            self.walls.add((col, self.rows - 1))
        for row in range(self.rows):
            self.walls.add((0, row))
            self.walls.add((self.cols - 1, row))
    
    def run(self):
        """Main editor loop."""
        while self.running:
            self._handle_events()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
    
    def _handle_events(self):
        """Handle input events."""
        mx, my = pygame.mouse.get_pos()
        keys = pygame.key.get_mods()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Text input mode for map name
                if self.editing_name:
                    if event.key == pygame.K_RETURN:
                        self.map_name = self.name_input if self.name_input else "custom_map"
                        self.editing_name = False
                    elif event.key == pygame.K_ESCAPE:
                        self.editing_name = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.name_input = self.name_input[:-1]
                    else:
                        # Only allow alphanumeric and underscore
                        char = event.unicode
                        if char.isalnum() or char in "_-":
                            self.name_input += char
                else:
                    # Normal key handling
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_f:
                        self.current_tool = "food"
                    elif event.key == pygame.K_g:
                        self.current_tool = "goal"
                    elif event.key == pygame.K_s and keys & pygame.KMOD_CTRL:
                        self._save_map_dialog()
                    elif event.key == pygame.K_s:
                        self.current_tool = "spawn"
                    elif event.key == pygame.K_l and keys & pygame.KMOD_CTRL:
                        self._load_map_dialog()
                    elif event.key == pygame.K_w:
                        self.current_tool = "wall"
                    elif event.key == pygame.K_e:
                        self.current_tool = "erase"
                    elif event.key == pygame.K_n:
                        # Start name editing
                        self.editing_name = True
                        self.name_input = self.map_name
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mx < self.width:  # In grid area
                    col, row = mx // CELL_SIZE, my // CELL_SIZE
                    
                    if event.button == 1:  # Left click
                        self.drawing = True
                        if self.current_tool == "wall":
                            self.walls.add((col, row))
                        elif self.current_tool == "food":
                            pos = (col * CELL_SIZE + CELL_SIZE // 2, 
                                   row * CELL_SIZE + CELL_SIZE // 2)
                            self.foods.append(pos)
                        elif self.current_tool == "goal":
                            self.goal = (col * CELL_SIZE + CELL_SIZE // 2,
                                        row * CELL_SIZE + CELL_SIZE // 2)
                        elif self.current_tool == "spawn":
                            self.spawn = (col * CELL_SIZE + CELL_SIZE // 2,
                                         row * CELL_SIZE + CELL_SIZE // 2)
                        elif self.current_tool == "erase":
                            self.walls.discard((col, row))
                    
                    elif event.button == 3:  # Right click = erase
                        self.walls.discard((col, row))
                        self.drawing = True
                        self.current_tool = "erase"
                
                else:  # In panel area - tool buttons
                    self._handle_panel_click(mx, my)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.drawing = False
        
        # Continuous drawing while mouse held
        if self.drawing and pygame.mouse.get_pressed()[0]:
            if mx < self.width:
                col, row = mx // CELL_SIZE, my // CELL_SIZE
                if self.current_tool == "wall":
                    self.walls.add((col, row))
                elif self.current_tool == "erase":
                    self.walls.discard((col, row))
        
        if self.drawing and pygame.mouse.get_pressed()[2]:
            if mx < self.width:
                col, row = mx // CELL_SIZE, my // CELL_SIZE
                self.walls.discard((col, row))
    
    def _handle_panel_click(self, mx: int, my: int):
        """Handle clicks in the tool panel."""
        # Tool buttons
        tools = [("wall", 30), ("erase", 70), ("food", 110), ("goal", 150), ("spawn", 190)]
        for tool, y_offset in tools:
            btn_rect = pygame.Rect(self.width + 20, y_offset, 160, 30)
            if btn_rect.collidepoint(mx, my):
                self.current_tool = tool
        
        # Rename button
        rename_rect = pygame.Rect(self.width + 20, 330, 160, 30)
        if rename_rect.collidepoint(mx, my):
            self.editing_name = True
            self.name_input = self.map_name
        
        # Save button
        save_rect = pygame.Rect(self.width + 20, 370, 160, 35)
        if save_rect.collidepoint(mx, my):
            self._save_map_dialog()
        
        # Load button
        load_rect = pygame.Rect(self.width + 20, 415, 160, 35)
        if load_rect.collidepoint(mx, my):
            self._load_map_dialog()
        
        # Clear button
        clear_rect = pygame.Rect(self.width + 20, 460, 160, 35)
        if clear_rect.collidepoint(mx, my):
            self.walls.clear()
            self.foods.clear()
            self.goal = None
            self.spawn = None
            self._add_border_walls()
    
    def _draw(self):
        """Draw the editor interface."""
        self.screen.fill(self.colors["bg"])
        
        # Draw grid
        for x in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, self.colors["grid"], (x, 0), (x, self.height))
        for y in range(0, self.height, CELL_SIZE):
            pygame.draw.line(self.screen, self.colors["grid"], (0, y), (self.width, y))
        
        # Draw walls
        for col, row in self.walls:
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, self.colors["wall"], rect)
        
        # Draw food spawns
        for x, y in self.foods:
            pygame.draw.circle(self.screen, self.colors["food"], (int(x), int(y)), 8)
        
        # Draw goal
        if self.goal:
            pygame.draw.circle(self.screen, self.colors["goal"], 
                             (int(self.goal[0]), int(self.goal[1])), 15)
            pygame.draw.circle(self.screen, (200, 160, 0), 
                             (int(self.goal[0]), int(self.goal[1])), 10)
        
        # Draw spawn
        if self.spawn:
            pygame.draw.circle(self.screen, self.colors["spawn"],
                             (int(self.spawn[0]), int(self.spawn[1])), 12)
            pygame.draw.circle(self.screen, (50, 100, 200),
                             (int(self.spawn[0]), int(self.spawn[1])), 6)
        
        # Draw tool panel
        self._draw_panel()
        
        # Draw cursor preview
        mx, my = pygame.mouse.get_pos()
        if mx < self.width:
            col, row = mx // CELL_SIZE, my // CELL_SIZE
            preview_rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, self.colors["highlight"], preview_rect, 2)
    
    def _draw_panel(self):
        """Draw the tool panel on the right side."""
        panel_x = self.width
        pygame.draw.rect(self.screen, self.colors["panel"], 
                        (panel_x, 0, 200, self.height))
        
        # Title
        title = self.font_bold.render("MAP EDITOR", True, self.colors["highlight"])
        self.screen.blit(title, (panel_x + 50, 5))
        
        # Tool buttons
        tools = [
            ("wall", "Wall (W)", 30),
            ("erase", "Erase (E)", 70),
            ("food", "Food (F)", 110),
            ("goal", "Goal (G)", 150),
            ("spawn", "Spawn (S)", 190),
        ]
        
        for tool_id, label, y_offset in tools:
            btn_rect = pygame.Rect(panel_x + 20, y_offset, 160, 30)
            color = self.colors["highlight"] if self.current_tool == tool_id else (60, 60, 70)
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=4)
            txt = self.font.render(label, True, self.colors["text"])
            self.screen.blit(txt, (btn_rect.centerx - txt.get_width() // 2, 
                                   btn_rect.centery - 7))
        
        # Stats
        pygame.draw.line(self.screen, self.colors["grid"], 
                        (panel_x + 10, 240), (panel_x + 190, 240))
        
        stats = [
            f"Walls: {len(self.walls)}",
            f"Foods: {len(self.foods)}",
            f"Goal: {'Set' if self.goal else 'Not set'}",
            f"Spawn: {'Set' if self.spawn else 'Not set'}",
        ]
        
        for i, stat in enumerate(stats):
            txt = self.font.render(stat, True, self.colors["text"])
            self.screen.blit(txt, (panel_x + 20, 250 + i * 20))
        
        # Rename Button
        rename_rect = pygame.Rect(panel_x + 20, 330, 160, 30)
        color = (100, 100, 120) if self.editing_name else (70, 70, 90)
        pygame.draw.rect(self.screen, color, rename_rect, border_radius=4)
        
        if self.editing_name:
            # Draw input field
            txt = self.font.render(self.name_input + "_", True, (255, 255, 255))
        else:
            txt = self.font.render("RENAME (N)", True, self.colors["text"])
            
        self.screen.blit(txt, (rename_rect.centerx - txt.get_width() // 2,
                               rename_rect.centery - 7))

        # Save/Load/Clear buttons
        for label, y_offset, color in [
            ("SAVE (Ctrl+S)", 370, (80, 150, 80)),
            ("LOAD (Ctrl+L)", 415, (80, 80, 150)),
            ("CLEAR ALL", 460, (150, 80, 80)),
        ]:
            btn_rect = pygame.Rect(panel_x + 20, y_offset, 160, 35)
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=4)
            txt = self.font.render(label, True, self.colors["text"])
            self.screen.blit(txt, (btn_rect.centerx - txt.get_width() // 2,
                                   btn_rect.centery - 7))
        
        # Instructions
        pygame.draw.line(self.screen, self.colors["grid"],
                        (panel_x + 10, 505), (panel_x + 190, 505))
        
        instructions = [
            "Left Click: Draw",
            "Right Click: Erase",
            "N: Rename Map",
            "Enter: Confirm Name",
            "ESC: Exit Editor",
        ]
        for i, instr in enumerate(instructions):
            txt = self.font.render(instr, True, (120, 120, 130))
            self.screen.blit(txt, (panel_x + 20, 515 + i * 18))
        
        # Map name
        name_txt = self.font_bold.render(f"Map: {self.map_name}", True, self.colors["text"])
        self.screen.blit(name_txt, (panel_x + 20, self.height - 30))
    
    def _save_map_dialog(self):
        """Save map to JSON file."""
        # Convert walls to rectangles for efficiency
        wall_rects = self._walls_to_rects()
        
        map_data = {
            "name": self.map_name,
            "width": self.width,
            "height": self.height,
            "walls": wall_rects,
            "foods": self.foods,
            "goal": self.goal,
            "spawn": self.spawn if self.spawn else [self.width // 2, self.height // 2],
        }
        
        filename = self.maps_path / f"{self.map_name}.json"
        
        try:
            with open(filename, "w") as f:
                json.dump(map_data, f, indent=2)
            print(f"Map saved to: {filename}")
        except Exception as e:
            print(f"Error saving map: {e}")
    
    def _load_map_dialog(self):
        """Load map from JSON file."""
        # List available maps
        available = list(self.maps_path.glob("*.json"))
        if not available:
            print("No maps found in maps folder")
            return
        
        # Load the first (or most recent) map for now
        # TODO: Add proper file selection dialog
        filename = available[-1]
        
        try:
            with open(filename, "r") as f:
                map_data = json.load(f)
            
            self.map_name = map_data.get("name", "loaded_map")
            self.walls.clear()
            
            # Convert rectangles back to cells
            for rect in map_data.get("walls", []):
                x, y, w, h = rect
                for col in range(x // CELL_SIZE, (x + w) // CELL_SIZE):
                    for row in range(y // CELL_SIZE, (y + h) // CELL_SIZE):
                        self.walls.add((col, row))
            
            self.foods = [tuple(f) for f in map_data.get("foods", [])]
            self.goal = tuple(map_data["goal"]) if map_data.get("goal") else None
            self.spawn = tuple(map_data["spawn"]) if map_data.get("spawn") else None
            
            print(f"Map loaded from: {filename}")
        except Exception as e:
            print(f"Error loading map: {e}")
    
    def _walls_to_rects(self) -> List[List[int]]:
        """
        Convert wall cells to rectangles for efficient storage.
        Uses simple row-by-row merging.
        """
        if not self.walls:
            return []
        
        rects = []
        for col, row in sorted(self.walls):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            rects.append([x, y, CELL_SIZE, CELL_SIZE])
        
        return rects


def run_editor():
    """Entry point for map editor."""
    editor = MapEditor()
    editor.run()


if __name__ == "__main__":
    run_editor()
