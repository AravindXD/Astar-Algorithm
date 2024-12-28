import pygame
from pathfinding import AStar

class App:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.cell_size = 20
        self.grid_width = self.width // self.cell_size
        self.grid_height = self.height // self.cell_size
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("A* Pathfinding")
        self.clock = pygame.time.Clock() 
        
        self.start = None
        self.goal = None
        self.obstacles = set()
        self.path = []
        self.visited = []
        self.current_path = []
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (147, 112, 219)
        
    def draw_grid(self):
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.width, y))
            
    def get_cell_position(self, pos):
        x = pos[0] // self.cell_size
        y = pos[1] // self.cell_size
        return (x, y)
        
    def draw_cell(self, pos, color):
        pygame.draw.rect(self.screen, color,
                        (pos[0]*self.cell_size, pos[1]*self.cell_size,
                         self.cell_size, self.cell_size))
        
    def run(self):
        running = True
        drawing_obstacles = False
        animating = False
        animation_index = 0
        path_found = False
        
        while running:
            self.screen.fill(self.WHITE)
            self.draw_grid()
            
            if animating:
                for i in range(min(animation_index + 1, len(self.visited))):
                    self.draw_cell(self.visited[i], self.YELLOW)
                if animation_index < len(self.visited):
                    self.draw_cell(self.visited[animation_index], self.PURPLE)
            
            for obs in self.obstacles:
                self.draw_cell(obs, self.BLACK)
            
            if self.start:
                self.draw_cell(self.start, self.BLUE)
            if self.goal:
                self.draw_cell(self.goal, self.GREEN)
                
            if path_found:
                for i in range(len(self.current_path)-1):
                    start_pos = (self.current_path[i][0]*self.cell_size + self.cell_size//2,
                               self.current_path[i][1]*self.cell_size + self.cell_size//2)
                    end_pos = (self.current_path[i+1][0]*self.cell_size + self.cell_size//2,
                             self.current_path[i+1][1]*self.cell_size + self.cell_size//2)
                    pygame.draw.line(self.screen, self.RED, start_pos, end_pos, 2)
                    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.get_cell_position(pygame.mouse.get_pos())
                    
                    if event.button == 1:
                        if not self.start:
                            self.start = pos
                        elif not self.goal:
                            self.goal = pos
                        else:
                            drawing_obstacles = True
                            
                elif event.type == pygame.MOUSEBUTTONUP:
                    drawing_obstacles = False
                    
                elif event.type == pygame.MOUSEMOTION and drawing_obstacles:
                    pos = self.get_cell_position(pygame.mouse.get_pos())
                    if pos != self.start and pos != self.goal:
                        self.obstacles.add(pos)
                        
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.start and self.goal:
                        astar = AStar(self.grid_width, self.grid_height, self.obstacles)
                        self.path, self.visited = astar.find_path(self.start, self.goal)
                        animation_index = 0
                        animating = True
                    elif event.key == pygame.K_c:
                        self.start = None
                        self.goal = None
                        self.obstacles.clear()
                        self.path = []
                        self.visited = []
                        self.current_path = []
                        animating = False
            
            if animating:
                if animation_index < len(self.visited):
                    animation_index += 1
                    pygame.time.delay(100)
                else:
                    animating = False
                    path_found = True
                    self.current_path = self.path
            
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    app = App()
    try:
        print("🖱️ Click on a cell to set the Source, then the Destination, and finally the Obstacles (click-drag supported).")
        print("🔄 Press 'C' to clear the grid.")
        print("▶️ Press 'Space' to start navigation.")
        print("🛑 Press 'Ctrl+C' in the terminal to terminate.")
        app.run()
    except KeyboardInterrupt:
        print("\n🚪 Terminated by user.")