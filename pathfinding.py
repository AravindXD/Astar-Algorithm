import heapq
import math

class AStar:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
    def is_valid(self, pos):
        return (0 <= pos[0] < self.width and 
                0 <= pos[1] < self.height and 
                pos not in self.obstacles)
    
    #Euclidean Heurisitc
    def heuristic(self, a, b):
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    def get_neighbors(self, pos):
        neighbors = []
        for dx, dy in self.directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def find_path(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = []  
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            visited.append(current) 
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        path = []
        if goal in came_from:
            current = goal
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            
        return path, visited