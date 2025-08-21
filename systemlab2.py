import heapq
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, x, y, is_walkable=True):
        self.x = x
        self.y = y
        self.is_walkable = is_walkable
        self.g = float('inf')  
        self.parent = None
    
    def __lt__(self, other):
        return self.g < other.g

class Grid:
    def __init__(self, width, height, obstacle_density=0.2):
        self.width = width
        self.height = height
        self.grid = self.create_grid(obstacle_density)
    
    def create_grid(self, obstacle_density):
        grid = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                if (x == 0 and y == 0) or (x == self.width-1 and y == self.height-1):
                    row.append(Node(x, y, True))
                else:
                    is_obstacle = np.random.random() < obstacle_density
                    row.append(Node(x, y, not is_obstacle))
            grid.append(row)
        return grid
    
    def get_node(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return None
    
    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = node.x + dx, node.y + dy
            neighbor = self.get_node(nx, ny)
            if neighbor and neighbor.is_walkable:
                neighbors.append(neighbor)
        return neighbors

def dijkstra(grid, start, goal):
    open_set = []  
    closed_set = set()  
    nodes_explored = 0
    
    start.g = 0
    heapq.heappush(open_set, start)
    
    while open_set:
        current = heapq.heappop(open_set)
        nodes_explored += 1
        
        if current.x == goal.x and current.y == goal.y:
            return get_path(current), nodes_explored
        
        closed_set.add((current.x, current.y))
        
        for neighbor in grid.get_neighbors(current):
            if (neighbor.x, neighbor.y) in closed_set:
                continue
            
            new_g = current.g + 1
            
            if new_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = new_g
                
                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
    
    return None, nodes_explored  

def get_path(node):
    path = []
    current = node
    while current:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]  
def visualize_dijkstra(grid, path, explored_nodes):
    plt.figure(figsize=(10, 8))
    
    grid_data = np.ones((grid.width, grid.height))
    for x in range(grid.width):
        for y in range(grid.height):
            if not grid.grid[x][y].is_walkable:
                grid_data[x, y] = 0  
    
    plt.imshow(grid_data.T, cmap='gray', origin='lower')
    
    plt.scatter(0, 0, c='green', s=300, marker='s', label='Start', edgecolors='white')
    plt.scatter(grid.width-1, grid.height-1, c='red', s=300, marker='s', label='Goal', edgecolors='white')
    
    if explored_nodes:
        explored_x = [node[0] for node in explored_nodes]
        explored_y = [node[1] for node in explored_nodes]
        plt.scatter(explored_x, explored_y, c='lightblue', s=40, alpha=0.7, label='Explored Nodes')
    
    if path:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        plt.plot(path_x, path_y, c='yellow', linewidth=4, label='Shortest Path')
        plt.scatter(path_x, path_y, c='yellow', s=60, alpha=0.8)
    
    plt.title('Dijkstra\'s Algorithm - Shortest Path Finding')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_explored_nodes(grid):
    explored_nodes = []
    for x in range(grid.width):
        for y in range(grid.height):
            if grid.grid[x][y].g < float('inf'):  
                explored_nodes.append((x, y))
    return explored_nodes

def main():
    print("=== Dijkstra's Pathfinding Algorithm ===\n")
    
    grid = Grid(20, 20, obstacle_density=0.25)
    start = grid.get_node(0, 0)
    goal = grid.get_node(19, 19)
    
    print(f"Grid Size: {grid.width}x{grid.height}")
    print(f"Start: ({start.x}, {start.y})")
    print(f"Goal: ({goal.x}, {goal.y})")
    print(f"Obstacle density: 25%\n")
    
    path, nodes_explored = dijkstra(grid, start, goal)
    
    explored_nodes = get_explored_nodes(grid)
    
    if path:
        print(f"✓ Path found!")
        print(f"Path length: {len(path)} steps")
        print(f"Nodes explored: {nodes_explored}")
        print(f"Shortest path cost: {len(path) - 1}")
    else:
        print(f"✗ No path found!")
        print(f"Nodes explored: {nodes_explored}")
    
    visualize_dijkstra(grid, path, explored_nodes)

if __name__ == "__main__":
    main()
