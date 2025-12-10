import math

ROW, COL = 0, 0

class Cell:
    def __init__(self, parent=None, f=float("inf"), g=float("inf"), h=float("inf")):
        self.parent = parent
        # A* pathfinding 
        self.f = f #total 
        self.g = g #fromstart
        self.h = h #estimate to goal manhattan, euclidean, octile

def is_valid(row, col):
    return 0 <= row < ROW and 0 <= col < COL

def is_unblocked(grid, row, col): #WALLCHECK
    return grid[row][col] == 0

def is_destination(row, col, dest):  #GOALCHECK
    return row == dest[0] and col == dest[1]

def trace_path(cell_details, dest):
    row, col = dest
    path = []
    while cell_details[row][col].parent is not None: #follow path backward from goal to start
        path.append((row, col))
        row, col = cell_details[row][col].parent
    path.reverse()  
    return path

def manhattan(r, c, dest): 
    return abs(r - dest[0]) + abs(c - dest[1]) #horizontal and vertical only

def euclidean(r, c, dest):
    return math.sqrt((r - dest[0])**2 + (c - dest[1])**2) #straight line distance

def octile(r, c, dest):
    dx = abs(r - dest[0])
    dy = abs(c - dest[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy) #both

def search(grid, src, dest, heuristic=manhattan, greedy=False, dijkstra=False): #function search
    global ROW, COL 
    ROW, COL = len(grid), len(grid[0])
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]): #check valid
        return None
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]): #check wall
        return None

    closed = [[False for _ in range(COL)] for _ in range(ROW)] #CHECK KUNG NAAGIAH
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)] #CHECK ANG NA AGIAN
#coordinates
    i, j = src
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent = None

    open_list = {(i, j): 0} #LIST POSIBLE MGA AGIAN

    while open_list:
        current = min(open_list, key=open_list.get) #PINAKA DUOL CHCK
        i, j = current
        del open_list[current] #MOVE
        closed[i][j] = True

        if is_destination(i, j, dest):
            return trace_path(cell_details, dest)#GOAL?

        moves = [(-1,0),(1,0),(0,-1),(0,1)] 
        for dr, dc in moves: #CKMOVE
            row, col = i + dr, j + dc
            if is_valid(row, col) and is_unblocked(grid, row, col) and not closed[row][col]:
                g_new = cell_details[i][j].g + 1
                if dijkstra:
                    h_new = 0
                    f_new = g_new
                elif greedy:
                    h_new = heuristic(row, col, dest)
                    f_new = h_new
                else:
                    h_new = heuristic(row, col, dest)
                    f_new = g_new + h_new

                if cell_details[row][col].f == float("inf") or cell_details[row][col].f > f_new: #CURRENT update
                    open_list[(row, col)] = f_new
                    cell_details[row][col].f = f_new
                    cell_details[row][col].g = g_new
                    cell_details[row][col].h = h_new
                    cell_details[row][col].parent = (i, j)
    return None
