from algo import search, manhattan, euclidean, octile
import matplotlib.pyplot as plt
import numpy as np

def draw_grid_heatmap(ax, grid, path=[], start=None, goal=None, title="Path Visualization"):
    rows, cols = len(grid), len(grid[0])
    
    # Create visualization grid
    vis_grid = np.zeros((rows, cols))
    
    # Mark walls
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                vis_grid[r][c] = -1
    
    # Mark path
    if path:
        for (r, c) in path:
            vis_grid[r][c] = 1
    
    # Mark start and goal
    if start:
        vis_grid[start[0]][start[1]] = 2
    if goal:
        vis_grid[goal[0]][goal[1]] = 3
    
    # Custom colormap
    colors = ['black', 'lightgray', 'red', 'green', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot as heatmap
    im = ax.imshow(vis_grid, cmap=cmap, norm=norm, aspect='equal')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Set labels
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add coordinate labels
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and vis_grid[r][c] not in [1, 2, 3]:
                ax.text(c, r, f'{c},{r}', ha='center', va='center', 
                       fontsize=6, color='gray', alpha=0.5)
    
    ax.set_title(title, pad=15)

grid = [
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,0,1,1,1,1,0,1,1],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,0,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,1,0,0,0,1,0],
    [0,1,1,0,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,0],
]

start, goal = (0, 0), (8, 9)

# --- A* VARIANTS ---
a_star_algorithms = {
    "A* (Manhattan)": manhattan,
    "A* (Euclidean)": euclidean,
    "A* (Octile)": octile
}

# --- OTHER ALGORITHMS ---
other_algorithms = {
    "Greedy (Manhattan)": (manhattan, True, False),
    "Dijkstra": (manhattan, False, True)
}

# Create a legend figure
fig_legend, ax_legend = plt.subplots(figsize=(6, 1))
ax_legend.axis('off')
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Wall'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=10, label='Empty Cell'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Path'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Start'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Goal')
]
ax_legend.legend(handles=legend_elements, ncol=5, loc='center', frameon=False)
plt.tight_layout()
plt.show()

# FIGURE 1: 3 A* Algorithms
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, heuristic) in zip(axes1, a_star_algorithms.items()):
    path = search(grid, start, goal, heuristic, greedy=False, dijkstra=False)
    print(name, "→ Path:", path)
    draw_grid_heatmap(ax, grid, path, start, goal, name)

plt.tight_layout()
plt.show()

# FIGURE 2: Greedy + Dijkstra
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

for ax, (name, (heuristic, greedy, dijkstra)) in zip(axes2, other_algorithms.items()):
    path = search(grid, start, goal, heuristic, greedy, dijkstra)
    print(name, "→ Path:", path)
    draw_grid_heatmap(ax, grid, path, start, goal, name)

plt.tight_layout()
plt.show()