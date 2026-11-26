import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from heapq import heappush, heappop # used if you want to use heap queue for priority queue (easier to implement) but not compulsory

def display_map(map_grid, path, start, goal, explored):

    map_display = np.full(map_grid.shape, 'white', dtype=object)

    # Obstacles
    map_display[map_grid == -1] = 'black'

    # Cases explorées
    for position in explored:
        if map_display[tuple(position)] == 'white':
            map_display[tuple(position)] = 'grey'

    # Chemin
    for position in path:
        if map_display[position[0], position[1]] in ['white', 'grey']:
            map_display[position[0], position[1]] = 'blue'  # Path

    # Départ et arrivée : toujours prioritaires
    map_display[start[0], start[1]] = 'green'
    map_display[goal[0], goal[1]] = 'red'

    color_mapping = {
        'white': 0,
        'black': 1,
        'blue': 2,
        'green': 3,
        'red': 4,
        'grey': 5
    }

    cmap = ListedColormap([
        'white', 'black', 'blue', 'green', 'red', 'grey'
    ])

    map_numeric_display = np.vectorize(color_mapping.get)(map_display)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(map_numeric_display, cmap=cmap, vmin=0, vmax=5)

    ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1))
    ax.grid(True, color='gray')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()


def heuristic_oct(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def grid_search_oct(map_grid, S, G):
    
    ## initialize the varibales above
    came_from = {}      # to reconstruct path
    g_costs = {S: 0}    # cost from start to the cell
    explored = set()    # to keep track of explored cells
    operation_count = 0 # to count the number of operations
    open_set = [(heuristic_oct(S, G), 0, S)]  # priority queue for A* (f_cost, g_cost, position)
        
    while open_set: # G goal is unmarked
        current_f_cost, current_g_cost, current_pos = heappop(open_set)

        explored.add(current_pos)
        
        if current_pos == G: # if G goal is marked
            break

        # Get neighbors (up, down, left, right)
        neighbors = [
            (current_pos[0]-1, current_pos[1]),  # Up
            (current_pos[0]+1, current_pos[1]),  # Down
            (current_pos[0], current_pos[1]-1),  # Left
            (current_pos[0], current_pos[1]+1),  # Right
            (current_pos[0]-1, current_pos[1]-1),  # Up-Left
            (current_pos[0]-1, current_pos[1]+1),  # Up-Right
            (current_pos[0]+1, current_pos[1]-1),  # Down-Left
            (current_pos[0]+1, current_pos[1]+1)   # Down-Right
        ]

        for neighbor in neighbors: # for each neighbor of marked cells
            # Check if neighbor is within bounds
            if (0 <= neighbor[0] < map_grid.shape[0]) and (0 <= neighbor[1] < map_grid.shape[1]): #check if the neighbor is less than 1 and greater than 0
                # Check if neighbor is not an obstacle
                if map_grid[neighbor[0], neighbor[1]] != -1:
                    if neighbor[0] != current_pos[0] and neighbor[1] != current_pos[1]:  # Diagonal movement
                        # avoid corner cutting
                        dx = neighbor[0] - current_pos[0]
                        dy = neighbor[1] - current_pos[1]
                        if map_grid[current_pos[0] + dx, current_pos[1]] == -1 or map_grid[current_pos[0], current_pos[1] + dy] == -1: 
                            # or = rèfle strict, si un des 2 est un obstacle il n'emprunte pas la diag, remplacer par AND si sûre que pas dangereux
                            continue
                        move_cost = np.sqrt(2)
                    else: # Straight movement
                        move_cost = 1

                    tentative_g_cost = current_g_cost + move_cost  + map_grid[neighbor[0], neighbor[1]]

                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g_cost
                        came_from[neighbor] = current_pos
                        operation_count += 1 
                        f_cost = tentative_g_cost + heuristic_oct(neighbor, G)
                        heappush(open_set, (f_cost, tentative_g_cost, neighbor))
                        
    # Reconstruct path
    if current_pos == G:
        path = []
        while current_pos != S:
            path.append(current_pos)
            current_pos = came_from[current_pos]
        path.append(S)
        path.reverse()
        return path, explored, operation_count  # Return reversed path and explored cells
    else:
    # If we reach here, no path was found
        return None, explored, operation_count


def grid_to_world(path, resolution, origin_x, origin_y):
    waypoints = []
    for row, col in path:
        # rows aug towards bottom, cols aug towards right, in cart frame, so does our frame
        # center of each cell avoid center of robot neer lim of cells
        x = origin_x + (col + 0.5) * resolution
        y = origin_y - (row + 0.5) * resolution
        waypoints.append((x, y))
    return waypoints


def nav(map_grid, S, G, resolution):
    #display_map(map_grid, [], S, G, [])

    path, explored, operation_count = grid_search_oct(map_grid, S, G)

    # if path :
    #     # print("Path length =", len(path)-1, "\n", path)
    #     # print("Length explored:", len(explored))
    #     # #display_map(map_grid, path, S, G, explored)
    # else:
    #     print("No path found with A*")

    waypoints = grid_to_world(path, resolution, 0, 0)
    return path, map_grid
    