import networkx as nx
import matplotlib.pyplot as plt
import heapq
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pygame

# Manhattan distance
def heuristic(node, goal):
    return 2 * (abs(node[0] - goal[0]) + abs(node[1] - goal[1]))

def visualize_state(graph, start, goal, came_from, current, open_set, path, step, g_score, f_score):
    plt.clf()
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1])
    
    # Main graph plot
    ax1 = fig.add_subplot(gs[:, 0])
    pos = {node: node for node in graph.nodes()}
    
    # Color nodes based on f-scores
    node_colors = []
    max_f = max([score for score in f_score.values() if score != float('inf')])
    for node in graph.nodes():
        if f_score[node] == float('inf'):
            node_colors.append('lightgray')
        else:
            color_intensity = 1 - (f_score[node] / max_f)
            node_colors.append(plt.cm.YlOrRd(color_intensity))
    
    # Draw basic graph
    nx.draw(graph, pos=pos, with_labels=False, node_color=node_colors, 
            node_size=700, edge_color='gray', width=2, ax=ax1)
    
    # Draw node labels with scores
    label_pos = {node: (node[0], node[1] + 0.1) for node in graph.nodes()}
    labels = {node: f'({node[0]},{node[1]})\nf={f_score.get(node, float("inf")):.1f}' 
             for node in graph.nodes()}
    nx.draw_networkx_labels(graph, label_pos, labels, ax=ax1)
    
    # Draw edge weights
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax1)
    
    # Highlight nodes
    if open_set:
        open_nodes = [node for _, node in open_set]
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=open_nodes, 
                             node_color='yellow', node_size=600, ax=ax1)
    
    if current:
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=[current], 
                             node_color='orange', node_size=600, ax=ax1)
    
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=[start], 
                          node_color='green', node_size=600, ax=ax1)
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=[goal], 
                          node_color='red', node_size=600, ax=ax1)
    
    # Draw paths
    if came_from:
        path_edges = [(came_from[node], node) for node in came_from]
        nx.draw_networkx_edges(graph, pos=pos, edgelist=path_edges, 
                             edge_color='blue', width=2, ax=ax1)
    
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos=pos, edgelist=path_edges, 
                             edge_color='red', width=3, ax=ax1)

    ax1.set_title(f"A* Pathfinding Visualization - Step {step}")
    
    legend_elements = [
        mpatches.Patch(color='green', label='Start'),
        mpatches.Patch(color='red', label='Goal'),
        mpatches.Patch(color='orange', label='Current Node'),
        mpatches.Patch(color='yellow', label='Open Set'),
        mpatches.Patch(color='blue', label='Explored Path'),
        mpatches.Patch(color='lightgray', label='Unexplored Node')
    ]
    ax1.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Current Node Analysis')
    
    if current:
        score_text = (
            f"Current Node: {current}\n"
            f"g-score (path cost): {g_score[current]:.1f}\n"
            f"h-score (heuristic): {heuristic(current, goal):.1f}\n"
            f"f-score (total): {f_score[current]:.1f}\n"
            f"\nLower scores are better!"
        )
        ax2.text(0, 0.6, score_text, fontsize=10, verticalalignment='top')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.set_title('Open Set Comparison')
    
    if open_set:
        comparison_text = "Next possible nodes:\n\n"
        for i, (_, node) in enumerate(sorted(open_set)):
            comparison_text += (
                f"Node {node}:\n"
                f"g={g_score[node]:.1f}, "
                f"h={heuristic(node, goal):.1f}, "
                f"f={f_score[node]:.1f}"
            )
            if f_score[node] == min(f_score[n] for _, n in open_set):
                comparison_text += " ✓"
            else:
                comparison_text += " ✗"
            comparison_text += "\n\n"
        ax3.text(0, 0.9, comparison_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def astar(graph, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start] = heuristic(start, goal)
    
    step = 0
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = reconstruct_path(came_from, current)
            yield (came_from, current, open_set, path, step, g_score, f_score)
            return path
        
        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph[current][neighbor]['weight']
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        yield (came_from, current, open_set, None, step, g_score, f_score)
        step += 1
    
    return None

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Create graph with multiple paths
G = nx.Graph()
edges = [
    # Main path
    ((0, 0), (0, 1), 1), ((0, 1), (0, 2), 1), 
    ((0, 2), (1, 2), 1), ((1, 2), (2, 2), 1),
    ((2, 2), (3, 2), 1), ((3, 2), (3, 1), 1),
    ((3, 1), (3, 0), 1), ((3, 0), (4, 0), 1),
    
    # Alternative paths
    ((0, 0), (1, 0), 2), ((1, 0), (2, 0), 2),
    ((2, 0), (2, 1), 1), ((2, 1), (2, 2), 1),
    ((2, 2), (2, 3), 1), ((2, 3), (3, 3), 1),
    ((3, 3), (4, 3), 1), ((4, 3), (4, 2), 1),
    ((4, 2), (4, 1), 1), ((4, 1), (4, 0), 1),
]

for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Set start and goal nodes
start_node = (0, 0)
goal_node = (4, 0)

print("A* Pathfinding Algorithm Visualization")
print("=====================================")
print("Lower scores are better!")
print("- g-score: Cost from start to current node")
print("- h-score: Estimated cost to goal")
print("- f-score: Total cost (g + h)")

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("A* Pathfinding Visualization")
clock = pygame.time.Clock()
print("Press 'Enter' to move to the next step")
# Main loop
running = True
step = 0
path = None
astar_gen = None
final_path_displayed = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if astar_gen is None:
                    astar_gen = astar(G, start_node, goal_node)
                try:
                    came_from, current, open_set, path, step, g_score, f_score = next(astar_gen)
                    visualize_state(G, start_node, goal_node, came_from, current, open_set, path, step, g_score, f_score)
                    if path is not None:
                        print("\nPath found:", path)
                        final_path_displayed = True
                except StopIteration:
                    running = False

    pygame.display.flip()
    clock.tick(60)

# Wait for final enter press to close
if final_path_displayed:
    print("Press 'Enter' to close the window.")
    waiting_for_close = True
    while waiting_for_close:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting_for_close = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    waiting_for_close = False

pygame.quit()
plt.close()