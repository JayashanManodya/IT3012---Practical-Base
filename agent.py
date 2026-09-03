from collections import deque
import heapq
import math


class SimpleReflexAgent:
    """Very small reflex agent that reacts to immediate perception."""

    def __init__(self):
        self.actions = ['Up', 'Right', 'Down', 'Left']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Up'
        if percept.get('wall_ahead'):
            return 'Right'
        return 'Up'


class ModelBasedAgent:
    """Minimal memory-based agent that changes its action if the same situation repeats."""

    def __init__(self):
        self.last_action = None
        self.last_percept = None

    def sense_and_act(self, percept: dict) -> str:
        if self.last_action is not None and self.last_percept == percept:
            alternatives = {'Up': 'Right', 'Right': 'Down', 'Down': 'Left', 'Left': 'Up'}
            action = alternatives.get(self.last_action, 'Up')
        else:
            action = 'Up' if not percept.get('wall_ahead', False) else 'Right'

        self.last_action = action
        self.last_percept = percept
        return action


class SearchAgent:
    """Graph-search problem-solving agent for grid navigation."""

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return int(abs(x1 - x2) + abs(y1 - y2))

    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        walls = set(walls)
        heuristic = self.euclidean_distance if heuristic_type.lower() == 'euclidean' else self.manhattan_distance
        priority_queue = []
        reached_states = set()
        start_g_cost = 0
        start_f_cost = start_g_cost + heuristic(start_pos, goal_pos)
        heapq.heappush(priority_queue, (start_f_cost, start_g_cost, start_pos, []))

        while priority_queue:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(priority_queue)
            if current_pos == goal_pos:
                return path_taken
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for action, next_pos in self._valid_moves(current_pos, walls, grid_size):
                if next_pos in reached_states:
                    continue
                new_g_cost = g_cost + 1
                new_h_cost = heuristic(next_pos, goal_pos)
                new_f_cost = new_g_cost + new_h_cost
                new_path = path_taken + [action]
                heapq.heappush(priority_queue, (new_f_cost, new_g_cost, next_pos, new_path))

        return []

    MOVE_DELTAS = {
        'Up': (0, 1),
        'Right': (1, 0),
        'Down': (0, -1),
        'Left': (-1, 0),
    }

    def _valid_moves(self, pos, walls, grid_size):
        x, y = pos
        width, height = grid_size
        walls = set(walls)
        for action in ['Up', 'Right', 'Down', 'Left']:
            dx, dy = self.MOVE_DELTAS[action]
            nx, ny = x + dx, y + dy
            new_pos = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and new_pos not in walls:
                yield action, new_pos

    def _reconstruct_path(self, parent_map, start_pos, goal_pos):
        if goal_pos not in parent_map:
            return []

        path = []
        current = goal_pos
        while current != start_pos:
            prev_pos, action = parent_map[current]
            path.append(action)
            current = prev_pos
        path.reverse()
        return path

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-first search with a FIFO queue; chooses shortest path in unweighted grids."""
        walls = set(walls)
        if start_pos == goal_pos:
            return []

        queue = deque([start_pos])
        reached = {start_pos}
        parent_map = {}

        while queue:
            current = queue.popleft()
            for action, next_pos in self._valid_moves(current, walls, grid_size):
                if next_pos in reached:
                    continue
                reached.add(next_pos)
                parent_map[next_pos] = (current, action)
                if next_pos == goal_pos:
                    return self._reconstruct_path(parent_map, start_pos, goal_pos)
                queue.append(next_pos)

        return []

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-first search with a LIFO stack; keeps a reached set to avoid loops."""
        walls = set(walls)
        if start_pos == goal_pos:
            return []

        stack = [start_pos]
        reached = {start_pos}
        parent_map = {}

        while stack:
            current = stack.pop()
            for action, next_pos in reversed(list(self._valid_moves(current, walls, grid_size))):
                if next_pos in reached:
                    continue
                reached.add(next_pos)
                parent_map[next_pos] = (current, action)
                if next_pos == goal_pos:
                    return self._reconstruct_path(parent_map, start_pos, goal_pos)
                stack.append(next_pos)

        return []

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-cost search using a priority queue ordered by accumulated path cost g(n)."""
        walls = set(walls)
        if start_pos == goal_pos:
            return []

        priority_queue = [(0, start_pos)]
        reached = {start_pos}
        parent_map = {start_pos: (None, None)}
        path_cost = {start_pos: 0}

        while priority_queue:
            cost, current = heapq.heappop(priority_queue)
            if current == goal_pos:
                return self._reconstruct_path(parent_map, start_pos, goal_pos)

            for action, next_pos in self._valid_moves(current, walls, grid_size):
                new_cost = cost + 1
                if next_pos in reached and new_cost >= path_cost.get(next_pos, float('inf')):
                    continue

                reached.add(next_pos)
                path_cost[next_pos] = new_cost
                parent_map[next_pos] = (current, action)
                heapq.heappush(priority_queue, (new_cost, next_pos))

        return []

    def sense_and_act(self, percept: dict) -> str:
        if self.plan:
            return self.plan.pop(0)

        all_food = percept.get('all_food', [])
        remaining_food = percept.get('remaining_food', len(all_food))
        if remaining_food == 0 or not all_food:
            return 'Up'

        start_pos = tuple(percept.get('agent_pos', (0, 0)))
        grid_size = percept.get('grid_size', (10, 10))
        walls = percept.get('walls', [])

        closest_food = min(all_food, key=lambda food: abs(food[0] - start_pos[0]) + abs(food[1] - start_pos[1]))
        algo_name = self.active_algo.upper()
        if algo_name == 'BFS':
            self.plan = self.bfs_search(start_pos, closest_food, walls, grid_size)
        elif algo_name == 'DFS':
            self.plan = self.dfs_search(start_pos, closest_food, walls, grid_size)
        elif algo_name == 'UCS':
            self.plan = self.ucs_search(start_pos, closest_food, walls, grid_size)
        elif self.active_algo == 'AStar':
            self.plan = self.astar_search(start_pos, closest_food, walls, grid_size)
        else:
            self.plan = self.bfs_search(start_pos, closest_food, walls, grid_size)

        if not self.plan:
            return 'Up'
        return self.plan.pop(0)


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept.get('agent_pos', (0, 0))
        return self.actions_pool[0]


if __name__ == '__main__':
    search_agent = SearchAgent()
    mock_start = (0, 0)
    mock_goal = (3, 4)
    print(search_agent.manhattan_distance(mock_start, mock_goal))
    print(search_agent.euclidean_distance(mock_start, mock_goal))