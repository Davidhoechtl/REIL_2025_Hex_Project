import torch
from copy import deepcopy
from hex_engine import hexPosition
import heapq

def shortest_path_cost(board, player=1):
    size = len(board)
    # Build cost grid: 0 for player's stones, 1 for empty, inf for opponent stones
    cost_grid = [[float('inf')] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if board[r][c] == player:
                cost_grid[r][c] = 0
            elif board[r][c] == 0:
                cost_grid[r][c] = 1
            else:
                cost_grid[r][c] = float('inf')

    # Initialize priority queue with all left edge positions (for white)
    pq = []
    dist = [[float('inf')] * size for _ in range(size)]
    for r in range(size):
        if cost_grid[r][0] != float('inf'):
            dist[r][0] = cost_grid[r][0]
            heapq.heappush(pq, (dist[r][0], (r, 0)))

    # Directions for hex neighbors
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]

    while pq:
        current_dist, (r, c) = heapq.heappop(pq)
        if c == size - 1:  # reached right edge
            return current_dist
        if current_dist > dist[r][c]:
            continue
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                nd = current_dist + cost_grid[nr][nc]
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(pq, (nd, (nr, nc)))
    return float('inf')


def generate_play_data(model_fn, enemy_fn, num_steps, board_size):
    """
    Generate play data by simulating games between model_fn and enemy_fn.

    Args:
        model_fn: The agent to collect training data for (must have `select_action`)
        enemy_fn: The opponent agent
        num_steps: Number of moves to generate
        board_size: Size of the hex board

    Returns:
        List of transition tuples: (state, action, reward, next_state, done)
        Win rate of model_fn
    """
    database = []
    steps_collected = 0
    wins = 0
    total_games = 0

    while steps_collected < num_steps:
        game = hexPosition(size=board_size)
        game.reset()
        trajectory = []

        player_turn = 1  # The one we collect data for (white)

        # Initial path cost before any moves
        prev_cost = shortest_path_cost(game.board, player=player_turn)

        while game.winner == 0:
            state = deepcopy(game.board)
            action_space = game.get_action_space()

            if game.player == player_turn:
                action = model_fn.select_action(torch.tensor(state, dtype=torch.float32), action_space)
            else:
                action = enemy_fn(torch.tensor(state, dtype=torch.float32), action_space)

            scalar_action = game.coordinate_to_scalar(action)
            game.move(action)
            next_state = deepcopy(game.board)
            done = game.winner != 0

            # Only collect data for model_fn's moves
            if game.player != player_turn:  # record previous move
                sate_shortest_path = shortest_path_cost(state, player=player_turn)
                next_state_shortest_path = shortest_path_cost(next_state, player=player_turn)
                shaped_reward = sate_shortest_path - next_state_shortest_path # positive if path got shorter

                # Store step info, reward will be updated at game end if needed
                trajectory.append({
                    "state": state,
                    "action": scalar_action,
                    "next_state": next_state,
                    "done": done,
                    "player": player_turn,
                    "shaped_reward": shaped_reward
                })

        total_games += 1
        if game.winner == player_turn:
            wins += 1

        # Assign final rewards based on outcome, override shaped_reward with +1/-1/0
        for step in trajectory:
            if step["player"] == game.winner:
                step["reward"] = 1
            elif game.winner == 0:
                step["reward"] = 0
            else:
                step["reward"] = -1

            # Combine shaped reward and final reward:
            # For example, weight final reward higher
            combined_reward = 0.8 * step["reward"] + 0.2 * step["shaped_reward"]

            transition = (
                torch.tensor(step["state"], dtype=torch.float32),
                step["action"],
                combined_reward,
                torch.tensor(step["next_state"], dtype=torch.float32),
                step["done"]
            )
            database.append(transition)
            steps_collected += 1
            if steps_collected >= num_steps:
                break

    return database, wins / total_games
