import random

def greedy_agent(board, action_set):
    """
    Greedy agent for Hex: selects the move that maximizes the number of adjacent friendly stones.
    """
    def count_adjacent_friends(board, move, player):
        size = len(board)
        i, j = move
        adjacent = [
            (i-1, j), (i+1, j), (i, j-1), (i, j+1),
            (i-1, j+1), (i+1, j-1)
        ]
        count = 0
        for x, y in adjacent:
            if 0 <= x < size and 0 <= y < size:
                if board[x][y] == player:
                    count += 1
        return count

    # Determine current player from board (sum of stones)
    flat = [cell for row in board for cell in row]
    player = 1 if flat.count(1) <= flat.count(-1) else -1

    best_move = None
    best_score = -1
    for move in action_set:
        score = count_adjacent_friends(board, move, player)
        if score > best_score:
            best_score = score
            best_move = move

    # Fallback: just pick the first move if all scores are equal
    return best_move if best_move is not None else action_set[0]

def random_agent(board, action_set):
    """
    Random agent for Hex: selects a move uniformly at random from the available actions.
    """
    return random.choice(action_set)

def opponent_adjacent_agent(board, action_set):
    """
    Picks moves adjacent to opponent stones to block, else random.
    """
    size = len(board)
    flat = [cell for row in board for cell in row]
    player = 1 if flat.count(1) <= flat.count(-1) else -1
    opponent = -player

    def is_adjacent_to_opponent(move):
        i, j = move
        adjacent = [
            (i-1, j), (i+1, j), (i, j-1), (i, j+1),
            (i-1, j+1), (i+1, j-1)
        ]
        for x, y in adjacent:
            if 0 <= x < size and 0 <= y < size:
                if board[x][y] == opponent:
                    return True
        return False

    blocking_moves = [move for move in action_set if is_adjacent_to_opponent(move)]

    if blocking_moves:
        return random.choice(blocking_moves)
    else:
        return random.choice(action_set)


def center_seeking_agent(board, action_set):
    """
    Prefers moves closest to the center of the board.
    """
    size = len(board)
    center = (size - 1) / 2

    def dist_to_center(move):
        i, j = move
        return abs(i - center) + abs(j - center)  # Manhattan distance to center

    min_dist = min(dist_to_center(move) for move in action_set)
    best_moves = [move for move in action_set if dist_to_center(move) == min_dist]
    return random.choice(best_moves)

def edge_seeking_agent(board, action_set):
    """
    Prefers moves closest to any board edge.
    """
    size = len(board)

    def dist_to_edge(move):
        i, j = move
        return min(i, j, size - 1 - i, size - 1 - j)

    min_dist = min(dist_to_edge(move) for move in action_set)
    best_moves = [move for move in action_set if dist_to_edge(move) == min_dist]
    return random.choice(best_moves)

def corner_seeking_agent(board, action_set):
    """
    Prefers moves closest to the top-left corner (0,0).
    """
    def dist_to_corner(move):
        i, j = move
        return i + j  # Manhattan distance to (0,0)

    min_dist = min(dist_to_corner(move) for move in action_set)
    best_moves = [move for move in action_set if dist_to_corner(move) == min_dist]
    return random.choice(best_moves)
