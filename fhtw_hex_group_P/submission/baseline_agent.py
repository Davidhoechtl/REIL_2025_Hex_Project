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