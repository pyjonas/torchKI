import chess
import math
from functools import lru_cache

piece_values = {
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0
}

@lru_cache(maxsize=10000)
def evaluate_board(fen):
    """
    Bewertet das Schachbrett basierend auf dem Materialwert.
    """
    board = chess.Board(fen)

    if board.is_checkmate():
        return 10000 if board.turn else -10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material_score = sum(
        piece_values[piece.symbol().lower()] if piece.color == board.turn else -piece_values[piece.symbol().lower()]
        for square in chess.SQUARES if (piece := board.piece_at(square))
    )

    return material_score

def order_moves(board):
    """
    Sortiert Züge basierend auf ihrer Wichtigkeit, z. B. Schlagzüge priorisieren.
    """
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: board.is_capture(move), reverse=True)
    return moves

def minimax(board, depth, maximizing_player, alpha, beta):
    """
    Implementiert den Minimax-Algorithmus mit Alpha-Beta-Pruning.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board.fen()), None

    best_move = None

    if maximizing_player:
        max_eval = float('-inf')
        for move in order_moves(board):
            board.push(move)
            eval, _ = minimax(board, depth - 1, False, alpha, beta)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_move

    else:
        min_eval = float('inf')
        for move in order_moves(board):
            board.push(move)
            eval, _ = minimax(board, depth - 1, True, alpha, beta)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval, best_move

def find_best_move(board, depth):
    """
    Findet den besten Zug mit Minimax.
    """
    best_move = None
    best_value = float('-inf') if board.turn else float('inf')

    for move in board.legal_moves:
        board.push(move)
        board_value, _ = minimax(board, depth=depth, maximizing_player=not board.turn, alpha=float('-inf'), beta=float('inf'))
        board.pop()

        if board.turn:  # Maximizing player
            if board_value > best_value:
                best_value = board_value
                best_move = move
        else:  # Minimizing player
            if board_value < best_value:
                best_value = board_value
                best_move = move

    return best_move

# Beispiel-Aufruf
if __name__ == "__main__":
    board = chess.Board()
    depth = 3
    best_move = find_best_move(board, depth)
    print(f"Bester Zug: {best_move}")