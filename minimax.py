import chess
import math
from functools import lru_cache


@lru_cache(maxsize=10000)  # Maximal 10000 Spielstellungen speichern
def evaluate_board(fen):
    """
    Bewertet das Schachbrett aus der Sicht des aktuellen Spielers.
    """
    #Klappts
    board = chess.Board(fen)
    if board.is_checkmate():
        return 10000 if board.turn else -10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material_score = sum(
        piece_values[piece.symbol().lower()] if piece.color == board.turn else -piece_values[piece.symbol().lower()]
        for square in chess.SQUARES if (piece := board.piece_at(square))
    )

    threat_penalty = sum(
        -piece_values[piece.symbol().lower()] * 0.5 if is_piece_threatened(board, square) else 0
        for square in chess.SQUARES if (piece := board.piece_at(square)) and piece.color == board.turn
    )
    
    # Belohne Bedrohung von gegnerischen Figuren
    attack_bonus = sum(
        piece_values[piece.symbol().lower()] * 0.5 if is_piece_threatened(board, square) else 0
        for square in chess.SQUARES if (piece := board.piece_at(square)) and piece.color != board.turn
    )

    return material_score + threat_penalty + attack_bonus


    
    

#gegener


def evaluate_board_with_opponent_moves(board):
    """
    Bewertet das Schachbrett und berücksichtigt mögliche Gegenzüge des Gegners.
    """
    evaluation = evaluate_board(board.fen())
    opponent_threat_penalty = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            opponent_threat_penalty -= 1000  # Gegnerischer Mattzug
        else:
            piece_captured = board.piece_at(move.to_square)
            if piece_captured and piece_captured.color == board.turn:
                opponent_threat_penalty -= piece_values[piece_captured.symbol().lower()] * 0.8  # Verlust-Penalty
            # Bedrohung durch Angriff
            if is_piece_threatened(board, move.to_square):
                opponent_threat_penalty -= piece_values[piece_captured.symbol().lower()] * 0.3
        board.pop()

    return evaluation + opponent_threat_penalty


def is_piece_threatened(board, square):
    """
    Überprüft, ob die Figur auf dem angegebenen Feld bedroht ist.
    """ 
    piece = board.piece_at(square)
    if not piece:
        return False
    attackers = board.attackers(not piece.color, square)
    for attacker in attackers:
        attacking_piece = board.piece_at(attacker)
        if attacking_piece and piece_values[attacking_piece.symbol().lower()] < piece_values[piece.symbol().lower()]:
            return True
    return False


def evaluate_opponent_moves(board):
    """
    Bewertet die Qualität der gegnerischen Züge aus der Sicht des aktuellen Spielers.
    """
    opponent_move_penalty = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece and captured_piece.color != board.turn:
                opponent_move_penalty += piece_values[captured_piece.symbol().lower()] * 0.7
        board.pop()
    return opponent_move_penalty


piece_values = {
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0
}

def dynamic_piece_value(piece, square, board):
    """
    Passt den Wert einer Figur basierend auf ihrer Sicherheit an.
    """
    value = piece_values[piece.symbol().lower()]
    if is_piece_threatened(board, square):
        value *= 0.5  # Beispiel: Wert halbieren, wenn bedroht
    return value

def is_piece_threatened(board, square):
    """
    Überprüft, ob die Figur auf dem angegebenen Feld bedroht ist.
    """
    attackers = board.attackers(not board.piece_at(square).color, square)
    return len(attackers) > 0

def evaluate_pawn_structure(pawns, color, board):
    """Bewertet die Bauernstruktur."""
    doubled_pawn_penalty = -0.5
    isolated_pawn_penalty = -0.5
    value = 0

    for pawn_square in pawns:
        file = chess.square_file(pawn_square)
        # Doppelbauern
        if len([sq for sq in pawns if chess.square_file(sq) == file]) > 1:
            value += doubled_pawn_penalty

        # Isolierte Bauern
        adjacent_files = [file - 1, file + 1]
        isolated = True
        for adj_file in adjacent_files:
            if 0 <= adj_file <= 7:
                if any(chess.square_file(sq) == adj_file and board.piece_at(sq).color == color for sq in pawns):
                    isolated = False
        if isolated:
            value += isolated_pawn_penalty

    return value


def order_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: board.is_capture(move), reverse=True)  # Priorisiere Schlagzüge
    return moves

def minimax(board, depth, maximizing_player, alpha, beta):
    if depth == 0 or board.is_game_over():
        base_evaluation = evaluate_board_with_opponent_moves(board)
        opponent_penalty = evaluate_opponent_moves(board)
        return base_evaluation - 2*opponent_penalty, None

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
    """Findet den besten Zug mit Minimax."""
    best_move = None
    best_value = -math.inf if board.turn else math.inf

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