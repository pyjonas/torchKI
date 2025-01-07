import pygame
import chess
from minimax import find_best_move
from loadModel2 import ChessAgent

# Farben und Größen
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8

# Figurenbilder laden
PIECE_IMAGES = {}

def load_piece_images():
    pieces = ['p', 'n', 'b', 'r', 'q', 'k']  # Example names for pieces
    colors = ['w', 's']  # White and black pieces

    for color in colors:
        for piece in pieces:
            image_name = f"{color}_{piece}.png"
            PIECE_IMAGES[f"{color}_{piece}"] = pygame.image.load(f"images/{image_name}")
            PIECE_IMAGES[f"{color}_{piece}"] = pygame.transform.scale(PIECE_IMAGES[f"{color}_{piece}"], (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, board):
    for rank in range(8):
        for file in range(8):
            color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x = chess.square_file(square) * SQUARE_SIZE
            y = (7 - chess.square_rank(square)) * SQUARE_SIZE
            piece_key = f"{'w' if piece.color == chess.WHITE else 's'}_{piece.symbol().lower()}"
            screen.blit(PIECE_IMAGES[piece_key], (x, y))

def get_square_under_mouse(mouse_pos):
    file = mouse_pos[0] // SQUARE_SIZE
    rank = 7 - (mouse_pos[1] // SQUARE_SIZE)
    return chess.square(file, rank)

def play_game_with_visuals(agent, depth):
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Schach: Spieler vs. PyTorch KI")
    clock = pygame.time.Clock()
    load_piece_images()

    board = chess.Board()
    running = True
    selected_square = None
    move_made = False

    while running:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        draw_board(screen, board)
        pygame.display.flip()

        if board.is_game_over():
            print("\nSpiel beendet:", board.result())
            running = False
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if board.turn:  # Spieler ist am Zug
                    clicked_square = get_square_under_mouse(pygame.mouse.get_pos())
                    if selected_square is None:
                        # Erste Auswahl eines Quadrats
                        if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == chess.WHITE:
                            selected_square = clicked_square
                    else:
                        # Zweite Auswahl (Zug machen)
                        move = chess.Move(from_square=selected_square, to_square=clicked_square)
                        if move in board.legal_moves:
                            board.push(move)
                            move_made = True
                        selected_square = None  # Auswahl zurücksetzen

        if not board.turn and not move_made:  # KI ist am Zug
            best_move = agent.choose_move(board)
            board.push(best_move)
            move_made = True

        if move_made:
            move_made = False  # Setze den Zug-Status zurück

        clock.tick(30)  # Aktualisierungsrate

    pygame.quit()

if __name__ == "__main__":
    # PyTorch-KI erstellen
    agent = ChessAgent(model_path="chess_move_predictor.pth")

    # Spiel starten
    play_game_with_visuals(agent, depth=2)