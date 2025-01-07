import chess.pgn
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from multiprocessing import Pool

# Globale Parameter
MODEL_PATH = "chess_move_predictor.pth"
BATCH_SAVE_DIR = "processed_batches"
os.makedirs(BATCH_SAVE_DIR, exist_ok=True)  # Ordner für gespeicherte Daten erstellen



def process_game(game):
    positions = []
    moves = []

    board = game.board()
    for move in game.mainline_moves():
        positions.append(board_to_tensor(board))
        moves.append(move_to_index(move))
        board.push(move)

    return positions, moves

def board_to_tensor(board):
    """
    Wandelt die Brettstellung in eine Tensor-Repräsentation um.
    - Output: Tensor der Größe (12, 8, 8), repräsentiert die Figuren auf dem Brett.
    """
    tensor = np.zeros((12, 8, 8))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        plane = piece_type_to_plane(piece)
        row, col = divmod(square, 8)
        tensor[plane, row, col] = 1
    return torch.tensor(tensor, dtype=torch.float32)

def move_to_index(move):
    """
    Konvertiere einen Zug in einen eindeutigen Index für das Output-Layer.
    """
    return move.from_square * 64 + move.to_square

def piece_type_to_plane(piece):
    """
    Ordne jeder Figur eine Ebene im Tensor zu.
    """
    type_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    return type_to_plane[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)

def augment_position(position):
    """
    Erzeugt augmentierte Versionen der Brettstellung (Spiegeln und Rotieren).
    """
    augmented_positions = []

    # Original
    augmented_positions.append(position)

    # Spiegelung entlang der vertikalen Achse
    flipped_position = torch.flip(position, dims=[2])
    augmented_positions.append(flipped_position)

    # Rotationen (90, 180, 270 Grad)
    for k in range(1, 4):
        rotated_position = torch.rot90(position, k, dims=[1, 2])
        augmented_positions.append(rotated_position)

    return augmented_positions

def augment_dataset(positions, moves):
    """
    Wendet Augmentation auf das gesamte Dataset an.
    """
    augmented_positions = []
    augmented_moves = []

    for position, move in zip(positions, moves):
        augmented = augment_position(position)
        for aug_pos in augmented:
            augmented_positions.append(aug_pos)
            augmented_moves.append(move)  # Der Zug bleibt gleich

    return augmented_positions, augmented_moves

def filter_by_piece_count(positions, moves, max_pieces):
    """
    Filtert Daten basierend auf der maximalen Anzahl an Figuren auf dem Brett.
    """
    filtered_positions = []
    filtered_moves = []

    for position, move in zip(positions, moves):
        # Zähle die Anzahl der Figuren
        piece_count = position.sum().item()
        if piece_count <= max_pieces:
            filtered_positions.append(position)
            filtered_moves.append(move)

    return filtered_positions, filtered_moves

class ChessDataset(Dataset):
    def __init__(self, positions, moves):
        self.positions = positions
        self.moves = moves

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]

def process_pgn_batch(file_path, batch_num):
    positions, moves = [], []
    batch_file = os.path.join(BATCH_SAVE_DIR, f"batch_{batch_num}.npz")

    # Überspringe, wenn bereits verarbeitet
    if os.path.exists(batch_file):
        print(f"Batch {batch_num} bereits verarbeitet.")
        return

    print(f"Verarbeite Batch {batch_num}...")
    with open(file_path) as pgn:
        for _ in range((batch_num) * 1000):  # Springe zu diesem Batch
            chess.pgn.read_game(pgn)
        for _ in range(1000):  # Lese bis zu 1000 Spiele
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            pos, mov = process_game(game)
            positions.extend(pos)
            moves.extend(mov)

    # Speichere den Batch
    np.savez_compressed(batch_file, positions=np.array(positions), moves=np.array(moves))
    print(f"Batch {batch_num} gespeichert.")

def is_legal_move(board, move_index):

    """
    Prüft, ob ein vorhergesagter Zug legal ist.
    """
    from_square = move_index // 64
    to_square = move_index % 64
    move = chess.Move(from_square, to_square)
    return move in board.legal_moves

def apply_legal_mask(output, board):
    """
    Wendet eine Maske an, die illegale Züge blockiert.
    """
    mask = torch.zeros(4096)  # Alle möglichen Züge initial blockieren
    for move in board.legal_moves:
        index = move_to_index(move)
        mask[index] = 1
    return output * mask  # Blockiere illegale Züge

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.fc1 = nn.Linear(12 * 8 * 8, 512)  # Eingabe: Brettstellung
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4096)       # Ausgabe: Alle möglichen Züge

    def forward(self, x, boards):
        x = x.view(-1, 12 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Maske für legale Züge anwenden
        for i, single_output in enumerate(x):
            x[i] = apply_legal_mask(single_output, boards[i])

        return x

def train_model():
    model = ChessMovePredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prüfe, ob ein gespeichertes Modell existiert
    if os.path.exists(MODEL_PATH):
        print("Lade vorhandenes Modell...")
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Modell erfolgreich geladen!")
    else:
        print("Kein gespeichertes Modell gefunden. Neues Training wird gestartet.")

    # Lade Daten batchweise und trainiere mit Curriculum Learning
    for max_pieces in range(6, 33, 5):  # Curriculum: Steigere max_pieces
        print(f"Training mit maximal {max_pieces} Figuren auf dem Brett...")
        for batch_file in sorted(os.listdir(BATCH_SAVE_DIR)):
            print(f"Lade {batch_file}...")
            data = np.load(os.path.join(BATCH_SAVE_DIR, batch_file))
            positions = torch.tensor(data["positions"], dtype=torch.float32)
            moves = torch.tensor(data["moves"], dtype=torch.long)

            # Filtere nach der Anzahl der Figuren
            positions, moves = filter_by_piece_count(positions, moves, max_pieces)

            # Überspringe, wenn keine Daten übrig bleiben
            if len(positions) == 0:
                continue

            # Wende Augmentation an
            positions, moves = augment_dataset(positions, moves)

            dataset = ChessDataset(positions, moves)
            dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

            for epoch in range(2):  # Zwei Epochen pro Batch
                total_loss = 0
                for batch_positions, batch_moves in dataloader:
                    optimizer.zero_grad()

                    # Erstelle ein Dummy-Brett für jede Position
                    boards = []
                    for pos in batch_positions:
                        board = chess.Board()
                        for move in pos:  # Rekonstruiere das Brett basierend auf Moves
                            board.push(move)
                        boards.append(board)

                    # Vorhersage mit Maskierung
                    outputs = model(batch_positions, boards)
                    loss = criterion(outputs, batch_moves)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Loss für {batch_file}, Epoch {epoch + 1}: {total_loss:.4f}")

    # Speichere das Modell
    print("Speichere Modell...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, MODEL_PATH)
    print(f"Modell gespeichert unter: {MODEL_PATH}")

if __name__ == "__main__":
    pgn_path = "lichess_db_standard_rated_2017-10.pgn"
    num_batches = 6

    with Pool(processes=4) as pool:
        pool.starmap(process_pgn_batch, [(pgn_path, i) for i in range(num_batches)])

    train_model()
