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


class ChessDataset(Dataset):
    def __init__(self, positions, moves):
        self.positions = positions
        self.moves = moves

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]


# Parallele Verarbeitung der Daten
def process_pgn_batch(file_path, batch_num):
    positions, moves = [], []
    batch_file = os.path.join(BATCH_SAVE_DIR, f"batch_{batch_num}.npz")

    # Überspringe, wenn bereits verarbeitet
    if os.path.exists(batch_file):
        print(f"Batch {batch_num} bereits verarbeitet.")
        return

    print(f"Verarbeite Batch {batch_num}...")
    with open(file_path) as pgn:
        for _ in range(batch_num * 1000):  # Springe zu diesem Batch
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


# Training des Modells
class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.fc1 = nn.Linear(12 * 8 * 8, 512)  # Eingabe: Brettstellung
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 4096)       # Ausgabe: Alle möglichen Züge

    def forward(self, x):
        x = x.view(-1, 12 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model():
    model = ChessMovePredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prüfe, ob ein gespeichertes Modell existiert
    if os.path.exists(MODEL_PATH):
        print("Lade vorhandenes Modell...")
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Modell erfolgreich geladen!")
    else:
        print("Kein gespeichertes Modell gefunden. Neues Training wird gestartet.")

    # Lade Daten batchweise
    for batch_file in sorted(os.listdir(BATCH_SAVE_DIR)):
        print(f"Lade {batch_file}...")
        data = np.load(os.path.join(BATCH_SAVE_DIR, batch_file))
        positions = torch.tensor(data["positions"], dtype=torch.float32)
        moves = torch.tensor(data["moves"], dtype=torch.long)
        dataset = ChessDataset(positions, moves)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        for epoch in range(2):  # Zwei Epochen pro Batch
            total_loss = 0
            for batch_positions, batch_moves in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_positions)
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


# Beispiel für Vorhersagen
def predict_move(board, model):
    model.eval()
    with torch.no_grad():
        input_tensor = board_to_tensor(board).unsqueeze(0)  # Batch-Größe von 1
        output = model(input_tensor)
        move_index = output.argmax().item()
        from_square = move_index // 64
        to_square = move_index % 64
        return chess.Move(from_square, to_square)


if __name__ == "__main__":
    # Parallele Verarbeitung starten
    pgn_path = "lichess_db_standard_rated_2017-10.pgn"
    num_batches = 36  # Passe an die Größe deiner Datei an

    with Pool(processes=4) as pool:
        pool.starmap(process_pgn_batch, [(pgn_path, i) for i in range(num_batches)])

    # Trainiere das Modell
    train_model()

    # Beispielvorhersage
    model = ChessMovePredictor()
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    board = chess.Board()
    predicted_move = predict_move(board, model)
    print("Vorhergesagter Zug:", predicted_move)
