import torch
import chess
from neuronalesNetzwerk import ChessMovePredictor, board_to_tensor, predict_move
from minimax3 import find_best_move  # Minimax Algorithmus importieren

class ChessAgent:
    def __init__(self, model_path="chess_move_predictor.pth"):
        self.model = ChessMovePredictor()
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.eval()  # Setzt das Modell in den Evaluierungsmodus

    def choose_move(self, board):
        """
        Wählt den besten Zug basierend auf der Modellvorhersage.
        """
        with torch.no_grad():
            input_tensor = board_to_tensor(board).unsqueeze(0)  # Brettstellung als Tensor vorbereiten
            output = self.model(input_tensor)  # Modellvorhersage
            move_index = output.argmax().item()  # Den vorhergesagten Index finden
            from_square = move_index // 64
            to_square = move_index % 64
            predicted_move = chess.Move(from_square, to_square)

            # Überprüfe, ob der Zug legal ist, und gib ihn zurück
            if predicted_move in board.legal_moves:
                return predicted_move
            else:

                # Fallback: Wähle den ersten legalen Zug, falls die Vorhersage ungültig ist
                print("Ungültiger Zug vorhergesagt, wähle einen Zufallszug.")
                return find_best_move(board,2)
