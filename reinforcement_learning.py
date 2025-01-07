import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from minimax import find_best_move  # Minimax Algorithmus importieren

class ChessAgent:
    def __init__(self):
        self.model = self.build_model()
        self.memory = []  # Erfahrungsspeicher (Replay Buffer)
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1.0  # Exploration vs. Exploitation
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self):
        """
        Erstellt ein Convolutional Neural Network zur Bewertung von Stellungen.
        """
        return nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),  # 12 Kanäle -> 32 Filter
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def choose_move(self, board):
        """
        Wählt einen Zug basierend auf dem aktuellen Zustand (Exploration vs. Exploitation).
        """
        legal_moves = list(board.legal_moves)
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)  # Zufälliger Zug
        else:
            best_move = None
            best_value = -np.inf
            for move in legal_moves:
                board.push(move)
                value = self.evaluate_position(board)
                board.pop()
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_move

    def evaluate_position(self, board):
        """
        Bewertet eine Position basierend auf dem neuronalen Netzwerk.
        """
        input_data = self.board_to_input(board)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.model(input_tensor).item()

    def board_to_input(self, board):
        """
        Konvertiert das Schachbrett in ein Eingabeformat für das neuronale Netzwerk.
        """
        input_data = np.zeros((12, 8, 8))  # 12 Kanäle: 6 für Weiß, 6 für Schwarz
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1  # Bauer: 0, Springer: 1, etc.
                if not piece.color:  # Schwarz
                    channel += 6
                rank, file = divmod(square, 8)
                input_data[channel, rank, file] = 1
        return input_data

    def train(self, batch_size=32):
        """
        Trainiert das neuronale Netzwerk mit Erfahrungen aus dem Replay Buffer.
        """
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                with torch.no_grad():
                    target += self.discount_factor * self.model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).item()
            targets.append(target)
            states.append(state)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        self.optimizer.zero_grad()
        predictions = self.model(states_tensor).squeeze()
        loss = self.criterion(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def store_experience(self, state, action, reward, next_state, done):
        """
        Speichert eine Erfahrung im Replay Buffer.
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)


def self_play(agent, episodes=100):
    """
    Führt Selbstspiele durch, um die KI zu trainieren.
    """
    for episode in range(episodes):
        print(f"Selbstspiel {episode + 1}/{episodes}")
        board = chess.Board()
        while not board.is_game_over():
            state = agent.board_to_input(board)
            move = agent.choose_move(board)
            board.push(move)

            done = board.is_game_over()
            reward = 0 if not done else 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0
            next_state = agent.board_to_input(board)
            agent.store_experience(state, move, reward, next_state, done)

        agent.train(batch_size=64)


def imitation_learning(agent, minimax_depth=3, episodes=100):
    """
    Trainiert die KI durch Nachahmung von Minimax-Zügen.
    """
    for episode in range(episodes):
        print(f"Imitation Learning Episode {episode + 1}/{episodes}")
        board = chess.Board()
        while not board.is_game_over():
            state = agent.board_to_input(board)
            if board.turn:
                # Minimax generiert den Zug
                move = find_best_move(board, minimax_depth)
            else:
                move = agent.choose_move(board)
            board.push(move)

            done = board.is_game_over()
            reward = 0 if not done else 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0
            next_state = agent.board_to_input(board)
            agent.store_experience(state, move, reward, next_state, done)

        agent.train(batch_size=64)


if __name__ == "__main__":
    agent = ChessAgent()

    # Imitation Learning
    imitation_learning(agent, minimax_depth=3, episodes=100)

    # Selbstspiele
    self_play(agent, episodes=100)

