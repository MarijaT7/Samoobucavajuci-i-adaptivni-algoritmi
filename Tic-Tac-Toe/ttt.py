import random
import numpy as np
from collections import defaultdict

board = [[".",".","."],
         [".",".","."],
         [".",".","."]]

def board_to_string(board):
    return ''.join([''.join(row) for row in board])

def string_to_board(s):
    board = []
    for i in range(3):
        board.append(list(s[i*3:(i+1)*3]))
    return board

def check_for_winner(board):
   run_sum = 0
   for row in board:
       for spot in row:
           if(spot == "X"):
               run_sum = run_sum + 1
           elif(spot == "O"):
               run_sum = run_sum - 1
       if(abs(run_sum) == 3):
          return 1
       run_sum = 0
   run_sum = 0
   columns = []

   for j in range(len(board[0])):
     column = []
     for i in range(len(board)):
         column.append(board[i][j])
     columns.append(column)

   for row in columns:
       for spot in row:
           if(spot == "X"):
               run_sum = run_sum + 1
           elif(spot == "O"):
               run_sum = run_sum - 1
       if(abs(run_sum) == 3):
          return 1
       run_sum = 0

   main_diagonal = []
   run_sum = 0
   main_diagonal.append(board[0][0])
   main_diagonal.append(board[1][1])
   main_diagonal.append(board[2][2])
   for spot in main_diagonal:
           if(spot == "X"):
               run_sum = run_sum + 1
           elif(spot == "O"):
               run_sum = run_sum - 1
           if(abs(run_sum) == 3):
               return 1
   sub_diagonal = []
   run_sum = 0
   sub_diagonal.append(board[0][2])
   sub_diagonal.append(board[1][1])
   sub_diagonal.append(board[2][0])
   for spot in sub_diagonal:
           if(spot == "X"):
               run_sum = run_sum + 1
           elif(spot == "O"):
               run_sum = run_sum - 1
           if(abs(run_sum) == 3):
               return 1
   return 0


def list_legal_moves(board):
    possible_moves = []
    i = 0
    for row in board:
        j = 0
        for spot in row:
            if(spot == "."):
                possible_moves.append((i,j))
            j = j + 1
        i = i + 1
    return possible_moves

def whos_turn(board):
    cc = 0
    for row in board:
        for spot in row:
            if(spot == "X"):
                cc = cc + 1
            if(spot == "O"):
                cc = cc - 1

    return cc

def make_move(board, spot, player):
    new_board = [row[:] for row in board]
    i, j = spot
    new_board[i][j] = player
    return new_board

def check_for_terminal(board):
    if check_for_winner(board):
        return True
    if not list_legal_moves(board):
        return True
    return False

def check_for_reward(board, agent_symbol):
    if not check_for_winner(board):
        return 0
    
    x_count = sum(row.count("X") for row in board)
    o_count = sum(row.count("O") for row in board)
    
    if x_count > o_count:
        winner = "X"
    else:
        winner = "O"
    
    if winner == agent_symbol:
        return 1
    else:
        return -1

def random_opp(board):
    possible_moves = list_legal_moves(board)
    make_move(board,random.choice(possible_moves))

def random_opponent_policy(board, player_symbol):
    legal_moves = list_legal_moves(board)
    if not legal_moves:
        return []
    prob = 1.0 / len(legal_moves)
    return [(move, prob) for move in legal_moves]

def minimax(board, is_maximizing, player_symbol):
    if check_for_winner(board):
        x_count = sum(row.count("X") for row in board)
        o_count = sum(row.count("O") for row in board)
        winner = "X" if x_count > o_count else "O"
        return 1 if winner == player_symbol else -1
    
    legal_moves = list_legal_moves(board)
    if not legal_moves:
        return 0
    
    if is_maximizing:
        best_value = -2
        for move in legal_moves:
            new_board = make_move(board, move, player_symbol)
            value = minimax(new_board, False, player_symbol)
            best_value = max(best_value, value)
        return best_value
    else:
        best_value = 2
        opponent = "O" if player_symbol == "X" else "X"
        for move in legal_moves:
            new_board = make_move(board, move, opponent)
            value = minimax(new_board, True, player_symbol)
            best_value = min(best_value, value)
        return best_value

def minimax_opp(board):
    legal_moves = list_legal_moves(board)
    if not legal_moves:
        return
    
    is_x_turn = whos_turn(board) == 0
    player_symbol = "X" if is_x_turn else "O"
    
    best_move = legal_moves[0]
    if is_x_turn:
        best_value = -2
        for move in legal_moves:
            new_board = make_move(board, move, player_symbol)
            value = minimax(new_board, False, player_symbol)
            if value > best_value:
                best_value = value
                best_move = move
    else:
        best_value = 2
        for move in legal_moves:
            new_board = make_move(board, move, player_symbol)
            value = minimax(new_board, True, player_symbol)
            if value < best_value:
                best_value = value
                best_move = move
    
    make_move(board, best_move, player_symbol)

def minimax_opponent_policy(board, player_symbol):
    legal_moves = list_legal_moves(board)
    if not legal_moves:
        return []
    
    best_move = legal_moves[0]
    best_value = -2
    
    for move in legal_moves:
        new_board = make_move(board, move, player_symbol)
        value = minimax(new_board, False, player_symbol)
        if value > best_value:
            best_value = value
            best_move = move
    
    return [(best_move, 1.0)]

class ValueIterationAgent:
    def __init__(self, agent_symbol, opponent_policy, gamma=0.9, delta=0.001):
        self.agent_symbol = agent_symbol
        self.opponent_symbol = "O" if agent_symbol == "X" else "X"
        self.opponent_policy = opponent_policy
        self.gamma = gamma
        self.delta = delta
        self.V = defaultdict(float)
        self.policy = {}
    
    def value_iteration(self, max_iterations=1000):
        iteration = 0
        while iteration < max_iterations:
            max_change = 0
            iteration += 1
            
            states = list(self.V.keys())
            for state_str in states:
                if state_str not in self.V:
                    continue
                    
                board = string_to_board(state_str)
                
                if check_for_terminal(board):
                    continue
                
                if whos_turn(board) == 0 and self.agent_symbol == "X":
                    old_value = self.V[state_str]
                    new_value = self.compute_value(board)
                    self.V[state_str] = new_value
                    max_change = max(max_change, abs(new_value - old_value))
                elif whos_turn(board) != 0 and self.agent_symbol == "O":
                    old_value = self.V[state_str]
                    new_value = self.compute_value(board)
                    self.V[state_str] = new_value
                    max_change = max(max_change, abs(new_value - old_value))
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, max change: {max_change:.6f}")
            
            if max_change < self.delta:
                print(f"Converged after {iteration} iterations")
                break
        
        if iteration >= max_iterations:
            print(f"Stopped after {max_iterations} iterations (did not converge)")
        
        self.derive_policy()
    
    def compute_value(self, board):
        legal_moves = list_legal_moves(board)
        if not legal_moves:
            return 0
        
        max_value = -float('inf')
        
        for action in legal_moves:
            new_board = make_move(board, action, self.agent_symbol)
            
            if check_for_terminal(new_board):
                reward = check_for_reward(new_board, self.agent_symbol)
                value = reward
            else:
                expected_value = 0
                opponent_moves = self.opponent_policy(new_board, self.opponent_symbol)
                
                for opp_action, prob in opponent_moves:
                    next_board = make_move(new_board, opp_action, self.opponent_symbol)
                    next_state_str = board_to_string(next_board)
                    
                    if next_state_str not in self.V:
                        self.V[next_state_str] = 0
                    
                    reward = check_for_reward(next_board, self.agent_symbol)
                    expected_value += prob * (reward + self.gamma * self.V[next_state_str])
                
                value = expected_value
            
            max_value = max(max_value, value)
        
        return max_value
    
    def derive_policy(self):
        for state_str in self.V.keys():
            board = string_to_board(state_str)
            
            if check_for_terminal(board):
                continue
            
            is_agent_turn = (whos_turn(board) == 0 and self.agent_symbol == "X") or \
                           (whos_turn(board) != 0 and self.agent_symbol == "O")
            
            if not is_agent_turn:
                continue
            
            legal_moves = list_legal_moves(board)
            best_action = None
            best_value = -float('inf')
            
            for action in legal_moves:
                new_board = make_move(board, action, self.agent_symbol)
                
                if check_for_terminal(new_board):
                    reward = check_for_reward(new_board, self.agent_symbol)
                    value = reward
                else:
                    expected_value = 0
                    opponent_moves = self.opponent_policy(new_board, self.opponent_symbol)
                    
                    for opp_action, prob in opponent_moves:
                        next_board = make_move(new_board, opp_action, self.opponent_symbol)
                        next_state_str = board_to_string(next_board)
                        reward = check_for_reward(next_board, self.agent_symbol)
                        expected_value += prob * (reward + self.gamma * self.V[next_state_str])
                    
                    value = expected_value
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            self.policy[state_str] = best_action
    
    def initialize_states(self, max_depth=9):
        def generate_states(board, depth=0):
            if depth > max_depth:
                print(f"WARNING: Max depth {max_depth} exceeded")
                return
            
            state_str = board_to_string(board)
            
            if state_str not in self.V:
                if check_for_terminal(board):
                    self.V[state_str] = check_for_reward(board, self.agent_symbol)
                else:
                    self.V[state_str] = 0
            
            if check_for_terminal(board):
                return
            
            legal_moves = list_legal_moves(board)
            current_player = "X" if whos_turn(board) == 0 else "O"
            
            for move in legal_moves:
                new_board = make_move(board, move, current_player)
                generate_states(new_board, depth + 1)
        
        initial_board = [[".",".","."], [".",".","."], [".",".","."]]
        generate_states(initial_board)
        print(f"Initialized {len(self.V)} states")
    
    def get_action(self, board):
        state_str = board_to_string(board)
        if state_str in self.policy:
            return self.policy[state_str]
        legal_moves = list_legal_moves(board)
        return legal_moves[0] if legal_moves else None

def evaluate_agent(agent, opponent_policy, num_games=1000, show_progress=False):
    wins = 0
    losses = 0
    draws = 0
    
    for game_num in range(num_games):
        if show_progress and (game_num + 1) % 100 == 0:
            print(f"  Progress: {game_num + 1}/{num_games} games completed", end='\r')
        
        board = [[".",".","."], [".",".","."], [".",".","."]]
        
        while not check_for_terminal(board):
            current_turn = whos_turn(board)
            
            if (current_turn == 0 and agent.agent_symbol == "X") or \
               (current_turn != 0 and agent.agent_symbol == "O"):
                action = agent.get_action(board)
                if action:
                    board = make_move(board, action, agent.agent_symbol)
            else:
                opp_moves = opponent_policy(board, agent.opponent_symbol)
                if opp_moves:
                    if len(opp_moves) == 1:
                        action = opp_moves[0][0]
                    else:
                        action = random.choices([m[0] for m in opp_moves], 
                                               weights=[m[1] for m in opp_moves])[0]
                    board = make_move(board, action, agent.opponent_symbol)
        
        reward = check_for_reward(board, agent.agent_symbol)
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
    
    if show_progress:
        print()
    
    return wins, losses, draws

print("=" * 60)
print("Training Agent as X (first player) vs Random Opponent")
print("=" * 60)
agent_x_random = ValueIterationAgent("X", random_opponent_policy)
agent_x_random.initialize_states()
agent_x_random.value_iteration()

print("\nEvaluating Agent X vs Random Opponent (1000 games)...")
wins, losses, draws = evaluate_agent(agent_x_random, random_opponent_policy, 1000)
print(f"Results: Wins={wins}, Losses={losses}, Draws={draws}")

print("\n" + "=" * 60)
print("Training Agent as O (second player) vs Random Opponent")
print("=" * 60)
agent_o_random = ValueIterationAgent("O", random_opponent_policy)
agent_o_random.initialize_states()
agent_o_random.value_iteration()

print("\nEvaluating Agent O vs Random Opponent (1000 games)...")
wins, losses, draws = evaluate_agent(agent_o_random, random_opponent_policy, 1000)
print(f"Results: Wins={wins}, Losses={losses}, Draws={draws}")

print("\n" + "=" * 60)
print("Training Agent as X (first player) vs Minimax Opponent")
print("=" * 60)
agent_x_minimax = ValueIterationAgent("X", minimax_opponent_policy)
agent_x_minimax.initialize_states()
agent_x_minimax.value_iteration()

print("\nEvaluating Agent X vs Minimax Opponent (1000 games)...")
wins, losses, draws = evaluate_agent(agent_x_minimax, minimax_opponent_policy, 1000, show_progress=True)
print(f"Results: Wins={wins}, Losses={losses}, Draws={draws}")

print("\n" + "=" * 60)
print("Training Agent as O (second player) vs Minimax Opponent")
print("=" * 60)
agent_o_minimax = ValueIterationAgent("O", minimax_opponent_policy)
agent_o_minimax.initialize_states()
agent_o_minimax.value_iteration()

print("\nEvaluating Agent O vs Minimax Opponent (1000 games)...")
wins, losses, draws = evaluate_agent(agent_o_minimax, minimax_opponent_policy, 1000, show_progress=True)
print(f"Results: Wins={wins}, Losses={losses}, Draws={draws}")
