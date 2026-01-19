board = [["X","O","X"],
         [".","O","X"],
         [".","O","."]]

def check_for_winner(board):
   #Row check
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
   #Column check
   run_sum = 0
   columns = []

   #Make columns into rows
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

   #Main Diagonal check
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
   #Sub Diagonal check
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

#Returns whos turn it is based on board state
#1 for player Os turn
#0 for player Xs turn
def whos_turn(board):
    cc = 0
    for row in board:
        for spot in row:
            if(spot == "X"):
                cc = cc + 1
            if(spot == "O"):
                cc = cc - 1

    return cc


def make_move(board, spot):
    i, j = spot
    if(whos_turn(board)):
        board[i][j] = "O"
    else:
        board[i][j] = "X"
    return board

def check_for_terminal(board):
    if check_for_winner(board):
        print("Winner is")
        if whos_turn(board):
            print("X")
            return 1
        else:
            print("O")
            return 1

    if not list_legal_moves(board):
        print("Terminal state: Draw")
        return 1
    return 0
    


print(list_legal_moves(board))
print(board)
i = check_for_terminal(board)
print(i)
