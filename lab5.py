import random

class TicTacToe:
   #cell
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
    
    def print_board(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)
    #player move player as X 
    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    #check winner move straight or diagonal if not return draw
    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return self.board[i][0]
        
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return self.board[0][i]
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None

    def is_draw(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True
    #get empty position or column to play
    def get_empty_positions(self):
        """Return list of all empty positions"""
        positions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    positions.append((i, j))
        return positions
    
    def reset(self):
        """Reset the game board"""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
    
    def get_board_as_string(self):
        """Return board as a string for display"""
        lines = []
        for row in self.board:
            lines.append('|'.join(row))
            lines.append('-' * 5)
        return '\n'.join(lines)


class IntelligentAgent:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent = 'X' if symbol == 'O' else 'O'
    
    def choose_move(self, game):
        board = game.board
        #ai find win move
        move = self.find_winning_move(board, self.symbol)
        if move:
            return move
        #ai block player move
        move = self.find_winning_move(board, self.opponent)
        if move:
            return move
        #ai find corner
        if board[1][1] == ' ':
            return (1, 1)
        
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for corner in corners:
            if board[corner[0]][corner[1]] == ' ':
                return corner
        #ai find edge
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        for edge in edges:
            if board[edge[0]][edge[1]] == ' ':
                return edge
        #ai find any empty cell
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    return (i, j)
        return (0, 0)
    #check win move for player
    def find_winning_move(self, board, player):
        """Check if player can win on next move, return that position"""
        for i in range(3):
            for j in range(3):  
                if board[i][j] == ' ':

                    board[i][j] = player
                    
                    
                    if board[i][0] == board[i][1] == board[i][2] == player:
                        board[i][j] = ' '  
                        return (i, j)
                    
                    if board[0][j] == board[1][j] == board[2][j] == player:
                        board[i][j] = ' '  
                        return (i, j)
                    
                    if i == j and board[0][0] == board[1][1] == board[2][2] == player:
                        board[i][j] = ' '  
                        return (i, j)
                    
                    if i + j == 2 and board[0][2] == board[1][1] == board[2][0] == player:
                        board[i][j] = ' '  
                        return (i, j)
                    
                     
                    board[i][j] = ' '
        
        return None

# Random player that dumb
class RandomPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def choose_move(self, game):
        empty_positions = game.get_empty_positions()
        if empty_positions:
            return random.choice(empty_positions)
        return (0, 0)


def simulate_games(num_games=10, agent_first=True):
    """Simulate multiple games between agent and random player"""
    wins = 0
    losses = 0
    draws = 0
    
    # Store all game results with boards
    all_game_results = []
    
    for game_num in range(num_games):
        game = TicTacToe()
        agent = IntelligentAgent('O')
        random_player = RandomPlayer('X')
        
        if not agent_first:
            agent = IntelligentAgent('X')
            random_player = RandomPlayer('O')
        
        while True:
            # ai turn
            if game.current_player == agent.symbol:
                row, col = agent.choose_move(game)
                game.make_move(row, col)
            # Random turn
            else:
                row, col = random_player.choose_move(game)
                game.make_move(row, col)
            
            winner = game.check_winner()
            if winner:
                if winner == agent.symbol:
                    wins += 1
                    result = "Agent wins"
                else:
                    losses += 1
                    result = "Random wins"
                
                # Store this game result with final board
                all_game_results.append({
                    'game_number': game_num + 1,
                    'result': result,
                    'winner': winner,
                    'final_board': game.get_board_as_string(),
                    'total_moves': len([cell for row in game.board for cell in row if cell != ' '])
                })
                break
            
            if game.is_draw():
                draws += 1
                result = "Draw"
                
                # Store this game result with final board
                all_game_results.append({
                    'game_number': game_num + 1,
                    'result': result,
                    'winner': None,
                    'final_board': game.get_board_as_string(),
                    'total_moves': 9  # All spots filled in a draw
                })
                break
    
    # Return both statistics and all game results
    return wins, losses, draws, all_game_results


def display_all_game_results(game_results, agent_first):
    """Display all game results with their final boards"""
    print("\n" + "="*60)
    print(f"ALL {len(game_results)} GAME RESULTS")
    print(f"Agent playing {'first' if agent_first else 'second'}")
    print("="*60)
    
    # Display 2 games per row for better viewing
    for i in range(0, len(game_results), 2):
        # First game in the row
        game1 = game_results[i]
        print(f"\nGame {game1['game_number']}: {game1['result']} ({game1['total_moves']} moves)")
        print(game1['final_board'])
        
        # Second game in the row if exists
        if i + 1 < len(game_results):
            game2 = game_results[i + 1]
            print(f"Game {game2['game_number']}: {game2['result']} ({game2['total_moves']} moves)")
            print(game2['final_board'])
        
        print("-" * 60)


def show_summary_statistics(wins, losses, draws, num_games):
    """Show summary statistics"""
    print("\n" + "="*40)
    print("SUMMARY STATISTICS")
    print("="*40)
    print(f"Total Games: {num_games}")
    print(f"Agent Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Random Wins: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    
    # Calculate agent performance
    if wins + losses > 0:
        win_ratio = wins / (wins + losses)
        print(f"Win/Loss Ratio: {win_ratio:.2f}")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games
    }


def play_human_game():
    """Play a single game between human and agent"""
    game = TicTacToe()
    agent = IntelligentAgent('O')
    
    while True:
        game.print_board()
        
        if game.current_player == 'X':
            try:
                row, col = map(int, input("Enter row and col (0-2): ").split())
                if not (0 <= row <= 2 and 0 <= col <= 2):
                    print("Invalid input! Use numbers 0-2.")
                    continue
                if not game.make_move(row, col):
                    print("Position already taken!")
                    continue
            except ValueError:
                print("Invalid input! Use format: row col (e.g., '1 1')")
                continue
        else:
            row, col = agent.choose_move(game)
            game.make_move(row, col)
            print(f"Agent plays at ({row}, {col})")
        
        winner = game.check_winner()
        if winner:
            game.print_board()
            print(f"Winner: {winner}")
            break
        
        if game.is_draw():
            game.print_board()
            print("Draw!")
            break


def main():
    """Main menu"""
    all_simulation_results = []  # Store results of all simulations
    
    while True:
        print("\n" + "="*40)
        print("TIC-TAC-TOE GAME")
        print("="*40)
        print("1. Play with AI")
        print("2. Simulation: AI vs Random")
        print("3. View Previous Simulation Results")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            play_human_game()
            
        elif choice == '2':
            num_games = int(input("How many games to simulate? (e.g., 10): "))
            
            print(f"\n{'='*60}")
            print(f"SIMULATION 1: AI plays FIRST (as 'O')")
            print('='*60)
            wins1, losses1, draws1, results1 = simulate_games(num_games, agent_first=True)
            
            # Display all game results with boards
            display_all_game_results(results1, agent_first=True)
            
            # Show summary
            stats1 = show_summary_statistics(wins1, losses1, draws1, num_games)
            stats1['agent_first'] = True
            stats1['results'] = results1
            all_simulation_results.append(stats1)
            
            print(f"\n{'='*60}")
            print(f"SIMULATION 2: AI plays SECOND (as 'X')")
            print('='*60)
            wins2, losses2, draws2, results2 = simulate_games(num_games, agent_first=False)
            
            # Display all game results with boards
            display_all_game_results(results2, agent_first=False)
            
            # Show summary
            stats2 = show_summary_statistics(wins2, losses2, draws2, num_games)
            stats2['agent_first'] = False
            stats2['results'] = results2
            all_simulation_results.append(stats2)
            
            # Compare both simulations
            print("\n" + "="*60)
            print("COMPARISON: AI FIRST vs AI SECOND")
            print("="*60)
            print(f"{'Metric':<20} {'AI First':<15} {'AI Second':<15}")
            print(f"{'-'*20:<20} {'-'*15:<15} {'-'*15:<15}")
            print(f"{'Win Rate':<20} {wins1/num_games*100:.1f}%{'':<8} {wins2/num_games*100:.1f}%")
            print(f"{'Loss Rate':<20} {losses1/num_games*100:.1f}%{'':<8} {losses2/num_games*100:.1f}%")
            print(f"{'Draw Rate':<20} {draws1/num_games*100:.1f}%{'':<8} {draws2/num_games*100:.1f}%")
            
        elif choice == '3':
            if not all_simulation_results:
                print("\nNo simulation results available. Run simulation first.")
                continue
                
            print("\n" + "="*60)
            print(f"PREVIOUS SIMULATION RESULTS ({len(all_simulation_results)} total)")
            print("="*60)
            
            for i, sim in enumerate(all_simulation_results):
                position = "First" if sim['agent_first'] else "Second"
                print(f"\nSimulation {i+1}: AI played {position}")
                print(f"  Games: {sim['total_games']}")
                print(f"  Wins: {sim['wins']} ({sim['wins']/sim['total_games']*100:.1f}%)")
                print(f"  Losses: {sim['losses']} ({sim['losses']/sim['total_games']*100:.1f}%)")
                print(f"  Draws: {sim['draws']} ({sim['draws']/sim['total_games']*100:.1f}%)")
                
                # Ask if user wants to see game boards
                view_boards = input(f"\nView all {sim['total_games']} game boards for this simulation? (y/n): ")
                if view_boards.lower() == 'y':
                    display_all_game_results(sim['results'], sim['agent_first'])
                    
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-4.")


if __name__ == '__main__':
    main()