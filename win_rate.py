import os
import json
import numpy as np
from regex import R
path = 'replays'
folder = os.listdir('replays')
# hyperparameters: Need to change it according to tournament!!!
num_team = 8
result = np.zeros((num_team,num_team,2)) 
# 3d array. [winner, loser, [win, total_game]]
# ex : [0,1,0] is the number of game team 0 beat team 1, [0,1,0]
teams = { } # key: agent name, value: agent index in result
i = 0
for file in folder:
    game = json.load(open(path + "/"+ file))
    players = game['teamDetails']
    player1 = players[0]['name']
    player2 = players[1]['name']
    #Add agent to the teams if not already there
    if player1 not in teams:
        teams[player1] = i
        i += 1
        result 
    if player2 not in teams:
        teams[player2] = i
        i += 1
    players_idx = [teams[player1], teams[player2]]
    ranks = game['results']['ranks']
    #update result for winner
    winner = players_idx[ranks[0]['agentID']]
    non_winner = players_idx[ranks[1]['agentID']]
    result[winner][non_winner][0] += 1
    result[winner][non_winner][1] += 1
    #update result for 2nd place
    result[non_winner][winner][1] += 1
    # print(result)
print(teams)


def individual_win_rate(result):
    """
    return win rate for each agent against each other
    """
    win_rate = np.zeros((num_team,num_team)) 
    for i in range(num_team):
        for j in range(num_team):
            if i != j:
                try:
                    win_rate[i][j] = result[i][j][0]/result[i][j][1]
                except:
                    win_rate[i][j] = 0
    print(f"win rate:---{result[i][j][0]}/{result[i][j][1]}")
    return win_rate

def total_win_rate(result):
    """
    return total win rate for each agent
    """
    win_rate = np.zeros((num_team)) 
    for i in range(num_team):
        wins = 0
        games = 0
        for j in range(num_team):
            wins += result[i][j][0]
            games += result[i][j][1]
        win_rate[i] = wins/games
    print("total win rate:---\n", win_rate)
    return win_rate

if __name__ == "__main__":
    individual_win_rate(result)
    total_win_rate(result)
