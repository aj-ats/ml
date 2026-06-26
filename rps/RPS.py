# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
"""
def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    return guess
"""
import random

def player(prev_play, opponent_history=[]):
    # Append the last play to history
    if prev_play:
        opponent_history.append(prev_play)

    # Dictionary to get the winning counter-move
    counter_move = {"R": "P", "P": "S", "S": "R"}

    # Start with a random choice if there is not enough history
    if len(opponent_history) < 4:
        return random.choice(["R", "P", "S"])

    # Look at the last few plays to identify a pattern
    # Here we look at the opponent's last play, but you can increase the history length for specific bots
    last_two = opponent_history[-3:]
    
    # Example predictive strategy: guess based on the most frequent move recently
    ideal_response = counter_move[prev_play]
    
    # If the opponent is very repetitive, we can switch to counter their last 3 moves
    if len(opponent_history) > 10:
        most_frequent = max(set(last_two), key=last_two.count)
        return counter_move[most_frequent]

    return ideal_response
