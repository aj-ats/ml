import random

MOVES = ("R", "P", "S")
BEAT = {
        "R": "P",
        "P": "S",
        "S": "R"
        }  # move that beats key

def _reward(my_move, opp_move):
    if my_move == opp_move:
        return 0
    return 1 if BEAT[opp_move] == my_move else -1


def player(
    prev_play,
    mem={
        "my_history": [],
        "my_trans": {a + b: 0 for a in MOVES for b in MOVES},  # emulate abbey view of us
        "Q": {},  # Q[(my_prev, abbey_pred)][action]
        "last_state": None,
        "last_action": None,
        "eps": 0.25,
    },
):
    # match reset
    if prev_play == "":
        mem["my_history"].clear()
        for k in mem["my_trans"]:
            mem["my_trans"][k] = 0
        mem["Q"].clear()
        mem["last_state"] = None
        mem["last_action"] = None
        mem["eps"] = 0.25
        first = random.choice(MOVES)
        mem["my_history"].append(first)
        mem["last_action"] = first
        return first

    # update transition counts from our last 2 moves (what abbey tracks)
    if len(mem["my_history"]) >= 2:
        pair = mem["my_history"][-2] + mem["my_history"][-1]
        mem["my_trans"][pair] += 1

    # current state = (our previous move, abbey's likely current move)
    my_prev = mem["my_history"][-1] if mem["my_history"] else "R"
    options = [my_prev + m for m in MOVES]
    predicted_our_next = max(options, key=lambda k: mem["my_trans"][k])[-1]
    abbey_pred = BEAT[predicted_our_next]
    state = (my_prev, abbey_pred)

    if state not in mem["Q"]:
        mem["Q"][state] = {m: 0.0 for m in MOVES}

    # Q update from last round result
    if mem["last_state"] is not None and prev_play in MOVES:
        if mem["last_state"] not in mem["Q"]:
            mem["Q"][mem["last_state"]] = {m: 0.0 for m in MOVES}
        r = _reward(mem["last_action"], prev_play)
        alpha, gamma = 0.30, 0.90
        old = mem["Q"][mem["last_state"]][mem["last_action"]]
        nxt = max(mem["Q"][state].values())
        mem["Q"][mem["last_state"]][mem["last_action"]] = old + alpha * (r + gamma * nxt - old)

    # epsilon-greedy action selection
    if random.random() < mem["eps"]:
        action = random.choice(MOVES)
    else:
        action = max(MOVES, key=lambda m: mem["Q"][state][m])

    mem["eps"] = max(0.02, mem["eps"] * 0.999)  # decay exploration
    mem["last_state"] = state
    mem["last_action"] = action
    mem["my_history"].append(action)
    return action
