BEAT = {"R": "P", "P": "S", "S": "R"}
MOVES = ("R", "P", "S")


def _new_state():
    return {
        "opp": [],
        "me": [],
        "scores": {
            "quincy": 0,
            "kris": 0,
            "mrugesh": 0,
            "abbey": 0,
            "ngram3": 0,
        },
        "last_predictions": {},
        "my_pair": {a + b: 0 for a in MOVES for b in MOVES},
        "opp_four": {
            a + b + c + d: 0
            for a in MOVES
            for b in MOVES
            for c in MOVES
            for d in MOVES
        },
        "quincy_phase_hits": [0, 0, 0, 0, 0],
    }


def _most_frequent_move(moves):
    counts = {m: 0 for m in MOVES}
    for m in moves:
        if m in counts:
            counts[m] += 1
    return max(MOVES, key=lambda m: counts[m])


def player(prev_play, state={"data": _new_state()}):
    data = state["data"]

    if prev_play == "":
        state["data"] = _new_state()
        data = state["data"]
        first = "R"
        data["me"].append(first)
        return first

    data["opp"].append(prev_play)

    if len(data["me"]) >= 2:
        data["my_pair"][data["me"][-2] + data["me"][-1]] += 1

    if len(data["opp"]) >= 4:
        data["opp_four"]["".join(data["opp"][-4:])] += 1

    for name, prediction in data["last_predictions"].items():
        if prediction == prev_play:
            data["scores"][name] += 3
        else:
            data["scores"][name] -= 1

    predictions = {}

    quincy_cycle = ["R", "R", "P", "P", "S"]
    round_idx = len(data["opp"]) - 1
    for offset in range(5):
        if quincy_cycle[(round_idx + offset) % 5] == prev_play:
            data["quincy_phase_hits"][offset] += 1
    best_offset = max(range(5), key=lambda i: data["quincy_phase_hits"][i])
    predictions["quincy"] = quincy_cycle[(len(data["opp"]) + best_offset) % 5]

    if data["me"]:
        predictions["kris"] = BEAT[data["me"][-1]]
    else:
        predictions["kris"] = "P"

    last_ten_me = data["me"][-10:] if data["me"] else ["R"]
    predictions["mrugesh"] = BEAT[_most_frequent_move(last_ten_me)]

    if data["me"]:
        prev_me = data["me"][-1]
        options = [prev_me + m for m in MOVES]
        predicted_my_next = max(options, key=lambda key: data["my_pair"][key])[-1]
        predictions["abbey"] = BEAT[predicted_my_next]
    else:
        predictions["abbey"] = "P"

    if len(data["opp"]) >= 3:
        key3 = "".join(data["opp"][-3:])
        options = [key3 + m for m in MOVES]
        predictions["ngram3"] = max(options, key=lambda key: data["opp_four"][key])[-1]
    else:
        predictions["ngram3"] = predictions["quincy"]

    priority = {"abbey": 5, "kris": 4, "mrugesh": 3, "quincy": 2, "ngram3": 1}
    best_model = max(
        predictions.keys(), key=lambda name: (data["scores"][name], priority[name])
    )
    predicted_opp = predictions[best_model]
    guess = BEAT[predicted_opp]

    data["last_predictions"] = predictions
    data["me"].append(guess)
    return guess
