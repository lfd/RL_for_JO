def action_penalise_invalid(action, possible_actions, all_combinations):
    action_pair = all_combinations[action]
    if action_pair not in possible_actions:
        return None, -1
    return action_pair, 0

def action_regular(action, possible_actions, all_combinations):
    action_pair = all_combinations[action]
    return action_pair, 0

def action_possible_idx(action, possible_actions, all_combinations):
    action_pair = possible_actions[action]
    return action_pair, 0

