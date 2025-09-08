import copy
import numpy as np

def aggregate_by_influence_and_quality_blended(global_state: dict,
                                               client_states: list[dict],
                                               group_prefixes: list[str],
                                               group_keys_list: list[list[str]],
                                               group_influences: list[float],
                                               data_quality_scores: list[float],
                                               alpha: float = 0.5):
    """
    Sort groups by influence (desc); for each group, pick best remaining client by quality,
    blend that group's params into the global state with factor alpha.
    Returns (new_state, mapping dict prefix->client_id).
    """
    new_state = copy.deepcopy(global_state)
    n_clients = len(client_states)
    order = sorted(range(len(group_prefixes)), key=lambda i: group_influences[i], reverse=True)
    available = set(range(n_clients))
    mapping: dict[str,int] = {}

    scores = np.array(data_quality_scores, dtype=float)
    for gi in order:
        if not available:
            available = set(range(n_clients))
        avail = list(available)
        best_idx = int(np.argmax(scores[avail]))
        best_client = avail[best_idx]

        for k in group_keys_list[gi]:
            gparam = global_state[k]
            cparam = client_states[best_client][k]
            new_state[k] = ((1.0 - alpha) * gparam + alpha * cparam).clone()

        mapping[group_prefixes[gi]] = best_client
        available.remove(best_client)

    return new_state, mapping