from .state_machine import State, StateMachnie


def design(layer):
    """
    Design a new layer compute according to
    the original layer compute

    Args:
    ---
    layer_state: ditto.auto_compute.graph.LayerState

    Returns:
    ---
    action
    """
    # TODO: implement the body
    # now, it just returns the original compute
    init_state = State(layer)
    fsm = StateMachnie([init_state])
    reachable_states = fsm.run()
    for i, state in enumerate(reachable_states):
        print(f"State {i} ==================")
        print(state.action_history)
        print()
    return 0


def auto_compute(layer_state, action):
    """ """
    return layer_state
