from search import *
from test_graph import test_graph1, test_graph2


def test_random_search():
    init_state = test_graph1()
    policy = RandomSearchPolicy(init_state, max_steps=8)
    out_states = policy.search()

    print(f'Found {len(out_states)} candidate operators.')

    for idx, stt in enumerate(sample(out_states, 5)):
        print(f'---------\nCandidate {idx}:')
        print(f'Compute: {stt.stages[0].compute_expr}')


def test_random_search_and_translation():
    init_state = test_graph1()
    policy = RandomSearchPolicy(init_state, max_steps=8, size_limit=2)
    out_states = policy.search()

    print(f'Found {len(out_states)} candidate operators.')
    
    for idx, stt in enumerate(sample(out_states, min(len(out_states), 5))):
        print(f'---------\nCandidate {idx}:')
        print(f'Compute: {stt.stages[0].compute_expr}')
        print('Iters:', [f'{it.name}({it.range.extent})' for it in stt.stages[0].iters])
        mod = StateEinsum.from_state(stt)
        inputs = {
            k: torch.randn(*shape)
            for k, shape in mod.input_cfg.items()
        }
        output = mod(**inputs)
        print(f'Einsum: {mod.sorted_mods[0].einsum_eq}')
        print(f'Inputs: {mod.input_cfg}')
        print(f'Outputs: { {k: v.shape for k, v in output.items()} }')


# TODO: wrong output dim order ([64, 7, 7] -> [7, 64, 7])
def test_random_search_and_translation2():
    init_state = test_graph2()
    policy = RandomSearchPolicy(init_state, max_steps=10, size_limit=16, max_sample_per_rule=1)
    out_states = policy.search()

    print(f'Found {len(out_states)} candidate operators.')
    
    for idx, stt in enumerate(sample(out_states, min(len(out_states), 5))):
        print(f'---------\nCandidate {idx}:')
        print(f'Compute: {stt.stages[0].compute_expr}')
        print('Iters:', [f'{it.name}({it.range.extent})' for it in stt.stages[0].iters])
        mod = StateEinsum.from_state(stt)
        inputs = {
            k: torch.randn(*shape)
            for k, shape in mod.input_cfg.items()
        }
        output = mod(**inputs)
        print(f'Einsum: {mod.sorted_mods[0].einsum_eq}')
        print(f'Inputs: {mod.input_cfg}')
        print(f'Outputs: { {k: v.shape for k, v in output.items()} }')


if __name__ == '__main__':
    test_random_search()
    test_random_search_and_translation()
    test_random_search_and_translation2()
