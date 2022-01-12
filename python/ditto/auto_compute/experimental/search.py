from itertools import combinations
from math import gcd
from random import sample, choice

from sympy import ntheory
from tqdm import tqdm

from expr import *
from translator import *


class SearchPolicy: pass


# TODO: sometimes random search cannot find any valid candidates
# TODO: implement multi-round search
class RandomSearchPolicy(SearchPolicy):
    def __init__(self, init_state: State, max_steps=8, size_limit=8, max_sample_per_rule=1):
        super().__init__()
        self.init_state = init_state
        self.max_steps = max_steps
        self.size_limit = size_limit
        self.max_sample_per_rule = max_sample_per_rule

        # ping-pong buffering
        self.states_buf_ping = list()
        self.states_buf_pong = list()

    # TODO: rename parameter ``n_trials``
    def search(self, n_trials=1000) -> "list[State]":
        output_states = list()

        param_cnt = self.init_state.count_params() / 1e6
        if param_cnt <= self.size_limit:
            output_states.append(self.init_state)
        
        self.states_buf_ping = [self.init_state]
        
        for step in range(self.max_steps):
            self.states_buf_pong.clear()
            
            for state in tqdm(self.states_buf_ping, desc=f'step {step}'):
                # try each rule and sample at most ``max_sample_per_rule`` candidates
                for rule in self._RULES:
                    success, new_states = rule(self, state)
                    if not success: continue
                    self.states_buf_pong.extend(new_states)

            # output new states who param size is under limit
            for state in self.states_buf_pong:
                param_cnt = state.count_params() / 1e6
                if param_cnt <= self.size_limit:
                    output_states.append(state)
            
            # skip remaining steps if we have found enough candidates
            if len(output_states) >= n_trials:
                break

            # swap ping-pong buffers
            ping, pong = self.states_buf_ping, self.states_buf_pong
            self.states_buf_ping, self.states_buf_pong = pong, ping

        return output_states

    def _sample(self, arr):
        return sample(arr, min(len(arr), self.max_sample_per_rule))

    def _random_split_rule(self, state: State) -> "tuple[bool, list[State]]": 
        split_choices = list()
        for st_id, stage in enumerate(state.stages):
            for iter in stage.iters:
                spl_fac_choices = list(ntheory.divisors(iter.range.extent))[1:]
                if iter.range.extent > 1:
                    spl_fac_choices.remove(iter.range.extent)
                if len(spl_fac_choices) == 0: continue
                split_choices.append((st_id, iter, spl_fac_choices))

        if len(split_choices) == 0: return False, list()
        
        sampled_params = list()
        for st_id, it, spl_fac_choices in self._sample(split_choices):
            lengths = self._sample(spl_fac_choices)
            sampled_params.extend([(st_id, it, [length]) for length in lengths])
        sampled_params = self._sample(sampled_params)
        
        try:
            new_states = [state.split(*param)[0] for param in sampled_params]
            return True, new_states
        except AssertionError:
            return False, list()

    def _random_group_rule(self, state: State) -> "tuple[bool, list[State]]":
        group_choices = list()
        for st_id, stage in enumerate(state.stages):
            for it1, it2 in combinations(stage.iters, 2):
                ext_gcd = gcd(it1.range.extent, it2.range.extent)
                n_grp_choices = list(ntheory.divisors(ext_gcd))[1:]
                if len(n_grp_choices) == 0: continue
                group_choices.append((st_id, [it1, it2], n_grp_choices))

        if len(group_choices) == 0: return False, list()
        
        sampled_params = list()
        for st_id, iters, n_grp_choices in self._sample(group_choices):
            num_groups = self._sample(n_grp_choices)
            sampled_params.extend([(st_id, iters, n_grp) for n_grp in num_groups])
        sampled_params = self._sample(sampled_params)

        try:
            new_states = [state.group(*param)[0] for param in sampled_params]
            return True, new_states
        except AssertionError:
            return False, list()

    def _random_elim_rule(self, state: State) -> "tuple[bool, list[State]]":
        elim_choices = [
            (st_id, iter) 
            for st_id, stage in enumerate(state.stages) 
            for iter in stage.iters
        ]
        
        if len(elim_choices) == 0: return False, list()
        
        sampled_params = self._sample(elim_choices)

        try:
            new_states = [state.eliminate(*param)[0] for param in sampled_params]
            return True, new_states
        except AssertionError:
            return False, list()

    def _random_wshare_rule(self, state: State) -> "tuple[bool, list[State]]":
        wshare_choices = list()
        for st_id, stage in enumerate(state.stages):
            for ta_id, tacc in enumerate(stage.compute_expr.operands):
                if not tacc.is_weight: continue
                for dim_id in range(tacc.tensor.ndim):
                    wshare_choices.append((st_id, ta_id, dim_id))
        
        if len(wshare_choices) == 0: return False, list()

        sampled_params = self._sample(wshare_choices)

        try:
            new_states = [state.weight_share(*param)[0] for param in sampled_params]
            return True, new_states
        except AssertionError:
            return False, list()

    def _random_swin_rule(self, state: State) -> "tuple[bool, list[State]]":
        # TODO: find a better way to specify search space
        STRIDE_LIMIT = 10
        KSIZE_CHOICES = [3, 5, 7, 9, 11]

        swin_choices = list()
        for st_id, stage in enumerate(state.stages):
            compute_expr = stage.compute_expr

            # we only consider operators of the form Y=X*W
            if len(compute_expr.operands) != 2: continue
            if sum(tacc.is_weight for tacc in compute_expr.operands) != 1: continue
            
            for ta_id, tacc in enumerate(compute_expr.operands):
                if tacc.is_weight: continue
                for dim_id, dim_size in enumerate(tacc.tensor.shape):
                    stride_choices = list(ntheory.divisors(dim_size))[1:]
                    stride_choices = [s for s in stride_choices if s < STRIDE_LIMIT]
                    if len(stride_choices) == 0: continue
                    swin_choices.append((st_id, ta_id, dim_id, stride_choices))
        
        if len(swin_choices) == 0: return False, list()

        sampled_params = list()
        for st_id, ta_id, dim_id, stride_choices in self._sample(swin_choices):
            strides = self._sample(stride_choices)
            ksize = choice(KSIZE_CHOICES)
            sampled_params.extend([(st_id, ta_id, dim_id, ksize, stride) for stride in strides])
        sampled_params = self._sample(sampled_params)

        try:
            new_states = [state.sliding_window(*param)[0] for param in sampled_params]
            return True, new_states
        except AssertionError:
            return False, list()

    # TODO: find a better way to group these functions
    _RULES = [
        _random_split_rule,
        _random_group_rule,
        _random_elim_rule,
        _random_wshare_rule,
        _random_swin_rule,
    ]
