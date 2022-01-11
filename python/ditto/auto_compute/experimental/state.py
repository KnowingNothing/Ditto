from expr import *
from utils import disjoint_set_union


class State:
    def __init__(self, stages: "list[Stage]") -> None:
        self.stages: list[Stage] = stages
        self.tfm_steps: list[Step] = list()
        self.sync_groups: list[tuple[Iter]] = self._build_sync_group()

    @property
    def tensors(self):
        tensors = list(set().union(*[stg.tensors for stg in self.stages]))
        return tensors

    def count_params(self):
        param_count = 0
        for stage in self.stages: 
            for tacc in stage.compute_expr.operands:
                if not tacc.is_weight: continue
                param_count += tacc.tensor.numel
        return param_count

    def copy(self, tensor_map):
        new_stages = [stg.copy(tensor_map) for stg in self.stages]
        new_state = State(new_stages)
        new_state.tfm_steps = self.tfm_steps.copy()
        new_state.sync_groups = self.sync_groups.copy()
        return new_state

    """ sync group definition """

    def _build_sync_group(self):
        # find tensors that are accessed in multiple stages
        tensor_to_access: dict[Tensor, list[TensorAccess]] = dict()
        for stage in self.stages:
            accesses = stage.compute_expr.operands + [stage.compute_expr.output]
            for acc in accesses:
                tensor = acc.tensor
                if tensor not in tensor_to_access:
                    tensor_to_access[tensor] = list()
                tensor_to_access[tensor].append(acc)

        iv_grps: list[tuple[Iter]] = list()
        
        for tensor, accesses in tensor_to_access.items():
            if not tensor.morphable: continue
            if len(accesses) == 1: continue

            iv_list = list()
            for acc in accesses:
                itervars = list()
                for expr in acc.md_idx_expr.idx_exprs:
                    assert len(expr.coeff_map) == 1
                    assert expr.const_term == 0
                    itervars.append(list(expr.coeff_map.keys())[0])
                iv_list.append(itervars)

            iv_grps.extend(list(zip(*iv_list)))

        sync_groups = disjoint_set_union(iv_grps)
        return sync_groups

    def get_synched_iters(self, iter: Iter):
        iter_exists = any([iter in stage.iters for stage in self.stages])
        assert iter_exists, f"Iterator {iter} not found"
        for sync_grp in self.sync_groups:
            if iter in sync_grp: return sync_grp
        return (iter,)

    """ program transforms """
    # TODO: support fuse?

    def split(self, stage_id, iter: Iter, lengths: "list[int]") -> "tuple[State, list[Iter]]":
        stage = self.stages[stage_id]
        iter_id = stage.iters.index(iter)
        step = SplitStep(stage_id, iter_id, iter.range.extent, lengths)
        new_state, split_iters = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, split_iters

    # TODO: reorder step is not used for now
    def reorder(self, stage_id, order: "list[Iter]") -> "tuple[State, None]":
        stage = self.stages[stage_id]
        after_ids = [stage.iters.index(it) for it in order]
        step = ReorderStep(stage_id, after_ids)
        new_state, none = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, none

    """ model transforms """

    def eliminate(self, stage_id, iter: Iter) -> "tuple[State, None]":
        stage = self.stages[stage_id]
        iter_id = stage.iters.index(iter)
        step = ElimStep(stage_id, iter_id)
        new_state, none = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, none

    def iter_share(self, stage_id, iters: "list[Iter]") -> "tuple[State, Iter]":
        stage = self.stages[stage_id]
        shared_iter_ids = [stage.iters.index(iter) for iter in iters]
        step = IShareStep(stage_id, shared_iter_ids)
        new_state, shared_iter = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, shared_iter

    def group(self, stage_id, iters: "list[Iter]", num_groups) -> "tuple[State, Iter]":
        stage = self.stages[stage_id]
        grouped_iter_ids = [stage.iters.index(iter) for iter in iters]
        step = GroupStep(stage_id, grouped_iter_ids, num_groups)
        new_state, grouped_iter = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, grouped_iter

    def weight_share(self, stage_id, operand_id, dim_id) -> "tuple[State, None]":
        step = WShareStep(stage_id, operand_id, dim_id)
        new_state, none = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, none

    def sliding_window(self, stage_id, operand_id, dim_id, kernel_size, stride) -> "tuple[State, Iter]":
        step = SWinStep(stage_id, operand_id, dim_id, kernel_size, stride)
        new_state, swin_iter = step.apply_to_state(self)
        new_state.tfm_steps.append(step)
        return new_state, swin_iter


class Step: 
    def __init__(self, stage_id):
        self.stage_id = stage_id

    # tensor map: old tensor -> new tensor (after transform)
    # avoid creating multiple copies of the same tensor
    def _build_tensor_map(self, state):
        tensor_map = {ts: ts.copy() for ts in state.tensors}
        # a newly created tensor maps to itself
        tensor_map.update(dict(zip(tensor_map.values(), tensor_map.values())))
        return tensor_map

    def apply_to_state(self, state: State) -> "tuple[State, object]":
        raise NotImplementedError


# TODO: simplify the implementation of ``Step``s
# TODO: add more comments


class ReorderStep(Step):
    def __init__(self, stage_id, after_ids: "list[int]"):
        super().__init__(stage_id)
        self.after_ids = after_ids

    def apply_to_state(self, state: State) -> "tuple[State, None]":
        tensor_map = self._build_tensor_map(state)
        new_state = state.copy(tensor_map)
        stage = new_state.stages[self.stage_id]
        assert len(self.after_ids) == len(stage.iters)
        reordered_iters = [stage.iters[i] for i in self.after_ids]
        new_stage = Stage(reordered_iters, stage.compute_expr.copy(tensor_map))
        new_state.stages[self.stage_id] = new_stage
        return new_state, None


class SplitStep(Step):
    def __init__(self, stage_id, iter_id, extent: int, lengths: "list[int]"):
        super().__init__(stage_id)
        self.iter_id = iter_id  # The id of the iter to split.
        self.extent = extent  # The extent length of the axis to split
        self.lengths = lengths # The split factors
        assert self.extent % np.prod(self.lengths) == 0

    def apply_to_state(self, state: State) -> "tuple[State, list[Iter]]":
        tensor_map = self._build_tensor_map(state)

        new_state = state.copy(tensor_map)
        orig_stage = new_state.stages[self.stage_id]
        orig_iter = orig_stage.iters[self.iter_id]
        assert self.extent == orig_iter.range.extent

        synced_iters = new_state.get_synched_iters(orig_iter)
        
        synced_stages = list()
        for iter in synced_iters:
            for stage in new_state.stages:
                if iter in stage.iters:
                    synced_stages.append(stage)
                    break

        assert len(set(synced_stages)) == len(synced_stages)
        assert all(stage is not None for stage in synced_stages)

        new_exts = [self.extent // np.prod(self.lengths), *self.lengths]
        new_mins = [0 for _ in range(len(self.lengths))] + [orig_iter.range.min]
        new_ranges = [(m, e) for m, e in zip(new_mins, new_exts)]
        strides = list(reversed([1] + np.cumprod(list(reversed(new_exts))).tolist()[:-1]))

        ret_spl_iters = None

        tensor_morph_map = dict()

        for iter, stage in zip(synced_iters, synced_stages):
            assert stage is not None
            iter_id = stage.iters.index(iter)
            spl_iters = [Iter(f"{iter.name}.{i}", Range(*rg)) for i, rg in enumerate(new_ranges)]
            if iter is orig_iter: ret_spl_iters = spl_iters
            new_iters = stage.iters[:iter_id] + spl_iters + stage.iters[iter_id+1:]

            new_accesses = list()
            for access in [stage.compute_expr.output, *stage.compute_expr.operands]:
                if iter not in access.md_idx_expr.iters: 
                    new_access = access.copy(tensor_map)
                elif access.morphable:
                    dim_id = [iter in expr.iters for expr in access.md_idx_expr.idx_exprs].index(True)
                    if access.tensor not in tensor_morph_map:
                        new_tensor = access.tensor.copy().split_dim(dim_id, self.lengths)
                        tensor_morph_map[access.tensor] = new_tensor
                    new_tensor = tensor_morph_map[access.tensor]
                    new_idxexp = access.md_idx_expr.split_iter(iter, spl_iters)
                    new_access = TensorAccess(new_tensor, new_idxexp)
                else:  # non-morphable
                    replace_idx_expr = IndexExpr(coeff_map=OrderedDict(zip(spl_iters, strides)))
                    new_idxexp = access.md_idx_expr.replace_iter(iter, replace_idx_expr)
                    new_access = TensorAccess(tensor_map[access.tensor], new_idxexp)
                new_accesses.append(new_access)
                access.detach()
                
            new_output, *new_operands = new_accesses
            new_stage = Stage(new_iters, ComputeExpr(new_output, new_operands))
            new_state.stages[new_state.stages.index(stage)] = new_stage

        new_state.sync_groups = new_state._build_sync_group()

        assert ret_spl_iters is not None
        return new_state, ret_spl_iters


class ElimStep(Step):
    def __init__(self, stage_id, iter_id):
        super().__init__(stage_id)
        self.iter_id = iter_id
    
    def apply_to_state(self, state: State) -> "tuple[State, None]":
        tensor_map = self._build_tensor_map(state)

        new_state = state.copy(tensor_map)
        orig_stage = new_state.stages[self.stage_id]
        orig_iter = orig_stage.iters[self.iter_id]

        synced_iters = new_state.get_synched_iters(orig_iter)
        
        synced_stages = list()
        for iter in synced_iters:
            for stage in new_state.stages:
                if iter in stage.iters:
                    synced_stages.append(stage)
                    break
        
        assert len(set(synced_stages)) == len(synced_stages)
        assert all(stage is not None for stage in synced_stages)

        tensor_morph_map = dict()
        
        for iter, stage in zip(synced_iters, synced_stages):
            assert stage is not None
            iter_id = stage.iters.index(iter)
            new_iters = stage.iters[:iter_id] + stage.iters[iter_id+1:]
            
            new_accesses = list()
            for access in [stage.compute_expr.output, *stage.compute_expr.operands]:
                if iter not in access.md_idx_expr.iters: 
                    new_access = access.copy(tensor_map)
                elif access.morphable:
                    dim_id = [iter in expr.iters for expr in access.md_idx_expr.idx_exprs].index(True)
                    if access.tensor not in tensor_morph_map:
                        new_tensor = access.tensor.copy().elim_dim(dim_id)
                        tensor_morph_map[access.tensor] = new_tensor
                    new_tensor = tensor_morph_map[access.tensor]
                    new_idxexp = access.md_idx_expr.delete_iter(iter)
                    new_access = TensorAccess(new_tensor, new_idxexp)
                else: 
                    new_idxexp: MultiDimIndexExpr = access.md_idx_expr.delete_iter(iter)

                    # avoid redundancy in non-morphable tensors
                    # TODO: handle more complicated cases (e.g., strided access)
                    for idx_expr in new_idxexp.idx_exprs:
                        if len(idx_expr.coeff_map) == 0 and idx_expr.const_term == 0:
                            raise AssertionError

                    new_access = TensorAccess(tensor_map[access.tensor], new_idxexp)
                new_accesses.append(new_access)
                access.detach()
            
            new_output, *new_operands = new_accesses
            new_stage = Stage(new_iters, ComputeExpr(new_output, new_operands))
            new_state.stages[new_state.stages.index(stage)] = new_stage

        new_state.sync_groups = new_state._build_sync_group()
        return new_state, None


class WShareStep(Step):
    def __init__(self, stage_id, operand_id, dim_id):
        super().__init__(stage_id)
        self.operand_id = operand_id
        self.dim_id = dim_id

    def apply_to_state(self, state: State) -> "tuple[State, None]":
        tensor_map = self._build_tensor_map(state)

        new_state = state.copy(tensor_map)
        stage = new_state.stages[self.stage_id]
        taccess = stage.compute_expr.operands[self.operand_id]

        # weight sharing only apply to weights (morphable + leaf)
        assert taccess.is_weight

        iter: Iter = list(taccess.md_idx_expr.idx_exprs[self.dim_id].coeff_map.keys())[0]
        assert not iter.reduce  # only share weights along spatial dimensions
        assert len(new_state.get_synched_iters(iter)) == 1
        
        # avoid redundancy in the output tensor
        assert any(iter in tacc.md_idx_expr.iters for tacc in stage.compute_expr.operands if tacc is not taccess)

        new_accesses = list()
        for access in [stage.compute_expr.output, *stage.compute_expr.operands]:
            if access is taccess:
                new_tensor = access.tensor.copy().elim_dim(self.dim_id)
                new_idxexp = access.md_idx_expr.delete_iter(iter)
                new_access = TensorAccess(new_tensor, new_idxexp)
            else:
                new_access = access.copy(tensor_map)
            new_accesses.append(new_access)
            access.detach()
        
        new_output, *new_operands = new_accesses
        new_stage = Stage(stage.iters.copy(), ComputeExpr(new_output, new_operands))
        new_state.stages[self.stage_id] = new_stage

        return new_state, None


class IShareStep(Step):
    def __init__(self, stage_id, shared_iter_ids: list):
        super().__init__(stage_id)
        self.shared_iter_ids = shared_iter_ids
        assert len(self.shared_iter_ids) == 2
    
    def _apply_to_stage(self, state, stage, iters, tensor_morph_map, tensor_map) -> "tuple[Stage, Iter]":
        iter1, iter2 = iters
        shared_iter = Iter(
            ".".join([iter1.name, iter2.name, "share"]), 
            range=iter1.range.copy(), 
            reduce=iter1.reduce and iter2.reduce
        )
        
        new_iters = stage.iters.copy()
        new_iters.remove(iter1)
        new_iters.remove(iter2)
        new_iters = [shared_iter, *new_iters]

        replace_idx_expr = IndexExpr(coeff_map=OrderedDict({shared_iter: 1}))

        new_accesses = list()
        for access in [stage.compute_expr.output, *stage.compute_expr.operands]:
            if not any(iter in access.md_idx_expr.iters for iter in [iter1, iter2]):
                new_access = access.copy(tensor_map)
            elif access.morphable:
                if all(iter in access.md_idx_expr.iters for iter in [iter1, iter2]):
                    iter1_dim_id = [iter1 in expr.iters for expr in access.md_idx_expr.idx_exprs].index(True)
                    iter2_dim_id = [iter2 in expr.iters for expr in access.md_idx_expr.idx_exprs].index(True)
                    dim_id = max(iter1_dim_id, iter2_dim_id)
                    iter_to_delele = iter1 if iter1_dim_id >= iter2_dim_id else iter2
                    iter_to_replace = iter2 if iter1_dim_id >= iter2_dim_id else iter1
                    if access.tensor not in tensor_morph_map:
                        new_tensor = access.tensor.copy().elim_dim(dim_id)
                        tensor_morph_map[access.tensor] = new_tensor
                    new_tensor = tensor_morph_map[access.tensor]
                    new_idxexp = access.md_idx_expr.delete_iter(iter_to_delele)
                    new_idxexp = new_idxexp.replace_iter(iter_to_replace, replace_idx_expr)
                else:  # only one of the shared iters appears in idx
                    new_tensor = tensor_map[access.tensor]
                    iter_to_replace = iter1 if iter1 in access.md_idx_expr.iters else iter2
                    new_idxexp = access.md_idx_expr.replace_iter(iter_to_replace, replace_idx_expr)
                new_access = TensorAccess(new_tensor, new_idxexp)
            else:  # non-morphable
                if all(iter in access.md_idx_expr.iters for iter in [iter1, iter2]):
                    new_idxexp = access.md_idx_expr.replace_iter(iter1, replace_idx_expr)
                    new_idxexp = new_idxexp.replace_iter(iter2, replace_idx_expr)
                elif iter1 in access.md_idx_expr.iters:
                    new_idxexp = access.md_idx_expr.replace_iter(iter1, replace_idx_expr)
                else:  # iter2 in access.md_idx_expr.iters
                    new_idxexp = access.md_idx_expr.replace_iter(iter2, replace_idx_expr)
                new_access = TensorAccess(tensor_map[access.tensor], new_idxexp)

            new_accesses.append(new_access)
            access.detach()

        new_output, *new_operands = new_accesses
        new_stage = Stage(new_iters, ComputeExpr(new_output, new_operands))
        return new_stage, shared_iter

    def apply_to_state(self, state: State) -> "tuple[State, Iter]":
        tensor_map = self._build_tensor_map(state)

        new_state = state.copy(tensor_map)
        cur_stage = new_state.stages[self.stage_id]
        iter1_id, iter2_id = self.shared_iter_ids
        iter1, iter2 = cur_stage.iters[iter1_id], cur_stage.iters[iter2_id]
        assert iter1.range.extent == iter2.range.extent
        assert iter1.range.min == iter2.range.min

        morphed_tensors: list[tuple[Tensor, TensorAccess]] = list()
        for access in [cur_stage.compute_expr.output, *cur_stage.compute_expr.operands]:
            if not access.morphable: continue
            if all(iter in access.md_idx_expr.iters for iter in [iter1, iter2]):
                morphed_tensors.append((access.tensor, access))
        
        assert len(morphed_tensors) <= 1, "morphing multiple tensors is not supported"
        
        other_affected_stages: list[tuple[Stage, TensorAccess]] = list()
        for tensor, __ in morphed_tensors:
            for access in tensor.accesses:
                assert access.compute_expr is not None
                access_stage = access.compute_expr.stage
                if access_stage is cur_stage: continue
                other_affected_stages.append((access_stage, access))

        modified_stages: dict[Stage, Stage] = dict()
        tensor_morph_map: dict[Tensor, Tensor] = dict()
        new_stage, shared_iter = self._apply_to_stage(
            new_state, cur_stage, [iter1, iter2], tensor_morph_map, tensor_map)
        modified_stages[cur_stage] = new_stage
        
        for stage, access in other_affected_stages:
            assert isinstance(access, TensorAccess)
            assert len(morphed_tensors) == 1
            __, cur_access = morphed_tensors[0]
            assert isinstance(cur_access, TensorAccess)
            iter1_dim_id = [iter1 in expr.iters for expr in cur_access.md_idx_expr.idx_exprs].index(True)
            iter2_dim_id = [iter2 in expr.iters for expr in cur_access.md_idx_expr.idx_exprs].index(True)
            target_iter1 = list(access.md_idx_expr.idx_exprs[iter1_dim_id].iters)[0]
            target_iter2 = list(access.md_idx_expr.idx_exprs[iter2_dim_id].iters)[0]
            new_stage, __ = self._apply_to_stage(new_state, stage, [target_iter1, target_iter2], tensor_morph_map, tensor_map)
            modified_stages[stage] = new_stage

        for stage, new_stage in modified_stages.items():
            new_state.stages[new_state.stages.index(stage)] = new_stage

        new_state.sync_groups = new_state._build_sync_group()
        return new_state, shared_iter


class SWinStep(Step):
    def __init__(self, stage_id, operand_id, dim_id, kernel_size: int, stride: int):
        super().__init__(stage_id)
        self.operand_id = operand_id
        self.dim_id = dim_id
        self.kernel_size = kernel_size
        self.stride = stride
    
    def apply_to_state(self, state: State) -> "tuple[State, Iter]":
        tensor_map = self._build_tensor_map(state)

        new_state = state.copy(tensor_map)
        stage = new_state.stages[self.stage_id]
        
        # Currently we only support Y = XW
        assert len(stage.compute_expr.operands) == 2
        act_acc = stage.compute_expr.operands[self.operand_id]
        wgt_acc = stage.compute_expr.operands[1 - self.operand_id]
        assert wgt_acc.is_weight

        idx = sum([iter.name.startswith(act_acc.tensor.name+".k") for iter in stage.iters])
        new_iter_name = act_acc.tensor.name + ".k" + str(idx)
        new_iter_range = Range(min=-(self.kernel_size//2), extent=self.kernel_size)
        new_iter = Iter(name=new_iter_name, range=new_iter_range, reduce=True)
        new_iters = stage.iters.copy() + [new_iter]

        act_acc.tensor.set_morphable(False)
        new_act_idx = act_acc.md_idx_expr.add_iter(new_iter, self.dim_id, self.stride)
        new_act_acc = TensorAccess(tensor_map[act_acc.tensor], new_act_idx)

        new_wgt_tensor = wgt_acc.tensor.copy().add_dim(self.kernel_size)
        new_wgt_idx = wgt_acc.md_idx_expr.add_iter(new_iter)
        new_wgt_acc = TensorAccess(new_wgt_tensor, new_wgt_idx)

        new_output = stage.compute_expr.output.copy(tensor_map)

        wgt_acc.detach()
        act_acc.detach()
        stage.compute_expr.output.detach()

        if self.operand_id == 0:
            new_operands = [new_act_acc, new_wgt_acc]
        else:
            new_operands = [new_wgt_acc, new_act_acc]
        new_stage = Stage(new_iters, ComputeExpr(new_output, new_operands))
        new_state.stages[self.stage_id] = new_stage
        new_state.sync_groups = new_state._build_sync_group()
        return new_state, new_iter


class GroupStep(Step):
    def __init__(self, stage_id, grouped_iter_ids: list, num_groups):
        super().__init__(stage_id)
        self.grouped_iter_ids = grouped_iter_ids
        self.num_groups = num_groups
    
    def apply_to_state(self, state: State) -> "tuple[State, Iter]":
        cur_stage = state.stages[self.stage_id]
        grouped_iters = [cur_stage.iters[it_id] for it_id in self.grouped_iter_ids]
        shared_iters = list()

        for iter in grouped_iters:
            iter_id = cur_stage.iters.index(iter)
            iter_ext = iter.range.extent
            assert iter_ext % self.num_groups == 0
            split_factor = [iter_ext // self.num_groups]
            split_step = SplitStep(self.stage_id, iter_id, iter_ext, split_factor)
            state, [outer_it, inner_it] = split_step.apply_to_state(state)
            
            # squeeze: eliminate inner iters with extent = 1
            if inner_it.range.extent == 1: 
                cur_stage = state.stages[self.stage_id]
                iter_id = cur_stage.iters.index(inner_it)
                elim_step = ElimStep(self.stage_id, iter_id)
                state, __ = elim_step.apply_to_state(state)

            cur_stage = state.stages[self.stage_id]
            shared_iters.append(outer_it)
        
        shared_iter_ids = [cur_stage.iters.index(it) for it in shared_iters]
        ishare_step = IShareStep(self.stage_id, shared_iter_ids)
        new_state, shared_iter = ishare_step.apply_to_state(state)

        return new_state, shared_iter
