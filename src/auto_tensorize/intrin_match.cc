#include <auto_tensorize/intrin_match.h>

namespace ditto {
namespace auto_tensorize {

Array<IterVar> IntrinMatcher::_extract_axes_from_op(
    const ComputeOpNode* op,
    bool include_reduce
) {
  Array<IterVar> axes;
  for (IterVar axis : op->axis) axes.push_back(axis);
  for (IterVar axis : op->reduce_axis) axes.push_back(axis);
  return std::move(axes);
}

Map<IterVar, Range> IntrinMatcher::_infer_bounds(Operation out) {
  Array<Operation> out_ops{out};
  Schedule sch = create_schedule(out_ops);
  sch = sch.normalize();
  Map<IterVar, Range> bounds = InferBound(sch);
  return bounds;
}

Array<Array<IterVar> > IntrinMatcher::match(Tensor target, Operation capsule) {
  auto target_bounds = _infer_bounds(target->op);
  auto intrin_bounds = _infer_bounds(capsule);
  bool success = _match(target, capsule, target_bounds, intrin_bounds);
  return success ? this->results : Array<Array<IterVar> >();
}

bool IntrinMatcher::_match(
    Tensor target,
    Operation capsule,
    Map<IterVar, Range> target_bounds,
    Map<IterVar, Range> intrin_bounds
) {
    size_t ivd = 0;
    // assume intrin->value_index is 0

    if (target->dtype != capsule->output_dtype(ivd)) {
        return false;
    }
    const ComputeOpNode* target_op = target->op.as<ComputeOpNode>();
    const ComputeOpNode* intrin_op = capsule.as<ComputeOpNode>();

    if (intrin_op == nullptr) {
        const PlaceholderOpNode* intrin_op = capsule.as<PlaceholderOpNode>();
        CHECK(intrin_op != nullptr) << "Intrin is neither from a ComputeOp "
                                    << "nor a PlaceholderOp .";
        return true;
    }

    const PrimExpr target_expr = target_op->body[target->value_index];
    const PrimExpr intrin_expr = intrin_op->body[ivd];

    Array<IterVar> intrin_axes = _extract_axes_from_op(intrin_op);
    Array<IterVar> target_axes = _extract_axes_from_op(target_op);
    CapsuleExprMatcher expr_matcher(buffer_map);
    Array<IterVarMap> possible_index_mappings;
    possible_index_mappings = expr_matcher.match(target_expr, intrin_expr, target_axes,
                                                 intrin_axes, target_bounds, intrin_bounds);
    if (possible_index_mappings.size() == 0) {  // expr matching failed
        return false;
    }

    for(auto pim : possible_index_mappings){
        IterVarMap index_mapping_rev;
        Array<IterVar> res;
        for(auto it : pim){
            index_mapping_rev.Set(it.second, it.first);
        }
        for(auto it : intrin_axes){
            res.push_back(index_mapping_rev[it]);
        }
        results.push_back(res);
    }

    return true;
}

void CapsuleExprMatcher::extract_indices(
    PrimExpr target, 
    PrimExpr intrin,
    Array<Array<PrimExpr>>& target_indices,
    Array<Array<PrimExpr>>& intrin_indices
) {
    VisitExpr(target, intrin);
    _check_intrin_const_dim();
    for (Array<PrimExpr> i : this->target_indices) {
        target_indices.push_back(i);
    }
    for (Array<PrimExpr> i : this->intrin_indices) {
        intrin_indices.push_back(i);
    }
}

void CapsuleExprMatcher::_check_intrin_const_dim() {
    bool has_const_dim = false;
    for (auto index : intrin_indices) {
        for (auto i : index) {
            if (is_const_int(i)) {
                has_const_dim = true;
            }
        }
    }
    CHECK(!has_const_dim);
}

Array<IterVarMap> CapsuleExprMatcher::match(
    PrimExpr target, PrimExpr intrin,
    Array<IterVar>& target_axes,
    Array<IterVar>& intrin_axes,
    Map<IterVar, Range> target_bounds,
    Map<IterVar, Range> intrin_bounds
) {
    bool structure_match = VisitExpr(target, intrin);  // buffer and op
    _check_intrin_const_dim();
    if (!structure_match) {
        return Array<IterVarMap>();
    }
    Array<IterVarMap> possible_index_mappings;
    IndexExprMatcher index_matcher;
    possible_index_mappings = index_matcher.match(target_indices, intrin_indices, target_axes,
                                                  intrin_axes, target_bounds, intrin_bounds);
    if (possible_index_mappings.size() == 0) {
        return Array<IterVarMap>();
    } else {
        return possible_index_mappings;
    }
}

Array<IterVarMap> enumerate_mappings(
    Array<IterVar> target_axes, 
    Array<IterVar> intrin_axes
) {
    size_t n = target_axes.size(), r = intrin_axes.size();
    if (n < r) return Array<IterVarMap>();

    std::vector<IterVar> target_axes_vec;
    for (auto axis : target_axes) target_axes_vec.push_back(axis);

    auto comp = [](const IterVar& x, const IterVar& y) { return x.get() < y.get(); };
    std::sort(target_axes_vec.begin(), target_axes_vec.end(), comp);

    Array<IterVarMap> all_itervar_mappings;

    std::vector<bool> selector(n);
    std::fill(selector.begin(), selector.begin() + r, true);

    do {
        std::vector<IterVar> comb;
        for (size_t i = 0; i < n; ++i) {
            if (!selector[i]) continue;
            comb.push_back(target_axes_vec[i]);
        }

        do {
            IterVarMap itervar_map;
            for (size_t i = 0; i < r; ++i) {
                // Need to match the axis type
                if (comb[i]->iter_type == intrin_axes[i]->iter_type)
                    itervar_map.Set(comb[i], intrin_axes[i]);
            }
            all_itervar_mappings.push_back(itervar_map);
        } while (std::next_permutation(comb.begin(), comb.end(), comp));

    } while (std::prev_permutation(selector.begin(), selector.end()));

    return std::move(all_itervar_mappings);
}

bool IndexExprMatcher::_match_index(
    Array<PrimExpr> target_idx, 
    Array<PrimExpr> intrin_idx
) {
    CheckExprEqual check_equal(true, true);
    size_t n_dim_target = target_idx.size();
    size_t n_dim_intrin = intrin_idx.size();

    for (size_t j = 0; j < n_dim_intrin; ++j) {
        PrimExpr intrin_i = intrin_idx[j];
        bool i_matched = false;
        for (size_t k = 0; k < n_dim_target; ++k) {
            PrimExpr target_i = target_idx[k];
            // for relaxed matching, the order is important
            // target index is more general than intrin index
            if (check_equal(target_i, intrin_i)) {
                target_idx.Set(k, make_zero(target_idx[0].dtype()));
                i_matched = true;
                break;
            }
        }
        if (!i_matched) {
            return false;
        }
    }

    for (PrimExpr i : target_idx) {
        if (!is_const_int(i)) {
            return false;
        } else if (!i.as<IntImmNode>()->value == 0) {
            std::cout << "Warning: found a non-zero constant in target_idx" << std::endl;
            std::cout << "target_idx: " << target_idx << std::endl;
            std::cout << "intrin_idx: " << intrin_idx << std::endl;
        }
    }

    return true;
}

bool IndexExprMatcher::_match_indices(
    Array<Array<PrimExpr>> target_indices,
    Array<Array<PrimExpr>> intrin_indices
) {
    size_t n_indices_intrin = intrin_indices.size();

    for (size_t i = 0; i < n_indices_intrin; ++i) {
        Array<PrimExpr> target_idx = target_indices[i];
        Array<PrimExpr> intrin_idx = intrin_indices[i];

        if (!_match_index(target_idx, intrin_idx)) {
            return false;
        }
    }

    return true;
}

Array<Array<PrimExpr>> IndexExprMatcher::_rewrite_indices(
    Array<Array<PrimExpr>> indices,
    IterVarMap itervar_map,
    Map<IterVar, Range> target_bounds,
    Map<IterVar, Range> intrin_bounds
) {
    IterVarRewriter itervar_rewriter(itervar_map, target_bounds);
    size_t n_indices = indices.size();
    auto simplify = [](const PrimExpr& x) { return arith::Analyzer().Simplify(x); };

    for (size_t i = 0; i < n_indices; ++i) {
        Array<PrimExpr> idx = indices[i];
        size_t n_dim = idx.size();
        for (size_t j = 0; j < n_dim; ++j) {
            PrimExpr mod_i = simplify(itervar_rewriter.VisitExpr(idx[j]));
            idx.Set(j, mod_i);
        }
        indices.Set(i, idx);
    }
    return std::move(indices);
}

Array<IterVarMap> IndexExprMatcher::match(
    Array<Array<PrimExpr>> target_indices,
    Array<Array<PrimExpr>> intrin_indices,
    Array<IterVar>& target_axes,
    Array<IterVar>& intrin_axes,
    Map<IterVar, Range> target_bounds,
    Map<IterVar, Range> intrin_bounds
) {
    CHECK(target_indices.size() == intrin_indices.size());
    Array<IterVarMap> possible_itervar_mappings;
    Array<IterVarMap> all_itervar_mappings = enumerate_mappings(target_axes, intrin_axes);

    for (IterVarMap itervar_map : all_itervar_mappings) {
        auto modified_target_indices =
            _rewrite_indices(target_indices, itervar_map, target_bounds, intrin_bounds);

        if (_match_indices(modified_target_indices, intrin_indices)) {
            possible_itervar_mappings.push_back(itervar_map);
        }
    }
    return std::move(possible_itervar_mappings);
}

TVM_REGISTER_GLOBAL("ditto.auto_tensorize.MatchIntrinsic")
    .set_body_typed([](Tensor target, Operation capsule) {
        return IntrinMatcher().match(target, capsule);
    });

} // namespace auto_tensorize
} // namespace ditto
