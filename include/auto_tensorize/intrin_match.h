#pragma once

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/te/schedule_pass.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace tvm;
using namespace te;

namespace ditto {
namespace auto_tensorize {

typedef Map<IterVar, IterVar> IterVarMap;
typedef Map<DataProducer, DataProducer> BufferMap;

class IntrinMatcher : public Object {
  public:
    Array<Array<IterVar> > match(Tensor target, Operation capsule);

  private:
    Array<Array<IterVar> > results;
    BufferMap buffer_map; // used for DAG match, useless here
    bool _match(Tensor target, Operation capsule,
                Map<IterVar, Range> target_bounds, Map<IterVar, Range> intrin_bounds);
    Map<IterVar, Range> _infer_bounds(Operation out);
    Array<IterVar> _extract_axes_from_op(const ComputeOpNode* op, bool include_reduce = true);
    // bool _check_elemwise(const ComputeOpNode* op, Array<Array<PrimExpr>>& indices);
};

class CapsuleExprMatcher : public ExprFunctor<bool(const PrimExpr&, const PrimExpr&)> {
  public:
    using ExprFunctor::VisitExpr;
    CapsuleExprMatcher(BufferMap& bm) : buffer_map(bm){};
    Array<IterVarMap> match(PrimExpr target, PrimExpr intrin, Array<IterVar>& target_axes,
                            Array<IterVar>& intrin_axes, Map<IterVar, Range> target_bounds,
                            Map<IterVar, Range> intrin_bounds);
    void extract_indices(PrimExpr target, PrimExpr intrin, Array<Array<PrimExpr>>& target_indices,
                       Array<Array<PrimExpr>>& intrin_indices);

  private:
    BufferMap& buffer_map;
    Array<Array<PrimExpr>> target_indices;
    Array<Array<PrimExpr>> intrin_indices;
    void _check_intrin_const_dim();

  protected:
    using ExprFunctor::VisitExpr_;
#define MATCH(T)                     \
    const T* another = expr.as<T>(); \
    if (another == nullptr) {        \
        return false;                \
    }

    bool VisitExpr_(const VarNode* op, const PrimExpr& expr) override {
        MATCH(VarNode)
        return true;
    }

    bool VisitExpr_(const SizeVarNode* op, const PrimExpr& expr) override {
        MATCH(SizeVarNode)
        return op->name_hint == another->name_hint;
    }

    bool VisitExpr_(const LoadNode* op, const PrimExpr& expr) {
        MATCH(LoadNode)
        return VisitExpr(op->index, another->index) && 
               VisitExpr(op->predicate, another->predicate) &&
               VisitExpr(op->buffer_var, another->buffer_var);
    }

    bool VisitExpr_(const LetNode* op, const PrimExpr& expr) {
        MATCH(LetNode)
        return VisitExpr(op->var, another->var) && 
               VisitExpr(op->value, another->value) &&
               VisitExpr(op->body, another->body);
    }

    bool VisitExpr_(const ProducerLoadNode* op, const PrimExpr& expr) override {
        MATCH(ProducerLoadNode)

        // check and update buffer map
        CHECK(op->producer.as<TensorNode>() != nullptr);
        CHECK(another->producer.as<TensorNode>() != nullptr);

        if (!buffer_map.count(op->producer)) {
            buffer_map.Set(op->producer, another->producer);
        } else if (buffer_map[op->producer] != another->producer) {
            return false;
        }

        // save indices
        target_indices.push_back(op->indices);
        intrin_indices.push_back(another->indices);

        return true;
    }

    bool VisitExpr_(const CallNode* op, const PrimExpr& expr) override {
        MATCH(CallNode)
        if (op->op != another->op) {
            return false;
        }
        if (op->args.size() != another->args.size()) {
            return false;
        }
        for (size_t i = 0; i < op->args.size(); ++i) {
            if (!VisitExpr(op->args[i], another->args[i])) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool VisitBinary(const T* op, const PrimExpr& expr) {
        MATCH(T)
        return VisitExpr(op->a, another->a) && VisitExpr(op->b, another->b);
    }

    bool VisitExpr_(const AddNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const SubNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const MulNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const DivNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const ModNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const FloorDivNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const FloorModNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const MinNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const MaxNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const EQNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const NENode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const LTNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const LENode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const GTNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const GENode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const AndNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const OrNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

    bool VisitExpr_(const ReduceNode* op, const PrimExpr& expr) {
        MATCH(ReduceNode)
        int num_lhs = op->combiner->lhs.size();
        if (num_lhs != (int)another->combiner->lhs.size()) {
            return false;
        }
        for (int i = 0; i < num_lhs; ++i) {
            if (!VisitExpr(op->combiner->lhs[i], another->combiner->lhs[i])) {
                return false;
            }
        }

        int num_rhs = op->combiner->rhs.size();
        if (num_rhs != (int)another->combiner->rhs.size()) {
            return false;
        }
        for (int i = 0; i < num_rhs; ++i) {
            if (!VisitExpr(op->combiner->rhs[i], another->combiner->rhs[i])) {
                return false;
            }
        }

        int num_res = op->combiner->result.size();
        if (num_res != (int)another->combiner->result.size()) {
            return false;
        }
        for (int i = 0; i < num_res; ++i) {
            if (!VisitExpr(op->combiner->result[i], another->combiner->result[i])) {
                return false;
            }
        }

        int num_src = op->source.size();
        if (num_src != (int)another->source.size()) {
            return false;
        }
        for (int i = 0; i < num_src; ++i) {
            if (!VisitExpr(op->source[i], another->source[i])) {
                return false;
            }
        }
        // do not check axis
        return VisitExpr(op->condition, another->condition) && op->value_index == another->value_index;
    }

    bool VisitExpr_(const CastNode* op, const PrimExpr& expr) {
        MATCH(CastNode)
        return VisitExpr(op->value, another->value);
    }

    bool VisitExpr_(const NotNode* op, const PrimExpr& expr) {
        MATCH(NotNode)
        return VisitExpr(op->a, another->a);
    }

    bool VisitExpr_(const SelectNode* op, const PrimExpr& expr) {
        MATCH(SelectNode)
        return VisitExpr(op->condition, another->condition) &&
               VisitExpr(op->true_value, another->true_value) &&
               VisitExpr(op->false_value, another->false_value);
    }

    bool VisitExpr_(const RampNode* op, const PrimExpr& expr) {
        MATCH(RampNode)
        return VisitExpr(op->base, another->base) && 
               VisitExpr(op->stride, another->stride) &&
               op->lanes == another->lanes;
    }

    bool VisitExpr_(const BroadcastNode* op, const PrimExpr& expr) {
        MATCH(BroadcastNode)
        return VisitExpr(op->value, another->value) && op->lanes == another->lanes;
    }

    bool VisitExpr_(const ShuffleNode* op, const PrimExpr& expr) {
        MATCH(ShuffleNode)
        int num_vec = op->vectors.size();
        if (num_vec != (int)another->vectors.size()) {
            return false;
        }
        for (int i = 0; i < num_vec; ++i) {
            if (!VisitExpr(op->vectors[i], another->vectors[i])) {
                return false;
            }
        }

        int num_ind = op->indices.size();
        if (num_ind != (int)another->indices.size()) {
            return false;
        }
        for (int i = 0; i < num_ind; ++i) {
            if (!VisitExpr(op->indices[i], another->indices[i])) {
                return false;
            }
        }

        return true;
    }

    bool VisitExpr_(const IntImmNode* op, const PrimExpr& expr) {
        MATCH(IntImmNode)
        return op->value == another->value;
    }

    bool VisitExpr_(const FloatImmNode* op, const PrimExpr& expr) {
        MATCH(FloatImmNode)
        return op->value == another->value;
    }

    bool VisitExpr_(const StringImmNode* op, const PrimExpr& expr) {
        MATCH(StringImmNode)
        return true;
    }
};

class IndexExprMatcher : public ExprVisitor {
  public:
    Array<IterVarMap> match(Array<Array<PrimExpr>> target_indices,
                            Array<Array<PrimExpr>> intrin_indices, Array<IterVar>& target_axes,
                            Array<IterVar>& intrin_axes, Map<IterVar, Range> target_bounds,
                            Map<IterVar, Range> intrin_bounds);

  private:
    bool _match_index(Array<PrimExpr> target_idx, Array<PrimExpr> intrin_idx);
    bool _match_indices(Array<Array<PrimExpr>> target_indices,
                        Array<Array<PrimExpr>> intrin_indices);
    Array<Array<PrimExpr>> _rewrite_indices(Array<Array<PrimExpr>> indices,
                                            IterVarMap itervar_map,
                                            Map<IterVar, Range> target_bounds,
                                            Map<IterVar, Range> intrin_bounds);
};

class IterVarRewriter final : public ExprMutator {
  public:
    using ExprMutator::VisitExpr;
    IterVarRewriter(
        IterVarMap& itervar_map,
        Map<IterVar, Range>& bounds
    ) : itervar_map(itervar_map) {
        for (auto it : bounds) {
            const VarNode* var = it.first.get()->var.get();
            this->bounds[var] = it.second;
        }
    }

  protected:
    using ExprMutator::VisitExpr_;
    PrimExpr VisitExpr_(const VarNode* op) {
        for (auto item : itervar_map) {
        if (op != item.first.get()->var.get()) continue;
            return item.second;
        }
        // return make_zero(op->dtype);
        return bounds[op].get()->min;
    }

  private:
    IterVarMap& itervar_map;
    std::unordered_map<const VarNode*, Range> bounds;
};


// ported from AutoTensorize/tg/autodiff
 /*
  * TODO: (yicheng): for intrin index match, use a better way to do relaxed matching
  * We allow some relaxed cases to be matched: (note x1 is left, x2 is right)
  * 0. x1 matches positive int const
  * 1. we always assume a pre-simplify before check expr equal
  * 2. if x1 matches with x2, then x1 + beta / beta + x1 matches with x2, beta is int
  * 3. if x1 matches with x2, then x1 * alpha / alpha * x1 matches with x2, alpha is positive int
  * 4. if x1 matches with x2, then x1 - beta matches with x2, beta is int
  */ 
class CheckExprEqual : public ExprFunctor<bool(const PrimExpr&, const PrimExpr&)> {
  private:
    bool check_name_;
    bool relax_const_;

  public:
    CheckExprEqual(bool check_name = false, bool relax_const = false)
        : check_name_(check_name), relax_const_(relax_const) {}

    bool check_equal(const PrimExpr& a, const PrimExpr& b) { return VisitExpr(a, b); }

    bool operator()(const PrimExpr& a, const PrimExpr& b) { return check_equal(a, b); }

  protected:
#define type_check(T)                   \
    const T* other_op = target.as<T>(); \
    if (other_op == nullptr) {          \
        return false;                   \
    }

    // list of functions to override.
    bool VisitExpr_(const VarNode* op, const PrimExpr& target) override {
        const IntImmNode* as_int = target.as<IntImmNode>();
        if (as_int != nullptr && as_int->value > 0) {
            return true;
        }
        type_check(VarNode)
        if (check_name_) { 
            return op->name_hint == other_op->name_hint; 
        }
        else {
            return true;
        }
    }

    bool VisitExpr_(const SizeVarNode* op, const PrimExpr& target) override {
        type_check(SizeVarNode) 
        return op->name_hint == other_op->name_hint;
    }

    bool VisitExpr_(const LoadNode* op, const PrimExpr& target) override {
        type_check(LoadNode) 
        return (VisitExpr(op->buffer_var, other_op->buffer_var) &&
                VisitExpr(op->index, other_op->index) &&
                VisitExpr(op->predicate, other_op->predicate));
    }

    bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& target) override {
        type_check(BufferLoadNode) 
        if (op->indices.size() != other_op->indices.size()) { 
            return false; 
        }
        for (size_t i = 0; i < op->indices.size(); ++i) {
            if (!VisitExpr(op->indices[i], other_op->indices[i])) {
                return false;
            }
        }
        return true;
    }

    bool VisitExpr_(const LetNode* op, const PrimExpr& target) override {
        type_check(LetNode) 
        return (VisitExpr(op->var, other_op->var) &&
                VisitExpr(op->value, other_op->value) &&
                VisitExpr(op->body, other_op->body));
    }

    bool VisitExpr_(const ProducerLoadNode* op, const PrimExpr& target) override {
        type_check(ProducerLoadNode)
        if (op->producer != other_op->producer) { return false; }
        if (op->indices.size() != other_op->indices.size()) {
            return false;
        }
        for (size_t i = 0; i < op->indices.size(); ++i) {
            if (!VisitExpr(op->indices[i], other_op->indices[i])) {
                return false;
            }
        }
        return true;
    }

    bool VisitExpr_(const CallNode* op, const PrimExpr& target) override {
        type_check(CallNode)
        if (op->op != other_op->op) { return false; }
        if (op->args.size() != other_op->args.size()) {
            return false;
        }
        for (size_t i = 0; i < op->args.size(); ++i) {
            if (!VisitExpr(op->args[i], other_op->args[i])) {
                return false;
            }
        }
        return true;
    }

    bool VisitExpr_(const AddNode* op, const PrimExpr& target) override {
        /*
         * Note: We always assume a pre-simplify before check_equal
         */ 
        if (relax_const_) {
            const IntImmNode* a_as_int = op->a.as<IntImmNode>();
            const IntImmNode* b_as_int = op->b.as<IntImmNode>();
            if (a_as_int != nullptr) {
                return VisitExpr(op->b, target);
            } else if (b_as_int != nullptr) {
                return VisitExpr(op->a, target);
            }
        }
        type_check(AddNode)
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const SubNode* op, const PrimExpr& target) override {
        /*
         * Note: We always assume a pre-simplify before check_equal
         */ 
        if (relax_const_) {
            const IntImmNode* b_as_int = op->b.as<IntImmNode>();
            // only consider if b is int
            if (b_as_int != nullptr) {
                return VisitExpr(op->a, target);
            }
        }
        type_check(SubNode)
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const MulNode* op, const PrimExpr& target) override {
        /*
         * Note: We always assume a pre-simplify before check_equal
         */ 
        if (relax_const_) {
            const IntImmNode* a_as_int = op->a.as<IntImmNode>();
            const IntImmNode* b_as_int = op->b.as<IntImmNode>();
            if (a_as_int != nullptr && a_as_int->value > 0) {
                return VisitExpr(op->b, target);
            } else if (b_as_int != nullptr && b_as_int->value > 0) {
                return VisitExpr(op->a, target);
            }
        }
        type_check(MulNode)
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const DivNode* op, const PrimExpr& target) override {
        type_check(DivNode)
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const ModNode* op, const PrimExpr& target) override {
        type_check(ModNode)
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const FloorDivNode* op, const PrimExpr& target) override {
        type_check(FloorDivNode)
        return (VisitExpr(op->a, other_op->a) &&
                VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const FloorModNode* op, const PrimExpr& target) override {
        type_check(FloorModNode) 
        return (VisitExpr(op->a, other_op->a) &&
                VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const MinNode* op, const PrimExpr& target) override {
        type_check(MinNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const MaxNode* op, const PrimExpr& target) override {
        type_check(MaxNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const EQNode* op, const PrimExpr& target) override {
        type_check(EQNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const NENode* op, const PrimExpr& target) override {
        type_check(NENode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const LTNode* op, const PrimExpr& target) override {
        type_check(LTNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const LENode* op, const PrimExpr& target) override {
        type_check(LENode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const GTNode* op, const PrimExpr& target) override {
        type_check(GTNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const GENode* op, const PrimExpr& target) override {
        type_check(GENode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const AndNode* op, const PrimExpr& target) override {
        type_check(AndNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const OrNode* op, const PrimExpr& target) override {
        type_check(OrNode) 
        return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
    }

    bool VisitExpr_(const ReduceNode* op, const PrimExpr& target) override {
        type_check(ReduceNode) 
        if (op->combiner != other_op->combiner) { return false; }
        if (op->source.size() != other_op->source.size()) {
            return false;
        }
        for (size_t i = 0; i < op->source.size(); ++i) {
            if (!VisitExpr(op->source[i], other_op->source[i])) {
                return false;
            }
        }
        if (op->axis.size() != other_op->axis.size()) {
            return false;
        }
        for (size_t i = 0; i < op->axis.size(); ++i) {
            if (op->axis[i] != other_op->axis[i]) {
                return false;
            }
        }
        return (VisitExpr(op->condition, other_op->condition) &&
                op->value_index == other_op->value_index);
    }

    bool VisitExpr_(const CastNode* op, const PrimExpr& target) override {
        type_check(CastNode) 
        return VisitExpr(op->value, other_op->value);
    }

    bool VisitExpr_(const NotNode* op, const PrimExpr& target) override {
        type_check(NotNode) 
        return VisitExpr(op->a, other_op->a);
    }

    bool VisitExpr_(const SelectNode* op, const PrimExpr& target) override {
        type_check(SelectNode) 
        return (VisitExpr(op->condition, other_op->condition) &&
                VisitExpr(op->true_value, other_op->true_value) &&
                VisitExpr(op->false_value, other_op->false_value));
    }

    bool VisitExpr_(const RampNode* op, const PrimExpr& target) override {
        type_check(RampNode) 
        return (VisitExpr(op->base, other_op->base) &&
                VisitExpr(op->stride, other_op->stride) &&
                op->lanes == other_op->lanes);
    }

    bool VisitExpr_(const BroadcastNode* op, const PrimExpr& target) override {
        type_check(BroadcastNode) 
        return (VisitExpr(op->value, other_op->value) &&
                op->lanes == other_op->lanes);
    }

    bool VisitExpr_(const ShuffleNode* op, const PrimExpr& target) override {
        type_check(ShuffleNode) 
        if (op->vectors.size() != other_op->vectors.size()) { return false; }
        for (size_t i = 0; i < op->vectors.size(); ++i) {
            if (!VisitExpr(op->vectors[i], other_op->vectors[i])) {
                return false;
            }
        }
        if (op->indices.size() != other_op->indices.size()) {
            return false;
        }
        for (size_t i = 0; i < op->indices.size(); ++i) {
            if (!VisitExpr(op->indices[i], other_op->indices[i])) {
                return false;
            }
        }
        return true;
    }

    bool VisitExpr_(const IntImmNode* op, const PrimExpr& target) override {
        type_check(IntImmNode) 
        return op->value == other_op->value;
    }

    bool VisitExpr_(const FloatImmNode* op, const PrimExpr& target) override {
        type_check(FloatImmNode) 
        return op->value == other_op->value;
    }

    bool VisitExpr_(const StringImmNode* op, const PrimExpr& target) override {
        type_check(StringImmNode) 
        return op->value == other_op->value;
    }
};

} // namespace auto_tensorize
} // namespace ditto
