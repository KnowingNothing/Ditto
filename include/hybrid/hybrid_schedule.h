#pragma once

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>

#include <hybrid/test.h>
#include <hybrid/tree.h>
#include <string>
#include <unordered_map>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

// Node container for HybridStage
class HybridStageNode;
// Node container for HybridSchedule
class HybridScheduleNode;

/*! \brief HybridStage, contains scheduling for a hybrid_stage of computation. */
class HybridStage : public ObjectRef {
 public:
  HybridStage() {}
  explicit HybridStage(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief create a new schedule for op.
   * \param op The operator in the schedule
   */
  explicit HybridStage(Operation op);
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const HybridStageNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline HybridStageNode* operator->();
  void Test();
  void display(std::string);
  /*!
  * \brief slice the stage into array.
  * \param slicept the itervar to slice. 
  * \param pinpoint the root to insert the sliced tree.
  */
  TVM_DLL HybridStage& slice(IterVar slicept, IterVar pinpt, PrimExpr factor);
  /*!
   * \brief set the memory scope of the stage
   * \param scope The memory scope.
   */
  TVM_DLL HybridStage& set_scope(std::string scope);  // NOLINT(*)
  /*!
   * \brief specify the schedule to be computed at the parent schedule's scope.
   * \param parent The parent schedule.
   * \param scope The iteration point to carry the schedule.
   * \return reference to self.
   */
  TVM_DLL HybridStage& compute_at(HybridStage parent, IterVar scope);  // NOLINT(*)
  /*!
   * \brief Compute the function inline.
   * \return reference to self.
   */
  TVM_DLL HybridStage& compute_inline();  // NOLINT(*)
  /*!
   * \brief Compute the function at group root.
   * \return reference to self.
   */
  TVM_DLL HybridStage& compute_root();  // NOLINT(*)
  /*!
   * \brief Bind the IterVar to thread index.
   *
   * \param ivar The IterVar to be bound.
   * \param thread_ivar The thread axis to be bound.
   * \return reference to self.
   */
  TVM_DLL HybridStage& bind(IterVar ivar, IterVar thread_ivar);
  /*!
   * \brief Set the predicate to determine whether a store to the array should be performed.
   *  Use this when there are multiple threads performing the same store and we only
   *  need one of them to do the store.
   *
   * \note This is a dangerous scheduling primitive that can change behavior of program.
   *    Only do when we are certain that thare are duplicated stores.
   * \param predicate The condition to be checked.
   * \return reference to self.
   */
  TVM_DLL HybridStage& set_store_predicate(PrimExpr predicate);
  /*!
   * \brief Specify environment threads that launched around the group's scope.
   *  This can only be used in group hybrid_stage.
   * \param threads The threads to be launched around the scope.
   * \note Each thread can only appear in one env_threads.
   *    This is a beta feature.
   * \return reference to self.
   */
  TVM_DLL HybridStage& env_threads(Array<IterVar> threads);
  /*!
   * \brief Split the parent by factor, generate
   * \param parent The parent iteration domain.
   * \param factor The split factor of the loop.
   * \param p_outer The result outer domain
   * \param p_inner The result inner domain.
   * \return reference to self.
   */
  TVM_DLL HybridStage& split(IterVar parent, PrimExpr factor, IterVar* p_outer,
                       IterVar* p_inner);  // NOLINT(*)
  /*!
   * \brief Split the iteration with given number of parts.
   *
   * \param parent The parent domain.
   * \param nparts The number of parts in the outer domain.
   * \param p_outer The result outer domain.
   * \param p_inner The result inner domain.
   * \return reference to self.
   */
  TVM_DLL HybridStage& split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer,
                                 IterVar* p_inner);  // NOLINT(*)
  /*!
   * \brief Fuse the inner outer domain to the target
   * \param outer The outer domain to be fused.
   * \param inner The inner domain to be fused
   * \param p_target The result target domain.
   * \return reference to self.
   */
  TVM_DLL HybridStage& fuse(IterVar outer, IterVar inner, IterVar* p_target);  // NOLINT(*)
  /*!
   * \brief Fuse all the axes together into a single axis.
   *
   * \param axes All the axes to be fused.
   * \param p_target The result target domain.
   *
   * \note axes can be an empty array,
   *       in that case, a singleton IterVar is created and
   *       inserted to the outermost loop.
   *       The fuse of empty array is used to support zero-dimension tensors.
   *
   * \return reference to self.
   */
  TVM_DLL HybridStage& fuse(const Array<IterVar>& axes, IterVar* p_target);  // NOLINT(*)
  /*!
   * \brief Reorder the iteration
   * \param order The order of iteration variable.
   * \return reference to self.
   */
  TVM_DLL HybridStage& reorder(const Array<IterVar>& order);  // NOLINT(*)
  /*!
   * \brief Perform tiling on two dimensions
   *  The final loop order from outmost to inner most are
   *  [x_outer, y_outer, x_inner, y_inner]
   *
   * \param x_parent The original x dimension
   * \param y_parent The original y dimension
   * \param x_factor The stride factor on x axis
   * \param y_factor The stride factor on y axis
   * \param p_x_outer Outer axis of x dimension
   * \param p_y_outer Outer axis of y dimension
   * \param p_x_inner Inner axis of x dimension
   * \param p_y_inner Inner axis of y dimension
   * \return reference to self.
   */
  TVM_DLL HybridStage& tile(IterVar x_parent, IterVar y_parent,  // NOLINT(*)
                      PrimExpr x_factor, PrimExpr y_factor, IterVar* p_x_outer, IterVar* p_y_outer,
                      IterVar* p_x_inner, IterVar* p_y_inner);
  /*!
   * \brief Vectorize iteration.
   * \param var The axis to be vectorized.
   * \return reference to self.
   */
  TVM_DLL HybridStage& vectorize(IterVar var);  // NOLINT(*)
  /*!
   * \brief Replace computation of the current stage by tensor intrinsic f.
   * \param var The axis marks beginning of tensorization.
   *  Every operations inside the axis(include axis itself is tensorized).
   * \param f The Tensor compute intrinsics.
   * \return reference to self.
   */
  TVM_DLL HybridStage& tensorize(IterVar var, TensorIntrin f);  // NOLINT(*)
  /*!
   * \brief Unroll iteration.
   * \param var The axis to be unrolled.
   * \return reference to self.
   */
  TVM_DLL HybridStage& unroll(IterVar var);  // NOLINT(*)
  /*!
   * \brief Parallelize iteration.
   * \param var The axis to be parallelized.
   * \return reference to self.
   */
  TVM_DLL HybridStage& parallel(IterVar var);  // NOLINT(*)
  /*!
   * \brief Annotate the iteration with pragma; no change needed
   *
   * \param var The axis to be parallelized.
   * \param pragma_type The pragma type.
   * \param pragma_value The pragma value
   *
   * \return reference to self.
   */
  TVM_DLL HybridStage& pragma(IterVar var, const std::string& pragma_type,
                        const PrimExpr& pragma_value = PrimExpr());  // NOLINT(*)
  /*!
   * \brief Fetch data in advance. not need change
   * \param domain the tensor to be prefetched
   * \param var the iteration point at which to apply prefetching
   * \param offset the number of iterations be to fetched in advance
   * \return reference to self
   */
  TVM_DLL HybridStage& prefetch(const Tensor& domain, IterVar var, PrimExpr offset);  // NOLINT(*)
  /*!
   * \brief Set alignment requirement for specific dimension.
   *
   *  Such that stride[axis] == k * factor + offset for some k.
   *
   * \param axis The dimension to be specified for alignment.
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor.
   * \return reference to self
   */
  TVM_DLL HybridStage& storage_align(IterVar axis, int factor, int offset);  // NOLINT(*)
  /*!
   * \brief Compute current stage with double buffering.
   * \return reference to self.
   */
  TVM_DLL HybridStage& double_buffer();  // NOLINT(*)
  
  TVM_DLL HybridStage& slice(
    TreeUnitNode<IterVar>* slicept, 
    TreeUnitNode<IterVar>* pinpt, 
    PrimExpr factor, 
    std::string mode, 
    Array<IterVar> *left, 
    Array<IterVar> *right
  );
  
  TVM_DLL HybridStage& slice(
    IterVar slicept, 
    IterVar pinpt, 
    PrimExpr factor, 
    std::string mode, 
    Array<IterVar> *left, 
    Array<IterVar> *right
  );
  
  TVM_DLL HybridStage& slice(
    IterVar slicept, 
    PrimExpr factor, 
    std::string mode, 
    Array<IterVar> *left, 
    Array<IterVar> *right
  );

  /*!
   * \brief whether the stage has been scheduled.
   * \return whether the stage has been scheduled.
   */
  bool is_scheduled() const;
  /*!
   * \brief Get attachment spec of current hybrid_stage.
   *  If the hybrid_stage compute at Group root, this function
   *  will traverse the group function to get the
   *  final spec from the group.
   * \return A hybrid_stage representing the attach spec of the group.
   */
  HybridStage GetAttachSpec() const;
  // declare container type
  using ContainerType = HybridStageNode;
};

/*!
 * \brief Global hybrid_schedule container
 */
class HybridSchedule : public ObjectRef {
 public:
  HybridSchedule() {}
  explicit HybridSchedule(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Create a hybrid_schedule for array of ops(and their dependencies).
   * \param ops The ops to be scheduled.
   * \return sch The created HybridSchedule.
   */
  TVM_DLL explicit HybridSchedule(Array<Operation> ops);
  /*!
   * \brief Get a copy of current hybrid_schedule.
   * \return The copied hybrid_schedule.
   */
  HybridSchedule copy() const;
  /*!
   * \brief Get the hybrid_stage corresponds to the op
   * \param op The operation.
   */
  TVM_DLL HybridStage operator[](const Operation& op);
  /*!
   * \brief Short hand for getting the hybrid_stage of tensor's operation.
   * \param tensor The tensor
   * \return The hybrid_stage corresponding to the tensor's op
   */
  TVM_DLL HybridStage operator[](const Tensor& tensor) { return this->operator[](tensor->op); }
  /*!
   * \brief Create a new hybrid_stage group for all intermediate
   *  operations between inputs and outputs.
   *
   * \param outputs The output boundary of the group.
   * \param inputs The input boundary of the group.
   * \param include_inputs Whether include inputs if they are reachable from outputs.
   * \return The new grouped hybrid_stage.
   */
  TVM_DLL HybridStage create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs = false);
  /*!
   * \brief create a cache read of original tensor for readers.
   *  This will mutate the body of the readers.
   *  A new hybrid_stage will be created for the tensor.
   * \param tensor The tensor cached.
   * \param scope The scope of the cache.
   * \param readers The readers to redirect to the tensor.
   * \return The created tensor.
   */
  TVM_DLL Tensor cache_read(const Tensor& tensor, const std::string& scope,
                            const Array<Operation>& readers);
  /*!
   * \brief Create a cache write tensor for producing tensor.
   *  The the tensor will take over body of original tensor op.
   *
   *  This function can be used to do data layout transformation.
   *  If there is a split/fuse/reorder on the data parallel axis of tensor
   *  before cache_write is called. The intermediate cache stores
   *  the data in the layout as the iteration order of leave axis.
   *  The data will be transformed back to the original layout in the original tensor.
   *  User can further call compute_inline to inline the original layout and keep
   *  the data stored in the transformed layout.
   *
   * \param tensor The tensors to be produced.
   * \param scope The scope of the storage.
   * \return The created tensor.
   */
  TVM_DLL Array<Tensor> cache_write(const Array<Tensor>& tensor, const std::string& scope);
  /*!
   * \brief Create a cache write tensor for producing tensor.
   *  The the tensor will take over body of original tensor op.
   *
   *  This function can be used to do data layout transformation.
   *  If there is a split/fuse/reorder on the data parallel axis of tensor
   *  before cache_write is called. The intermediate cache stores
   *  the data in the layout as the iteration order of leave axis.
   *  The data will be transformed back to the original layout in the original tensor.
   *  User can further call compute_inline to inline the original layout and keep
   *  the data stored in the transformed layout.
   *
   * \param tensor The tensor to be produced.
   * \param scope The scope of the storage.
   * \return The created tensor.
   */
  TVM_DLL Tensor cache_write(const Tensor& tensor, const std::string& scope);
  /*!
   * \brief Factor a reduction axis in tensor's hybrid_schedule to be an explicit axis.
   * This will create a new hybrid_stage that generated the new tensor with axis
   * as the first dimension. The tensor's body will be rewritten as a reduction
   * over the factored tensor.
   *
   *  P. Suriana, A. Adams and S. Kamil. Parallel associative reductions in halide. CGO'17
   *
   * \param tensor The tensor to be factored.
   * \param axis The reduction axis in tensor's hybrid_schedule to be factored.
   * \param factor_axis The position where the new axis is placed.
   * \return The created factored tensors.
   */
  TVM_DLL Array<Tensor> rfactor(const Tensor& tensor, const IterVar& axis, int factor_axis = 0);
  /*!
   * \brief Normalize the hybrid_schedule.
   *  This is needed before bound inference.
   *  Insert necessary RebaseNode to make sure all leaf_iter_vars
   *  are in form [0, extent)
   *
   * \return A normalized hybrid_schedule, can be same as current one.
   */
  HybridSchedule normalize();

  /*!
   * \brief Normalize the hybrid_schedule for feature extraction in auto-scheduler.
   * This is similar to `HybridSchedule::normalize`, but we do aggressive simplification
   * to the TE compute with const_matrix=True for faster compilation and feature extraction.
   * The resulted hybrid_schedule may be wrong, but it is good enough for feature extraction
   * purposes.
   *
   * \return A normalized hybrid_schedule, can be same as current one.
   */
  HybridSchedule normalize_for_feature_extraction();

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const HybridScheduleNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline HybridScheduleNode* operator->();
  // declare container type
  using ContainerType = HybridScheduleNode;
};

/*!
 * \brief represents a hybrid_stage.
 *
 *  relations form a Directed acylic hypergraph in bipartite manner.
 *  With each node is represented by a IterVar,
 *  and each hyper-edge is represented by a IterVarRelation.
 *  The relations connects the IterVars in the graph.
 *
 *  Besides typical hybrid_stage that corresponds to operations.
 *  There is also group hybrid_stage, which groups hybrid_stages together.
 *  Each hybrid_stage's group(given by group) represent an constraint,
 *  the hybrid_stage can only be attached to hybrid_stages within the group.
 *
 *  The group hybrid_stage node can be attached to IterVars as in normal hybrid_stage.
 */
class HybridStageNode : public Object {
 public:
  /*!
   * \brief The operation of hybrid_stage, can be different from original op.
   *  If it is null, then this hybrid_stage is a group hybrid_stage.
   */
  Operation op;
  /*!
   * \brief The original operator.
   *  The op field can change during schedule to alternate the dataflow,
   *  while origin_op remains fixed.
   */
  Operation origin_op;
  /*! \brief All the nodes in the iter var */
  Array<IterVar> all_iter_vars;
  /*! \brief The current active leaf iter vars in the hybrid_stage as a array. */
  Array<IterVar> leaf_iter_vars;
  /*! \brief The current active leaf iter vars in the hybrid_stage as a tree. */
  // TreeNode<IterVar> leaf_iter_vars_tree;
  Tree<IterVar> leaf_iter_vars_tree;
  /*!
   * \brief Specify threads to be launched at the hybrid_stage.
   *  This is only valid for composite ops such as Scan.
   * \note Experimental primitive: used for thread persistence.
   */
  Array<IterVar> env_threads;
  /*!
   * \brief The predicate under which store can happen
   *  Use this when there can be duplicated threads doing the same store.
   * \note Experimental primitive: used by cross thread-reduction.
   */
  PrimExpr store_predicate;
  /*! \brief The relation bwteen of IterVars */
  Array<IterVarRelation> relations;
  /*! \brief additional attributes about iter var. */
  Map<IterVar, IterVarAttr> iter_var_attrs;
  /*! \brief The attachment type of the schedule */
  AttachType attach_type{kGroupRoot};
  /*! \brief The attach point of this schedule. */
  IterVar attach_ivar;
  /*! \brief The hybrid_stage this node attaches to */
  HybridStage attach_stage;
  /*! \brief The thread storage scope level of the hybrid_stage */
  std::string scope;
  /*! \brief Whether this is an output hybrid_stage */
  bool is_output{false};
  /*! \brief Whether apply double buffer optimization to this hybrid_stage */
  bool double_buffer{false};
  /*!
   * \brief The parent group of the current hybrid_stage.
   *  The hybrid_stage cannot be assigned to hybrid_stages outside the group.
   */
  HybridStage group;
  /*! \brief Number of direct child hybrid_stages, only used for group hybrid_stage.*/
  int num_child_stages{0};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("origin_op", &origin_op);
    v->Visit("all_iter_vars", &all_iter_vars);
    v->Visit("leaf_iter_vars", &leaf_iter_vars);
    v->Visit("env_threads", &env_threads);
    v->Visit("relations", &relations);
    v->Visit("iter_var_attrs", &iter_var_attrs);
    v->Visit("attach_type", &attach_type);
    v->Visit("attach_ivar", &attach_ivar);
    v->Visit("attach_stage", &attach_stage);
    v->Visit("scope", &scope);
    v->Visit("is_output", &is_output);
    v->Visit("double_buffer", &double_buffer);
    v->Visit("group", &group);
    v->Visit("num_child_stages", &num_child_stages);
  }

  static constexpr const char* _type_key = "HybridStage";
  TVM_DECLARE_FINAL_OBJECT_INFO(HybridStageNode, Object);
};

/*! \brief node container for hybrid_schedule */
class HybridScheduleNode : public Object {
 public:
  /*! \brief The output operations in original data flow graph */
  Array<Operation> outputs;
  /*!
   * \brief list of all hybrid_stages for ops.
   * The hybrid_stages are sorted in dependency order.
   */
  Array<HybridStage> stages;
  /*!
   * \brief List of all stage groups.
   */
  Array<HybridStage> groups;
  /*! \brief map of original operation to the hybrid_stages */
  Map<Operation, HybridStage> stage_map;
  /*!
   * \brief Internal hybrid_stage map to map internal ops to hybrid_stages.
   *  This is created on demand and can be invalidated.
   */
  std::unordered_map<const Object*, HybridStage> op2stage_cache_;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("outputs", &outputs);
    v->Visit("stages", &stages);
    v->Visit("groups", &groups);
    v->Visit("stage_map", &stage_map);
  }

  /*! \brief Initialize temp cache. */
  void InitCache();
  /*! \brief Invalidate temp cache. */
  void InvalidateCache();

  /*!
   * \brief Check if the hybrid_schedule contains an Operation.
   * \param op The candidate Operation.
   * \return true if the hybrid_schedule has the Operation. Otherwise, false.
   */
  TVM_DLL bool Contain(const Operation& op) const;

  /*!
   * \brief Check if the hybrid_schedule contains a Tensor.
   * \param tensor The candidate tensor.
   * \return true if the hybrid_schedule has the tensor. Otherwise, false.
   */
  TVM_DLL bool Contain(const Tensor& tensor) const { return Contain(tensor->op); }

  static constexpr const char* _type_key = "HybridSchedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(HybridScheduleNode, Object);
};

/*!
 * \brief Create a hybrid_schedule for array of ops(and their dependencies).
 * \param ops The ops to be scheduled.
 * \return sch The created HybridSchedule.
 */
inline HybridSchedule create_hybrid_schedule(Array<Operation> ops) { return HybridSchedule(ops); }


/*!
 * \brief Slice.
 */
class SliceNode : public IterVarRelationNode {
 public:
  /*! \brief The old domain */
  Array<IterVar> old;
  /*! \brief The left domain */
  Array<IterVar> left;
  /*! \brief The right domain */
  Array<IterVar> right;
  /*! \brief The slice point */
  IterVar slicept;
  /*! \brief The pin point */
  IterVar pinpt;
  /*! \brief The slice mode */
  std::string mode;
  /*! \brief The slice factor */
  PrimExpr factor;
  /*! \brief var for select node*/
  Var sel;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("old", &old);
    v->Visit("left", &left);
    v->Visit("right", &right);
    v->Visit("slicept", &slicept);
    v->Visit("pinpt", &pinpt);
    v->Visit("mode", &mode);
    v->Visit("factor", &factor);
    v->Visit("sel", &sel);
  }

  static constexpr const char* _type_key = "Slice";
  TVM_DECLARE_FINAL_OBJECT_INFO(SliceNode, IterVarRelationNode);
};

/*!
 * \brief Managed reference to SliceNode
 * \sa SliceNode
 */
class Slice : public IterVarRelation {
 public:
  TVM_DLL Slice(Array<IterVar> old, Array<IterVar> left, Array<IterVar> right, IterVar slicept, IterVar pinpt, std::string mode, PrimExpr factor, Var sel);

  TVM_DEFINE_OBJECT_REF_METHODS(Slice, IterVarRelation, SliceNode);
};


// implementations
inline const HybridStageNode* HybridStage::operator->() const { return static_cast<const HybridStageNode*>(get()); }
inline HybridStageNode* HybridStage::operator->() { return static_cast<HybridStageNode*>(get_mutable()); }

inline const HybridScheduleNode* HybridSchedule::operator->() const {
  return static_cast<const HybridScheduleNode*>(get());
}
inline HybridScheduleNode* HybridSchedule::operator->() { return static_cast<HybridScheduleNode*>(get_mutable()); }

}  // namespace ditto
}  // namespace hybrid
