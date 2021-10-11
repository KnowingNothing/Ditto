#pragma once

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>

#include <string>
#include <unordered_map>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid {

// Node container for Schedule
class HybridScheduleNode;

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
   * \brief Get the stage corresponds to the op
   * \param op The operation.
   */
  TVM_DLL Stage operator[](const Operation& op);
  /*!
   * \brief Short hand for getting the stage of tensor's operation.
   * \param tensor The tensor
   * \return The stage corresponding to the tensor's op
   */
  TVM_DLL Stage operator[](const Tensor& tensor) { return this->operator[](tensor->op); }
  /*!
   * \brief Create a new stage group for all intermediate
   *  operations between inputs and outputs.
   *
   * \param outputs The output boundary of the group.
   * \param inputs The input boundary of the group.
   * \param include_inputs Whether include inputs if they are reachable from outputs.
   * \return The new grouped stage.
   */
  TVM_DLL Stage create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs = false);
  /*!
   * \brief create a cache read of original tensor for readers.
   *  This will mutate the body of the readers.
   *  A new stage will be created for the tensor.
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
   * This will create a new stage that generated the new tensor with axis
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

/*! \brief node container for hybrid_schedule */
class HybridScheduleNode : public Object {
 public:
  /*! \brief The output operations in original data flow graph */
  Array<Operation> outputs;
  /*!
   * \brief list of all stages for ops.
   * The stages are sorted in dependency order.
   */
  Array<Stage> stages;
  /*!
   * \brief List of all stage groups.
   */
  Array<Stage> groups;
  /*! \brief map of original operation to the stages */
  Map<Operation, Stage> stage_map;
  /*!
   * \brief Internal stage map to map internal ops to stages.
   *  This is created on demand and can be invalidated.
   */
  std::unordered_map<const Object*, Stage> op2stage_cache_;

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

// implementations
inline const HybridScheduleNode* HybridSchedule::operator->() const {
  return static_cast<const HybridScheduleNode*>(get());
}
inline HybridScheduleNode* HybridSchedule::operator->() { return static_cast<HybridScheduleNode*>(get_mutable()); }

}  // namespace ditto
}  // namespace hybrid
