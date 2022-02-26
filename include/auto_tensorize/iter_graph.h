#pragma once

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>


#include <string>
#include <unordered_map>
#include <vector>

#include <hardware/base/hw_param.h>

using namespace tvm;
namespace ditto {
namespace auto_tensorize {
class FusionItem;
class FusionSpace;
class FusionResult;
enum class IV_Type : int {
  SPATIAL = 0,
  REDUCE = 1,
};
typedef int FACTOR;

/*!
 * \brief A class for access function.
 */
class AccessFunctionNode : public Object {
public:
  /*! \brief The tensor op to access */
  te::Operation op;
  /*! \brief The indices for one tensor */
  Array<Array<PrimExpr>> access_indices;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("op", &op);
    v->Visit("access_indices", &access_indices);
  }
  /*! \brief get the data footprint */
  std::vector<int> getFootprint(Map<tir::Var, IntImm> bounds);
  /*! \brief get the workload */
  int getWorkload(Map<tir::Var, IntImm> bounds);
  static constexpr const char *_type_key =
      "ditto.auto_tensorize.AccessFunction";
  TVM_DECLARE_BASE_OBJECT_INFO(AccessFunctionNode, Object);
};

class AccessFunction : public ObjectRef {
public:
  /*!
   * \brief The constructor.
   * \param op The operation
   */
  TVM_DLL AccessFunction(te::Operation op,
                         Array<Array<PrimExpr>> access_indices);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AccessFunction, ObjectRef,
                                        AccessFunctionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AccessFunctionNode);
};

class IterVarNode : public Object {
public:
  /*! \brief the original index in tvm IR */
  int index;
  FACTOR ext;
  IV_Type iv_type;
  tvm::tir::Var name;
  tvm::tir::Var originVar;
  bool shared; // mark shared & common axis
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("index", &index);
    v->Visit("ext", &ext);
    v->Visit("iv_type", &iv_type);
    v->Visit("name", &name);
    v->Visit("originVar", &originVar);
  }

  bool isSpatial() { return iv_type == IV_Type::SPATIAL; }
  bool isReduce() { return iv_type == IV_Type::REDUCE; }
  void setExt(FACTOR ext_) { ext = ext_; }
  static constexpr const char *_type_key = "ditto.auto_tensorize.IterVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterVarNode, Object);
};

class IterVar : public ObjectRef {
public:
  TVM_DLL IterVar(int idx, FACTOR ext, IV_Type iv_type, tvm::tir::Var name,
                  tvm::tir::Var originVar);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IterVar, ObjectRef, IterVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterVarNode);
};

class RelationNode : public Object {
public:
  static constexpr const char *_type_key = "ditto.auto_tensorize.Relation";
  TVM_DECLARE_BASE_OBJECT_INFO(RelationNode, Object);
};

class Relation : public ObjectRef {
public:
  Relation() {}
  explicit Relation(ObjectPtr<Object> n) : ObjectRef(n) {}

  inline const RelationNode *operator->() const;
};

class SplitNode : public RelationNode {
public:
  /*! \brief the splited iter */
  IterVar parent;
  /*! \brief the outer iter */
  IterVar outer;
  /*! \brief the inner iter */
  IterVar inner;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("parent", &parent);
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
  }

  static constexpr const char *_type_key = "ditto.auto_tensorize.Split";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitNode, RelationNode);
};

class Split : public Relation {
public:
  TVM_DLL Split(IterVar parent, IterVar outer, IterVar inner);
  TVM_DEFINE_OBJECT_REF_METHODS(Split, Relation, SplitNode);
};

class ShareNode : public RelationNode {
public:
  /*! \brief the iter from the upper op*/
  IterVar upper;
  /*! \brief the iter from the lower op*/
  IterVar lower;
  void VisitAttrs(AttrVisitor *v) {
    v->Visit("upper", &upper);
    v->Visit("lower", &lower);
  }

  static constexpr const char *_type_key = "ditto.auto_tensorize.Share";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShareNode, RelationNode);
};

class Share : public Relation {
public:
  TVM_DLL Share(IterVar upper, IterVar lower);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Share, Relation, ShareNode);
};

// class AttachNode: public RelationNode{
// public:
//     /*! \brief Head is from the second op.*/
//     int attachPos;
//     void VisitAttrs(AttrVisitor* v){
//     }

//     static constexpr const char * _type_key="Share";
//     TVM_DECLARE_FINAL_OBJECT_INFO(AttachNode, RelationNode);
// };

// class Attach: public Relation{
// public:
//     TVM_DLL Attach(IterVar head, IterVar remain, IterVar orginal);
//     TVM_DEFINE_OBJECT_REF_METHODS(Attach, Relation, AttachNode);
// };

class IterGraphNode : public Object {
public:
  te::Operation op1, op2;
  Array<IterVar> _firstOpIters;
  Array<IterVar> _secondOpIters;
  Array<Share> _sharedIterPairs;

  Array<AccessFunction> _firstOpReadAccessFuncs;
  Array<AccessFunction> _secondOpReadAccessFuncs;
  AccessFunction _firstOpWriteAccessFunc;
  AccessFunction _secondOpWriteAccessFunc;
  int _readProducerPos;
  Array<Share> shareRelations;

  Array<IterVar> commonIters;
  Array<IterVar> firstOpIters;
  int firstOpNumLoops;
  Array<IterVar> secondOpIters;
  int secondOpNumLoops;
  Array<AccessFunction> firstOpReadAccessFuncs;
  Array<AccessFunction> secondOpReadAccessFuncs;
  AccessFunction firstOpWriteAccessFunc;
  AccessFunction secondOpWriteAccessFunc;
  int readProducerPos;
  Array<Split> splitRelations;
  Map<tir::Var, IntImm> bounds;
  Array<IterVar> tensorizeIters;
  
  size_t attachPos = 0; // default: independent loops

  String resultPath;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("init_firstOpIters", &_firstOpIters);
    v->Visit("init_secondOpIters", &_secondOpIters);
    v->Visit("init_firstReadOpAccessFunctions", &_firstOpReadAccessFuncs);
    v->Visit("init_secondReadOpAccessFunctions", &_secondOpReadAccessFuncs);
    v->Visit("init_firstOpWriteAccessFunction", &_firstOpWriteAccessFunc);
    v->Visit("init_secondOpWriteAccessFunction", &_secondOpWriteAccessFunc);
    v->Visit("init_readProducerPos", &_readProducerPos);

    v->Visit("commonIters", &commonIters);
    v->Visit("firstOpIters", &firstOpIters);
    v->Visit("secondOpIters", &secondOpIters);
    v->Visit("firstOpNumLoops", &firstOpNumLoops);
    v->Visit("secondOpNumLoops", &secondOpNumLoops);
    v->Visit("firstOpReadAccessFuncs", &firstOpReadAccessFuncs);
    v->Visit("secondOpReadAccessFuncs", &secondOpReadAccessFuncs);
    v->Visit("firstOpWriteAccessFunc", &firstOpWriteAccessFunc);
    v->Visit("secondOpWriteAccessFunc", &secondOpWriteAccessFunc);
    v->Visit("shareRelations", &shareRelations);
    v->Visit("splitRelations", &splitRelations);
  }
  /*!
   *  \brief Set the tiling factors for the first op.
   *  \param factors(List[int]): the tiling factors for the first op
   */
  void setFirstOpTiling(Array<IntImm> factors);
  /*!
   *  \brief Set the tiling factors for the second op.
   *  \param factors(List[int]): the tiling factors for the second op
   */
  void setSecondOpTiling(Array<IntImm> factors);
  /*!
   *  \brief the permution of first op's outer loops
   *  \param permutation: new order of the iterators, new ith object contains
   * permutation[i]
   */
  void setFirstOpPermute(Array<IntImm> permutation);
  /*!
   *  \brief the permution of second op's outer loops
   *  \param permutation: new order of the iterators, new ith object contains
   * permutation[i]
   */
  void setSecondOpPermute(Array<IntImm> permutation);
  /*!
   *  \brief op1 compute_at op2's attach_pos's loop
   */
  void setAttach(size_t attach_pos);
  /*!
   * \brief set the fusion config in one run
   */
  void setFusion(FusionItem fusionItem);
  /*!
   * \brief apply all the schedules, solve conflicts in schedules
   */
  void applyAll();

  /*! \brief get the searchSpace */
  FusionSpace getSearchSpace();

  /*The feature extraction*/
  /*! \brief get the bounds of all iters */
  Map<tir::Var, IntImm> inferBound() const;
  /*! \brief get the number of inner blocks */
  int getNumOfBlocks() const;
  /*! \brief get the parallelism */
  int getParallelism() const;
  /*! \brief get the first Op's memvisit*/
  int getFirstOpDataVolume() const;
  /*! \brief get the first Op's workload*/
  int getFirstOpWorkload() const;
  /*! \brief get the first Op's blockSize*/
  int getFirstOpBufferSize() const;
  /*! \brief get the second Op's memvisit*/
  int getSecondOpDataVolume() const;
  /*! \brief get the second Op's workload*/
  int getSecondOpWorkload() const;
  /*! \brief get the second Op's blockSize*/
  int getSecondOpBufferSize(bool writeThrough = true) const;
  /*! \brief get the redundant compute volume*/
  int getRedundancy() const;
  /*! \brief get the analytical result */
  FusionResult getAnalyticalResult(hardware::HardwareParam hw_param, int bytePerEle, bool writeThrough = true);
  /*! \brief looplike lightweight visualize */
  void visualize();
  /*! \brief write result */
  void writeResult(FusionResult res);
  static constexpr const char *_type_key = "ditto.auto_tensorize.IterGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterGraphNode, Object);
};

class IterGraph : public ObjectRef {
public:
  TVM_DLL IterGraph(Array<IterVar> firstOpIters, Array<IterVar> secondOpIters,
                    Array<Share> sharedIterPairs,
                    Array<AccessFunction> firstOpReadAccessFuncs,
                    Array<AccessFunction> secondOpReadAccessFuncs,
                    AccessFunction firstOpWriteAccessFunc,
                    AccessFunction secondOpWriteAccessFunc, int readProducerPos,
                    te::Operation op1, te::Operation op2, Array<IterVar> tensorizeIters, String path = "");
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IterGraph, ObjectRef, IterGraphNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterGraphNode);
};
} // namespace auto_tensorize

} // namespace ditto