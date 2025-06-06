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
  FIRSTSPATIAL = 0,
  SECONDSPATIAL = 1,
  REDUCE = 2,
  TENSORIZE = 3
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
  /*! \brief vars doesn't appear in this tensor*/
  Array<tir::Var> absentVars;
  /*! \brief present vars*/
  Array<tir::Var> presentVars;
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("op", &op);
    v->Visit("access_indices", &access_indices);
    v->Visit("absent vars", &absentVars);
  }
  /*! \brief get the data footprint */
  std::vector<int> getFootprint(Map<tir::Var, IntImm> bounds);
  /*! \brief get the workload */
  int getWorkload(Map<tir::Var, IntImm> bounds);
  /*! \brief get the product of absent axis*/
  int getProductOfAbsentVars(Map<tir::Var, IntImm> bounds);
  /*! \brief replace vars by map */
  void repalceVars(Map<tir::Var, tir::Var> map);
  /*! \brief set the absent vars */
  void setAbsentVars(Array<tir::Var> newAbsentVars);
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
                         Array<Array<PrimExpr>> access_indices,
                         Array<tir::Var> absentVars,
                         Array<tir::Var> presentVars);

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

  bool isSpatial() {
    return iv_type == IV_Type::FIRSTSPATIAL ||
           iv_type == IV_Type::SECONDSPATIAL;
  }
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
  Array<IterVar> firstOpTensorizeIters;
  Array<IterVar> secondOpTensorizeIters;
  /*! \brief the cost for read/write tensor */
  std::vector<double> tensorWeight;
  size_t attachPos = 0; // default: independent loops
  bool configSet; // the config is set
  std::vector<double> cacheSizes;
  std::vector<double> cacheBandwidth;
  int fusionLevel;
  int totalFp;
  double parallelism_ = -1;
  std::unordered_map<int, IntImm> _parallelSchedule;
  Map<tir::Var, IntImm> _boundsAfterParallel;
  bool writeThrough; // whether E matrix is directly written back to global memory
  int bytePerEle;
  double _outerCost; // singleThreadWorkload * ceil (n_thread / n_core)
  double _maxThreadIter; // ceil (n_thread / n_core)
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
   *  \brief Set the total footprint
   */
  void setTotalFp();
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
  /*!
  *  \brief get the vertical & horizontal datamovement 
  */ 
  double getTotalDM(std::unordered_map<std::string, double> & features);
  /*! \brief get the searchSpace */
  FusionSpace getSearchSpace();

  /*The feature extraction*/
  /*! \brief get the bounds of all iters */
  Map<tir::Var, IntImm> inferBound() const;
  /*! \brief get the number of inner blocks */
  double getNumOfBlocks() const;
  /*! \brief get the parallelism */
  double getParallelism() const;
  /*! \brief get the first Op's memvisit*/
  double getFirstOpDataVolume() const;
  /*! \brief get the first Op's workload*/
  double getFirstOpWorkload() const;
  /*! \brief get the first Op's blockSize*/
  double getFirstOpBufferSize(bool considerWrite = false) const;
  /*! \brief get the second Op's memvisit*/
  double getSecondOpDataVolume() const;
  /*! \brief get the second Op's workload*/
  double getSecondOpWorkload() const;
  /*! \brief get the second Op's blockSize*/
  double getSecondOpBufferSize() const;
  /*! \brief get the redundant compute volume*/
  double getRedundancy() const;
  /*! \brief manually set the fusion level*/
  void setFusionLevel(int fusionLevel_);
  /*! \brief fusionLevels*/
  std::vector<int> fusionLevels;
  /*! \brief get the analytical result */
  FusionResult getAnalyticalResult();
  /*! \brief get the data movement*/
  std::pair<bool, double> getCost(double *occupancy = NULL, double * parallelism = NULL, double * memUse = NULL);
  /*! \brief looplike lightweight visualize */
  void visualize();
  /*! \brief scheduleOuterParallel */
  void scheduleParallel();
  /*! \brief get the problem size after parallel */
  Map<tir::Var, IntImm> getPbsz();
  /*! \brief set hardware related param */
  void setConfig(hardware::HardwareParam hw_param, int bytePerEle_){
    bytePerEle = bytePerEle_;
    configSet = true;
    if (hw_param->platform == "CPU"){
      cacheSizes = hw_param->cacheSizePerThread;
      tensorWeight = hw_param->tensorWeight;
      parallelism_ = hw_param->num_groups;
      cacheBandwidth = hw_param->cacheBandwidth;
      fusionLevels.clear();
      fusionLevel = -1;
      for (int i = 1; i < (int)cacheSizes.size(); i++)
        fusionLevels.push_back(i);
      writeThrough = false;
    }
    else if (hw_param->platform == "NVGPU"){
      cacheSizes.clear();
      cacheSizes.push_back(hw_param->shared_memory_per_group_kb * 1e3);
      
      tensorWeight = hw_param->tensorWeight;
      parallelism_ = hw_param->num_groups * hw_param->num_processors_per_group;
      cacheBandwidth.clear();
      cacheBandwidth.push_back(hw_param->global_memory_bandwidth_gbs * 1e9);
      fusionLevels.clear();
      fusionLevels.push_back(0);
      fusionLevel = -1;
      writeThrough = true;
      std::cout << "cacheSizes: " << cacheSizes[0] << std::endl;
      std::cout << "parallel_: " << parallelism_ << std::endl;

    }
    else {
      throw Error(hw_param->platform + "not supported");
    }
  }
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
                    te::Operation op1, te::Operation op2,
                    Array<IterVar> firstOpTensorizeIters,
                    Array<IterVar> secondOpTensorizeIters);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IterGraph, ObjectRef, IterGraphNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IterGraphNode);
};
} // namespace auto_tensorize

} // namespace ditto