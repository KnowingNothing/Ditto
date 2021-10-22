#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <hybrid/tree.h>
#include <hybrid/test.h>
#include <hybrid/hybrid_schedule.h>
#include "graph.h"

#include <stack>
#include <unordered_set>

using namespace tvm;
using namespace te;

namespace ditto {
namespace hybrid{

// for test for tree
void HybridStage::Test(){
  test a(1), b(2), c(3), d(4), e(5), f(6), g(7);

  Tree<test> t;
  t.insertChild(a);
  t.insertChild(a, b);
  t.insertChild(a, c);
  t.insertChild(b, d);
  t.insertChild(b, e);
  t.insertChild(e, f);

  // std::cout << "a is b' s imm parent? " <<   t.is_immediate_parent(a, b) << std::endl; 
  // std::cout << "c is a' s imm parent? " <<   t.is_immediate_parent(c, a) << std::endl; 
  // std::cout << "num of a' s child" << t.count_children(a) << std::endl;
  // std::cout << "test a node not in tree" << t.count_children(g) << std::endl;
  t.display("before replace");
  t.replace(f, g);
  t.display("after replace");
  Tree<test> b_subTree = t.getSubTree(b);
  std::cout << "parent correct after subTree ? " << t.check_parent(false) << std::endl;
  b_subTree.display("b_subTree");
  Tree<test> t_(b_subTree);
  t_.display("t_");
  t_.apply([](test & tmp)->void{
    tmp.operator->()->data += 1;
  });
  t_.display("t_ after + 1");
  t.display("before insert");
  t.insertTree(t_);
  t.display("after insert");
  t.eraseTree(b);
  t.display("after erase Tree");
  // std::cout << "a is b's parent" << t.is_parent(a, b) << std::endl;
  // std::cout << "c is a's parent" << t.is_parent(c, a) << std::endl;
  // t.display("t");
  // t.erase(a);
  // t.display("t after erase");
  // Tree<test> t_ = Tree<test>(t);
  // t_.display("t_ copied");
  // t_.insert(a);
  // t_.display("t_ after insert");
  // t.setValue(f, g);
  // t.display("t");
  // t_.display("t_");
}
void HybridStage::display(std::string s){
  (*this)->leaf_iter_vars_tree.display((*this)->op->name, s);
}
// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->size(); ++i) {
    if (array_node->at(i).get() == n) return i;
  }
  return array_node->size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars, const IterVar& v) {
  size_t pos = FindNodeRef(leaf_vars, v);
  if (pos < leaf_vars->size()) return pos;

  if (FindNodeRef(all_vars, v) < all_vars->size()) {
    LOG(FATAL) << "Operate on iter var " << v << "that has already been split";
  } else {
    LOG(FATAL) << "Operate on iter var " << v << "that is not part of the schedule";
  }
  return 0;
}

DataType MatchDataType(std::vector<DataType> dtypes) {
  int max_bits = -1;
  for (const auto& dtype : dtypes) {
    ICHECK(dtype.is_int());
    ICHECK(dtype.is_scalar());
    max_bits = std::max(max_bits, dtype.bits());
  }
  return DataType::Int(max_bits);
}
// ** changed, need testing.
void SplitHelper(HybridStageNode* self, IterVar parent, PrimExpr factor, PrimExpr nparts,
                 IterVar* p_outer, IterVar* p_inner) {
  // Check if split is valid.
  ICHECK(parent->iter_type == kDataPar || parent->iter_type == kCommReduce ||
         parent->iter_type == kOrdered)
      << "Cannot split on " << IterVarType2String(parent->iter_type);
  IterVar outer = IterVar(Range(), parent->var.copy_with_suffix(".outer"), parent->iter_type);
  IterVar inner = IterVar(Range(), parent->var.copy_with_suffix(".inner"), parent->iter_type);
  *p_outer = outer;
  *p_inner = inner;
  // The splits
  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  Tree<IterVar>& leaf_vars_tree = self->leaf_iter_vars_tree;
  size_t pos = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), parent);
  self->relations.push_back(Split(parent, outer, inner, factor, nparts));
  // add vars to all vars
  all_vars.push_back(outer);
  all_vars.push_back(inner);
  // replace the position.
  leaf_vars.erase(leaf_vars.begin() + pos);
  leaf_vars.insert(leaf_vars.begin() + pos, inner);
  leaf_vars.insert(leaf_vars.begin() + pos, outer);
  leaf_vars_tree.replace(parent, outer);
  leaf_vars_tree.insert(outer, inner);
}
// spd modify
HybridStage::HybridStage(Operation op) {
  auto n = make_object<HybridStageNode>();
  n->op = op;
  n->origin_op = op;
  n->all_iter_vars = op->root_iter_vars();
  // remove opaque var from leaf.
  Array<IterVar> clean;
  for (IterVar iv : n->all_iter_vars) {
    if (iv->iter_type != kOpaque) clean.push_back(iv);
  }
  if (clean.size() == n->all_iter_vars.size()) {
    n->leaf_iter_vars = n->all_iter_vars;
  } else {
    n->leaf_iter_vars = clean;
  }

  if(n->leaf_iter_vars.size()>0){
    n->leaf_iter_vars_tree.insert(n->leaf_iter_vars[0]);
    for(size_t i = 1; i < n->leaf_iter_vars.size(); i++) {
      n->leaf_iter_vars_tree.insert(n->leaf_iter_vars[i-1], n->leaf_iter_vars[i]);
    }
  }
  data_ = std::move(n);
}

bool HybridStage::is_scheduled() const {
  const HybridStageNode* n = operator->();
  return !(n->relations.empty() && n->attach_type == kGroupRoot &&
           n->all_iter_vars.same_as(n->leaf_iter_vars));
}

HybridStage HybridStage::GetAttachSpec() const {
  HybridStage attach_spec = *this;
  while (attach_spec->attach_type == kGroupRoot && attach_spec->group.defined()) {
    attach_spec = attach_spec->group;
  }
  return attach_spec;
}

HybridStage& HybridStage::set_scope(std::string scope) {  // NOLINT(*)
  (*this)->scope = scope;
  return *this;
}

HybridStage& HybridStage::slice(
  TreeUnitNode<IterVar>* slicept, 
  TreeUnitNode<IterVar>* pinpt, 
  PrimExpr factor, 
  std::string mode, 
  Array<IterVar> *left, 
  Array<IterVar> *right
){
  HybridStageNode* self = operator->();
  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  Tree<IterVar>& leaf_vars_tree = self->leaf_iter_vars_tree;
  // checking
  ICHECK(leaf_vars_tree.is_ancestor(pinpt, slicept)) << "slice pt not in the subtree of the pin point.";

  Tree<IterVar> subTree = leaf_vars_tree.getSubTree(pinpt);

  for(TreeUnitNode<IterVar>* iter = subTree.getBase(); iter->pChild != NULL; iter = iter->pChild){
    ICHECK(iter->count_child() == 1) << "cannot slice when exist node on the pinpt-slicept path that has more than 1 children.";
  }

  Tree<IterVar> l(subTree, [](const IterVar & e)->IterVar{
      return IterVar(Range(), e->var.copy_with_suffix(".left"), e->iter_type);
  });
  Tree<IterVar> r(subTree, [](const IterVar & e)->IterVar{
      return IterVar(Range(), e->var.copy_with_suffix(".right"), e->iter_type);
  });

  Array<IterVar> old;
  // add new nodes to all_vars, leaf_vars
  l.apply([&all_vars, &leaf_vars, left](IterVar & t)->void{
    all_vars.push_back(t);
    leaf_vars.push_back(t);
    left->push_back(t);
  }, "RootFirst");
  r.apply([&all_vars, &leaf_vars, right](IterVar & t)->void{
    all_vars.push_back(t);
    leaf_vars.push_back(t);
    right->push_back(t);
  }, "RootFirst");

  old.push_back(*(pinpt->data_ptr));

  // Remove old tree from leaf_vars
  leaf_vars_tree.getSubTree(pinpt->pChild).apply([&all_vars, &leaf_vars, &old](IterVar & t){
    size_t pos = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), t);
    leaf_vars.erase(leaf_vars.begin() + pos);
    old.push_back(t);
  }, "RootFirst");

  self->relations.push_back(Slice(old, *left, *right, *slicept->data_ptr, *pinpt->data_ptr, mode, factor));
  
  leaf_vars_tree.eraseTree(pinpt->pChild);
  leaf_vars_tree.insertTree(pinpt, l);
  leaf_vars_tree.insertTree(pinpt, r);
  //   TVM_DLL Slice(Array<IterVar> old, Array<IterVar> left, Array<IterVar> right, IterVar slicept, IterVar pinpt, std::string mode, PrimExpr factor);
  return *this;
}

HybridStage& HybridStage::slice(
  IterVar slicept, 
  IterVar pinpt, 
  PrimExpr factor, 
  std::string mode, 
  Array<IterVar> *left, 
  Array<IterVar> *right
){
  Tree<IterVar> & tree = (*this)->leaf_iter_vars_tree;
  return slice(tree.getUnit(slicept), tree.getUnit(pinpt), factor, mode, left, right);
  /*
  deepcopy(f)
  left,right = pinpt, deepCopy(f) * 2
  all vars push back subTree(slice pt)
  leaf vars . remove (pinpt)
  while (slicept. parent is not pinpt) {
    parent_l, parent_r = slicept .parent . deepcopy() * 2
    left . setparent (slice pt . parent)
    right . setparent (slice pt . parent)
    delete (slicept .parent)
    slice pt = slice pt. parent
    all vars push back (parent_l)
    leaf vars push back (paraent_l, parent_r )
  }
  pinpt insert child(left, right)
  */
}

HybridStage& HybridStage::slice(
  IterVar slicept, 
  PrimExpr factor, 
  std::string mode, 
  Array<IterVar> *left, 
  Array<IterVar> *right
){
  Tree<IterVar> & tree = (*this)->leaf_iter_vars_tree;
  return slice(tree.getUnit(slicept), tree.getUnit(slicept), factor, mode, left, right);
}

HybridStage& HybridStage::compute_at(HybridStage parent, IterVar scope) {  // NOLINT(*)
  ICHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  // Group constraint checking.
  HybridStage group = (*this)->group;
  if (group.defined()) {
    HybridStage pg = parent->group;
    while (pg.defined() && !pg.same_as(group)) {
      pg = pg->group;
    }
    ICHECK(pg.same_as(group)) << "Can only assign compute_at to hybrid_stages within the same group";
  }

  (*this)->attach_type = kScope;
  (*this)->attach_ivar = scope;
  (*this)->attach_stage = parent;
  bool found = false;
  for (size_t i = 0; i < parent->leaf_iter_vars.size(); ++i) {
    if (scope == parent->leaf_iter_vars[i]) {
      found = true;
      break;
    }
  }
  ICHECK(found) << "Cannot find the axis " << scope << " in parent's leaf_iter_vars"
                << " parent=" << parent;
  return *this;
}

HybridStage& HybridStage::compute_inline() {  // NOLINT(*)
  ICHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kInline;
  return *this;
}

HybridStage& HybridStage::compute_root() {  // NOLINT(*)
  ICHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kGroupRoot;
  return *this;
}

HybridStage& HybridStage::bind(IterVar ivar, IterVar thread_ivar) {  // NOLINT(*)
  HybridStageNode* self = operator->();
  ICHECK(ivar->iter_type == kDataPar || ivar->iter_type == kCommReduce)
      << "Cannot bind " << IterVarType2String(ivar->iter_type) << " to thread";
  ICHECK(thread_ivar->iter_type == kThreadIndex)
      << "Cannot rebase by " << IterVarType2String(ivar->iter_type)
      << ", only thread axis is allowed so far";
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, ivar);

  auto it = self->iter_var_attrs.find(ivar);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
    if (n->bind_thread.defined() && !n->bind_thread.same_as(thread_ivar)) {
      LOG(WARNING) << "Axis " << ivar << " is already bind to another thread " << n->bind_thread;
    }
  } else {
    n = make_object<IterVarAttrNode>();
  }
  n->bind_thread = thread_ivar;
  self->iter_var_attrs.Set(ivar, IterVarAttr(n));
  return *this;
}

HybridStage& HybridStage::env_threads(Array<IterVar> threads) {
  HybridStageNode* self = operator->();
  ICHECK(self->op.defined() && self->op.as<ScanOpNode>())
      << "env_threads is only valid for composite ops such as ScanOp";
  ICHECK_EQ(self->env_threads.size(), 0U) << "Already set env_threads";
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  Array<IterVar>& all_vars = self->all_iter_vars;
  Tree<IterVar>& leaf_vars_tree = self->leaf_iter_vars_tree;
  std::vector<ObjectRef> temp;
  for (IterVar iv : threads) {
    temp.push_back(iv);
  }
  leaf_vars.insert(leaf_vars.begin(), temp.begin(), temp.end());
  all_vars.insert(all_vars.end(), temp.begin(), temp.end());
  for(size_t i = 0; i < temp.size(); i++){
    leaf_vars_tree.insert(leaf_vars[temp.size() - i - 1]);
  }
  self->env_threads = threads;
  return *this;
}

HybridStage& HybridStage::set_store_predicate(PrimExpr predicate) {
  HybridStageNode* self = operator->();
  self->store_predicate = predicate;
  return *this;
}

HybridStage& HybridStage::split(IterVar parent, PrimExpr factor, IterVar* p_outer,
                    IterVar* p_inner) {  // NOLINT(*)
  SplitHelper(operator->(), parent, factor, PrimExpr(), p_outer, p_inner);
  return *this;
}

HybridStage& HybridStage::split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer,
                              IterVar* p_inner) {  // NOLINT(*)
  SplitHelper(operator->(), parent, PrimExpr(), nparts, p_outer, p_inner);
  return *this;
}

HybridStage& HybridStage::fuse(IterVar outer, IterVar inner, IterVar* p_target) {  // NOLINT(*)
  HybridStageNode* self = operator->();
  ICHECK(outer->iter_type == kDataPar || outer->iter_type == kCommReduce ||
         outer->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(outer->iter_type);
  ICHECK(inner->iter_type == kDataPar || inner->iter_type == kCommReduce ||
         inner->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(inner->iter_type);

  IterVarType iter_type = outer->iter_type;
  if (inner->iter_type > iter_type) iter_type = inner->iter_type;
  std::string fused_name = outer->var->name_hint + "." + inner->var->name_hint + ".fused";
  DataType iter_dtype = MatchDataType({inner->var.dtype(), outer->var.dtype()});

  IterVar fused = IterVar(Range(), Var(fused_name, iter_dtype), iter_type);

  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  Tree<IterVar>& leaf_vars_tree = self->leaf_iter_vars_tree;

  size_t pos_inner = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), inner);
  size_t pos_outer = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), outer);
  if(leaf_vars_tree.is_immediate_parent(inner, outer)){
    std::swap(outer, inner);
    std::swap(pos_inner, pos_outer);
  }
  ICHECK(leaf_vars_tree.is_immediate_parent(outer, inner))
      << "Can only fuse iterations that are consecutive between each other";
  ICHECK_EQ(leaf_vars_tree.count_children(outer), 1)
      << "Can only fuse iterations that have only 1 child iter_var";
  self->relations.push_back(Fuse(outer, inner, fused));
  all_vars.push_back(fused);
  leaf_vars.erase(leaf_vars.begin() + pos_outer, leaf_vars.begin() + pos_outer + 1);
  pos_inner = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), inner);
  leaf_vars.erase(leaf_vars.begin() + pos_inner, leaf_vars.begin() + pos_inner + 1);
  leaf_vars.insert(leaf_vars.begin() + pos_inner, fused);
  leaf_vars_tree.erase(inner);
  leaf_vars_tree.replace(outer, fused);
  *p_target = fused;
  return *this;
}

HybridStage& HybridStage::fuse(const Array<IterVar>& axes, IterVar* p_target) {  // NOLINT(*)
  if (axes.size() != 0) {
    IterVar fused = axes[0];
    for (size_t i = 1; i < axes.size(); ++i) {
      this->fuse(fused, axes[i], &fused);
    }
    *p_target = std::move(fused);
  } else {
    HybridStageNode* self = operator->();
    // special handle fuse empty array.
    // insert at the outer most loop
    IterVar singleton =
        IterVar(Range::FromMinExtent(0, 1), Var("singleton", DataType::Int(32)), kDataPar);
    self->relations.push_back(Singleton(singleton));
    Array<IterVar>& all_vars = self->all_iter_vars;
    Array<IterVar>& leaf_vars = self->leaf_iter_vars;
    Tree<IterVar>& leaf_vars_tree = self->leaf_iter_vars_tree;
    all_vars.push_back(singleton);
    leaf_vars.insert(leaf_vars.begin(), singleton);
    leaf_vars_tree.insert(singleton);
    *p_target = singleton;
  }
  return *this;
}

HybridStage& HybridStage::reorder(const Array<IterVar>& order) {  // NOLINT(*)
  std::unordered_set<IterVar> seen_var;
  HybridStageNode* self = operator->();
  for (IterVar iv : order) {
    ICHECK(iv->iter_type == kDataPar || iv->iter_type == kCommReduce ||
           iv->iter_type == kThreadIndex)
        << "Cannot reorder IterVar(" << IterVarType2String(iv->iter_type) << ")";

    ICHECK_EQ(seen_var.count(iv), 0) << "Same axis can not appear more than once " << iv;
    seen_var.insert(iv);
  }
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  Tree<IterVar>& leaf_vars_tree = self->leaf_iter_vars_tree;

  std::vector<size_t> pos;

  // reorder on array
  for (size_t i = 0; i < order.size(); ++i) {
    pos.push_back(FindLeafVar(all_vars, leaf_vars, order[i]));
  }
  std::vector<ObjectRef> temp;
  for (size_t i = 0; i < pos.size(); ++i) {
    temp.emplace_back(leaf_vars->at(pos[i]));
  }
  std::sort(pos.begin(), pos.end());
  for (size_t i = 0; i < pos.size(); ++i) {
    leaf_vars->SetItem(pos[i], temp[i]);
  }

  // reorder on tree
  std::vector<int> children_count;
  for (size_t i = 0; i < order.size(); i++) {
    children_count.push_back(0);
    for(size_t j = 0; j < order.size(); j++)
      if(i != j && leaf_vars_tree.is_ancestor(order[i], order[j])) 
        children_count[i]++;
  }
  for (size_t i = 0; i < order.size(); i++)
    ICHECK_EQ(std::count(children_count.begin(), children_count.end(), i), 1)
      << "Reorder must perform on a single path on IterVar tree.";
  size_t deepest = 0;
  for (size_t i = 0; i < order.size(); i++)
    if(children_count[i] < children_count[deepest]) deepest = i;
  size_t shallowest = 0;
  for (size_t i = 0; i < order.size(); i++)
    if(children_count[i] > children_count[shallowest]) shallowest = i;
  TreeUnitNode<IterVar>* ptr = leaf_vars_tree.getUnit(order[deepest]);
  while(!(*(ptr->data_ptr)).same_as(order[shallowest])){
    ptr = leaf_vars_tree.getUnit(*(leaf_vars_tree.get_parent_ptr(*(ptr->data_ptr))));
    ICHECK_EQ(leaf_vars_tree.count_children(*(ptr->data_ptr)), 1)
      << "Reorder must perform on a path without branches.";
  }
  std::vector<TreeUnitNode<IterVar>*> unit_ptr;
  for (size_t i = 0; i < order.size(); i++) {
    unit_ptr.push_back(leaf_vars_tree.getUnit(order[i]));
  }
  std::vector<IterVar*> iter_ptr;
  for (size_t i = 0; i < order.size(); i++) {
    iter_ptr.push_back(unit_ptr[i]->data_ptr);
  }
  for (size_t i = 0; i < order.size(); i++) {
    unit_ptr[i]->data_ptr = iter_ptr[order.size()-children_count[i]-1];
  }
  
  return *this;
}


// ** change **
HybridStage& HybridStage::tile(IterVar x_parent, IterVar y_parent, PrimExpr x_factor, PrimExpr y_factor,
                   IterVar* p_x_outer, IterVar* p_y_outer, IterVar* p_x_inner, IterVar* p_y_inner) {
  split(x_parent, x_factor, p_x_outer, p_x_inner);
  split(y_parent, y_factor, p_y_outer, p_y_inner);
  reorder(Array<IterVar>({*p_x_outer, *p_y_outer, *p_x_inner, *p_y_inner}));
  return *this;
}

template <typename FUpdate>
inline void UpdateIterVarAttr(HybridStageNode* self, IterVar var, FUpdate fupdate,
                              bool need_leaf = true) {
  if (need_leaf) {
    ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
    ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
    FindLeafVar(all_vars, leaf_vars, var);
  }
  auto it = self->iter_var_attrs.find(var);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_object<IterVarAttrNode>();
  }
  fupdate(n.get());
  self->iter_var_attrs.Set(var, IterVarAttr(n));
}

inline void SetAttrIterType(HybridStageNode* self, IterVar var, IterVarType iter_type) {
  UpdateIterVarAttr(self, var, [iter_type](IterVarAttrNode* n) { n->iter_type = iter_type; });
}

HybridStage& HybridStage::vectorize(IterVar var) {  // NOLINT(*)
  ICHECK(var->iter_type == kDataPar || var->iter_type == kOpaque || var->iter_type == kUnrolled ||
         var->iter_type == kVectorized || var->iter_type == kTensorized ||
         var->iter_type == kParallelized)
      << "Cannot vectorize on " << IterVarType2String(var->iter_type);
  SetAttrIterType(operator->(), var, kVectorized);
  return *this;
}

HybridStage& HybridStage::tensorize(IterVar var, TensorIntrin f) {  // NOLINT(*)
  UpdateIterVarAttr(operator->(), var, [f](IterVarAttrNode* n) {
    n->iter_type = kTensorized;
    n->tensor_intrin = f;
  });
  return *this;
}

HybridStage& HybridStage::unroll(IterVar var) {  // NOLINT(*)
  SetAttrIterType(operator->(), var, kUnrolled);
  return *this;
}

HybridStage& HybridStage::parallel(IterVar var) {  // NOLINT(*)
  SetAttrIterType(operator->(), var, kParallelized);
  return *this;
}

HybridStage& HybridStage::pragma(IterVar var, const std::string& pragma_type,
                     const PrimExpr& pragma_value) {  // NOLINT(*)
  if (pragma_type == "unroll") {
    this->unroll(var);
  } else if (pragma_type == "vectorize") {
    this->vectorize(var);
  } else {
    UpdateIterVarAttr(operator->(), var, [pragma_type, pragma_value](IterVarAttrNode* n) {
      n->pragma_keys.push_back(tir::StringImm(pragma_type));
      n->pragma_values.push_back(pragma_value);
    });
  }
  return *this;
}

HybridStage& HybridStage::prefetch(const Tensor& tensor, IterVar var, PrimExpr offset) {
  HybridStageNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, var);
  auto it = self->iter_var_attrs.find(var);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_object<IterVarAttrNode>();
  }
  n->prefetch_data.push_back(tensor);
  n->prefetch_offset.push_back(offset);
  self->iter_var_attrs.Set(var, IterVarAttr(n));
  return *this;
}

HybridStage& HybridStage::storage_align(IterVar axis, int factor, int offset) {
  HybridStageNode* self = operator->();
  UpdateIterVarAttr(
      self, axis,
      [factor, offset](IterVarAttrNode* n) {
        n->dim_align_factor = factor;
        n->dim_align_offset = offset;
      },
      false);
  return *this;
}

HybridStage& HybridStage::double_buffer() {
  HybridStageNode* self = operator->();
  ICHECK(!self->is_output) << "Cannot apply double buffer on output";
  self->double_buffer = true;
  return *this;
}

HybridStage CopyStage(const HybridStage& s) {
  ObjectPtr<HybridStageNode> n = make_object<HybridStageNode>(*s.operator->());
  return HybridStage(n);
}

HybridSchedule HybridSchedule::copy() const {
  // map of hybrid_stages.
  const HybridScheduleNode* self = operator->();
  std::unordered_map<HybridStage, HybridStage, ObjectPtrHash, ObjectPtrEqual> smap;
  ObjectPtr<HybridScheduleNode> n = make_object<HybridScheduleNode>();
  n->outputs = self->outputs;
  // Copy the hybrid_stages.
  for (HybridStage s : self->stages) {
    HybridStage scopy = CopyStage(s);
    smap[s] = scopy;
    n->stages.push_back(scopy);
  }
  for (HybridStage g : self->groups) {
    HybridStage gcopy = CopyStage(g);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }
  // Remaps the reference relations.
  for (auto kv : self->stage_map) {
    n->stage_map.Set(kv.first, smap.at(kv.second));
  }
  for (HybridStage s : n->stages) {
    if (s->attach_stage.defined()) {
      ICHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      ICHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  for (HybridStage s : n->groups) {
    if (s->attach_stage.defined()) {
      ICHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      ICHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  return HybridSchedule(n);
}

HybridStage HybridSchedule::operator[](const Operation& op) {
  auto it = (*this)->stage_map.find(op);
  ICHECK(it != (*this)->stage_map.end())
      << "Cannot find HybridStage for operator " << op << " in the hybridschedule";
  return (*it).second;
}

HybridStage LeastCommonAncestor(HybridStage g1, HybridStage g2) {
  if (!g1.defined()) return g1;
  if (!g2.defined()) return g2;
  if (g1.same_as(g2)) return g1;
  HybridStage g = g1;
  while (g.defined()) {
    if (g.same_as(g2)) return g2;
    g = g->group;
  }
  g = g2;
  while (g.defined()) {
    if (g.same_as(g1)) return g1;
    g = g->group;
  }
  return g;
}

Array<Tensor> RemapTensor(HybridScheduleNode* self, const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      ICHECK(self->stage_map.count(t->op)) << "Given tensor is not in the hybrid_schedule plan";
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// Group the hybrid_schedule hybrid_stages.
HybridStage HybridSchedule::create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs) {
  HybridScheduleNode* self = operator->();
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  // Get the ops.
  Array<Operation> ops =
      hybrid::GetSubGraph(RemapTensor(self, outputs), RemapTensor(self, inputs), include_inputs);
  // local counter entry
  // Automatically initialize to 0 during creation.
  struct Entry {
    int count{0};
  };
  // Map of group->touched counter
  std::unordered_map<HybridStage, Entry, ObjectPtrHash, ObjectPtrEqual> counter;
  // The parent group;
  HybridStage parent_group;
  // Detect common parent and child.
  for (size_t i = 0; i < ops.size(); ++i) {
    Operation op = ops[i];
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
    HybridStage op_group = it->second->group;
    if (i == 0) {
      parent_group = op_group;
    } else {
      parent_group = LeastCommonAncestor(parent_group, op_group);
    }
    if (op_group.defined()) {
      ++counter[op_group].count;
    }
  }
  // Create the new group hybrid_stage.
  HybridStage gstage(make_object<StageNode>());
  gstage->group = parent_group;
  if (parent_group.defined()) {
    ++parent_group->num_child_stages;
  }
  // Propagate the counter statistics from by checking if subgroup
  // Is full and propagate.
  std::vector<HybridStage> stack;
  for (auto& kv : counter) {
    if (!kv.first.same_as(parent_group)) {
      if (kv.first->num_child_stages == kv.second.count) {
        stack.push_back(kv.first);
      }
    }
  }
  while (!stack.empty()) {
    HybridStage g = stack.back();
    stack.pop_back();
    if (g->group.defined() && !g->group.same_as(parent_group)) {
      Entry& e = counter[g->group];
      ++e.count;
      if (e.count == g->group->num_child_stages) {
        stack.push_back(g->group);
      }
    }
  }
  // Verification and remappig the subgroups.
  for (auto& kv : counter) {
    if (kv.first.same_as(parent_group)) continue;
    ICHECK_EQ(kv.first->num_child_stages, kv.second.count)
        << "Trying to group region that intersect with an already existed group";
    if (kv.first->group.same_as(parent_group)) {
      HybridStage s = kv.first;
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Remap the group of op hybrid_stages.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
    HybridStage s = it->second;
    if (s->group.same_as(parent_group)) {
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Correct the attach to keep everything in group.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
    HybridStage s = it->second;
    if (s->attach_type == kScope) {
      HybridStage cg = LeastCommonAncestor(s->attach_stage->group, gstage);
      if (!cg.same_as(gstage)) {
        LOG(WARNING) << "group invalidates some previous compute_at relation "
                     << " and keeps things to be computed inside the group";
        s.compute_root();
      }
    }
  }

  self->groups.push_back(gstage);
  return gstage;
}

void HybridScheduleNode::InvalidateCache() { op2stage_cache_.clear(); }

void HybridScheduleNode::InitCache() {
  if (op2stage_cache_.size() == stages.size()) return;
  InvalidateCache();
  for (HybridStage s : stages) {
    if (s->op.defined()) {
      op2stage_cache_[s->op.get()] = s;
    }
  }
  ICHECK_EQ(op2stage_cache_.size(), stages.size());
}

bool HybridScheduleNode::Contain(const Operation& op) const {
  return stage_map.find(op) != stage_map.end();
}

HybridSchedule::HybridSchedule(Array<Operation> ops) {
  auto n = make_object<HybridScheduleNode>();
  data_ = n;
  n->outputs = ops;
  auto g = hybrid::CreateReadGraph(n->outputs);
  Array<Operation> post_order = hybrid::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    HybridStage stage(op);
    stage->is_output = output_set.count(op) != 0;
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
    // mark scan updates.
    if (const ScanOpNode* scan = op.as<ScanOpNode>()) {
      Array<Tensor> inputs;
      for (Tensor t : scan->state_placeholder) {
        inputs.push_back(t);
      }
      for (Tensor t : scan->inputs) {
        inputs.push_back(t);
      }
      // Create the scan group.
      HybridStage scan_group = this->create_group(scan->update, inputs, false);
      scan_group->attach_type = kScanUpdate;
      scan_group->attach_stage = stage;

      for (size_t i = 0; i < scan->update.size(); ++i) {
        HybridStage s = n->stage_map[scan->update[i]->op];
        ICHECK(scan_group.same_as(s->group));
      }
    }
  }
}

Slice::Slice(Array<IterVar> old, Array<IterVar> left, Array<IterVar> right, IterVar slicept, IterVar pinpt, std::string mode, PrimExpr factor) {
  auto n = make_object<SliceNode>();
  n->old = old;
  n->left = left;
  n->right = right;
  n->slicept = slicept;
  n->pinpt = pinpt;
  n->mode = mode;
  n->factor = factor;
  data_ = std::move(n);
}


TVM_REGISTER_NODE_TYPE(HybridStageNode);
TVM_REGISTER_NODE_TYPE(HybridScheduleNode);
TVM_REGISTER_NODE_TYPE(SliceNode);

TVM_REGISTER_GLOBAL("ditto.Display").set_body_method(&HybridStage::display);

TVM_REGISTER_GLOBAL("ditto.CreateHybridSchedule").set_body_typed(create_hybrid_schedule);

TVM_REGISTER_GLOBAL("ditto.HybridStageSetScope").set_body_method(&HybridStage::set_scope);

TVM_REGISTER_GLOBAL("ditto.HybridStageBind").set_body_method(&HybridStage::bind);

TVM_REGISTER_GLOBAL("ditto.Test").set_body_method(&HybridStage::Test);

TVM_REGISTER_GLOBAL("ditto.HybridStageSplitByFactor")
    .set_body_typed([](HybridStage stage, IterVar parent, PrimExpr factor) {
      IterVar outer, inner;
      stage.split(parent, factor, &outer, &inner);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("ditto.HybridStageSliceByMid")
    .set_body_typed([](HybridStage stage, IterVar slicept, IterVar pinpt, PrimExpr factor, std::string mode) {
      Array<IterVar> left, right;
      stage.slice(slicept, pinpt, factor, mode, &left, &right);
      return Array<Array<IterVar> >({left, right});
    });

TVM_REGISTER_GLOBAL("ditto.HybridStageSliceAtRoot")
    .set_body_typed([](HybridStage stage, IterVar slicept, PrimExpr factor, std::string mode) {
      Array<IterVar> left, right;
      stage.slice(slicept, factor, mode, &left, &right);
      return Array<Array<IterVar> >({left, right});
    });


TVM_REGISTER_GLOBAL("ditto.HybridStageSplitByNParts")
    .set_body_typed([](HybridStage stage, IterVar parent, PrimExpr nparts) {
      IterVar outer, inner;
      stage.split_by_nparts(parent, nparts, &outer, &inner);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("ditto.HybridStageFuse").set_body_typed([](HybridStage stage, Array<IterVar> axes) {
  IterVar fused;
  stage.fuse(axes, &fused);
  return fused;
});

TVM_REGISTER_GLOBAL("ditto.HybridStageComputeAt").set_body_method(&HybridStage::compute_at);

TVM_REGISTER_GLOBAL("ditto.HybridStageComputeInline").set_body_method(&HybridStage::compute_inline);

TVM_REGISTER_GLOBAL("ditto.HybridStageComputeRoot").set_body_method(&HybridStage::compute_root);

TVM_REGISTER_GLOBAL("ditto.HybridStageReorder").set_body_method(&HybridStage::reorder);

TVM_REGISTER_GLOBAL("ditto.HybridStageTile")
    .set_body_typed([](HybridStage stage, IterVar x_parent, IterVar y_parent, PrimExpr x_factor,
                       PrimExpr y_factor) {
      IterVar x_outer, y_outer, x_inner, y_inner;
      stage.tile(x_parent, y_parent, x_factor, y_factor, &x_outer, &y_outer, &x_inner, &y_inner);
      return Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
    });

TVM_REGISTER_GLOBAL("ditto.HybridStageEnvThreads").set_body_method(&HybridStage::env_threads);

TVM_REGISTER_GLOBAL("ditto.HybridStageSetStorePredicate").set_body_method(&HybridStage::set_store_predicate);

TVM_REGISTER_GLOBAL("ditto.HybridStageUnroll").set_body_method(&HybridStage::unroll);

TVM_REGISTER_GLOBAL("ditto.HybridStageVectorize").set_body_method(&HybridStage::vectorize);

TVM_REGISTER_GLOBAL("ditto.HybridStageTensorize").set_body_method(&HybridStage::tensorize);

TVM_REGISTER_GLOBAL("ditto.HybridStageParallel").set_body_method(&HybridStage::parallel);

TVM_REGISTER_GLOBAL("ditto.HybridStagePragma").set_body_method(&HybridStage::pragma);

TVM_REGISTER_GLOBAL("ditto.HybridStagePrefetch").set_body_method(&HybridStage::prefetch);

TVM_REGISTER_GLOBAL("ditto.HybridStageStorageAlign").set_body_method(&HybridStage::storage_align);

TVM_REGISTER_GLOBAL("ditto.HybridStageDoubleBuffer").set_body_method(&HybridStage::double_buffer);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleNormalize").set_body_method(&HybridSchedule::normalize);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleCreateGroup").set_body_method(&HybridSchedule::create_group);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleCacheRead").set_body_method(&HybridSchedule::cache_read);

TVM_REGISTER_GLOBAL("ditto.HybridScheduleCacheWrite").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args[1].IsObjectRef<Tensor>()) {
    *ret = args[0].operator HybridSchedule().cache_write(args[1].operator Tensor(), args[2]);
  } else {
    *ret = args[0].operator HybridSchedule().cache_write(args[1].operator Array<Tensor>(), args[2]);
  }
});

TVM_REGISTER_GLOBAL("ditto.HybridScheduleRFactor").set_body_method(&HybridSchedule::rfactor);

// TVM_REGISTER_GLOBAL("ditto.HybridScheduleSlice")
//     .set_body_typed([](HybridSchedule rsch, Tensor tensor, IterVar axis, PrimExpr slice_point) {
//       Tensor tensor_l, tensor_r;
//       rsch.slice(tensor, axis, slice_point, &tensor_l, &tensor_r);
//       return Array<Tensor>({tensor_l, tensor_r});
//     });

}  // namespace hybrid
}  // namespace ditto
