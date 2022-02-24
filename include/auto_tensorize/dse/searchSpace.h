#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <stdio.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <auto_tensorize/iter_graph.h>
#include <hardware/base/hw_param.h>

using namespace tvm;
namespace ditto {
namespace auto_tensorize {
class ItemNode : public Object {
public:
  void VisitAttrs(AttrVisitor *v) {}
  static constexpr const char *_type_key = "ditto.auto_tensorize.Item";
  TVM_DECLARE_BASE_OBJECT_INFO(ItemNode, Object);
};

class Item : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Item, ObjectRef, ItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ItemNode);
};

class TilingItemNode : public ItemNode {
public:
  Array<IntImm> factors;
  static constexpr const char *_type_key = "ditto.auto_tensorize.TilingItem";
  void VisitAttrs(AttrVisitor *v) { v->Visit("factors", &factors); }
  TVM_DECLARE_FINAL_OBJECT_INFO(TilingItemNode, ItemNode);
};

class TilingItem : public Item {
public:
  TVM_DLL TilingItem(Array<IntImm> factors);
  TVM_DEFINE_OBJECT_REF_METHODS(TilingItem, Item, TilingItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TilingItemNode);
};

class PermuteItemNode : public ItemNode {
public:
  Array<IntImm> permute;
  static constexpr const char *_type_key = "ditto.auto_tensorize.PermuteItem";
  void VisitAttrs(AttrVisitor *v) { v->Visit("permute", &permute); }
  TVM_DECLARE_FINAL_OBJECT_INFO(PermuteItemNode, ItemNode);
};

class PermuteItem : public Item {
public:
  TVM_DLL PermuteItem(Array<IntImm> Permute);
  TVM_DEFINE_OBJECT_REF_METHODS(PermuteItem, Item, PermuteItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PermuteItemNode);
};

class AttachItemNode : public ItemNode {
public:
  size_t attachPos;
  static constexpr const char *_type_key = "ditto.auto_tensorize.AttachItem";
  void VisitAttrs(AttrVisitor *v) { v->Visit("attachPos", &attachPos); }
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachItemNode, ItemNode);
};

class AttachItem : public Item {
public:
  TVM_DLL AttachItem(size_t AttachPos);
  TVM_DEFINE_OBJECT_REF_METHODS(AttachItem, Item, AttachItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttachItemNode);
};

class FusionItemNode : public ItemNode {
public:
  TilingItem firstOpTiling, secondOpTiling;
  PermuteItem firstOpPermute, secondOpPermute;
  AttachItem attachPos;
  static constexpr const char *_type_key = "ditto.auto_tensorize.FusionItem";
  void VisitAttrs(AttrVisitor *v) {
    v->Visit("fristOpTiling", &firstOpTiling);
    v->Visit("secondOpTiling", &secondOpTiling);
    v->Visit("firstOpPermute", &firstOpPermute);
    v->Visit("secondOpPermute", &secondOpPermute);
    v->Visit("attachPos", &attachPos);
  }
  TVM_DECLARE_FINAL_OBJECT_INFO(FusionItemNode, ItemNode);
};

class FusionItem : public Item {
public:
  TVM_DLL FusionItem(TilingItem firstOpTiling, TilingItem secondOpTiling,
                     PermuteItem firstOpPermute, PermuteItem secondOpPermute,
                     AttachItem attachPos);
  TVM_DEFINE_OBJECT_REF_METHODS(FusionItem, Item, FusionItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FusionItemNode);
};
// template<class T>
// struct Iterator
// {
//     using iterator_category = std::input_iterator_tag;
//     using difference_type = std::ptrdiff_t;
//     SearchSpace searchSpace;
// public:
//     Iterator(pointer ptr): searchSpace(searchSpace){}
//     Item operator*() const {return searchSpace->getItem();}
//     Iterator& operator++() {searchSpace ++; return *this;}
//     bool operator == (const Iterator & a, const Iterator & b) {return
//     a.searchSpace == b.searchSpace; }; bool operator != (const Iterator & a,
//     const Iterator & b) {return a.searchSpace != b.searchSpace; };
// };
class SearchSpaceNode : public Object {
public:
  std::string name;
  size_t cardinal;
  size_t index;
  // whether the paramter is given by the user
  bool mandatory;

  void VisitAttrs(AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("cardinal", &cardinal);
  }
  static constexpr const char *_type_key = "ditto.auto_tensorize.SearchSpace";
  void reset() { index = 0; }
  virtual Item idxToItem(size_t idx) const { return Item(); }
  TVM_DECLARE_BASE_OBJECT_INFO(SearchSpaceNode, Object);
};

class SearchSpace : public ObjectRef {
public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchSpace, ObjectRef,
                                        SearchSpaceNode);
};

class TilingSpaceNode : public SearchSpaceNode {
public:
  Array<IntImm> mandatories;
  Array<IntImm> extents;
  void setMandatory(Array<IntImm>);
  void VisitAttrs(AttrVisitor *v) { v->Visit("extents", &extents); }
  Item idxToItem(size_t idx) const override;
  TVM_DECLARE_FINAL_OBJECT_INFO(TilingSpaceNode, SearchSpaceNode);
};

class TilingSpace : public SearchSpace {
public:
  TVM_DLL TilingSpace(Array<IntImm> extents);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingSpace, SearchSpace,
                                        TilingSpaceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TilingSpaceNode);
};

class AttachSpaceNode : public SearchSpaceNode {
public:
  Array<IntImm> mandatories;
  size_t numOfLoops;
  void setMandatory(Array<IntImm>);
  Item idxToItem(size_t id) const override;
  TVM_DECLARE_FINAL_OBJECT_INFO(AttachSpaceNode, SearchSpaceNode);
};

class AttachSpace : public SearchSpace {
public:
  TVM_DLL AttachSpace(size_t numLoops);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AttachSpace, SearchSpace,
                                        AttachSpaceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttachSpaceNode);
};

class PermuteSpaceNode : public SearchSpaceNode {
public:
  Array<Array<IntImm>> mandatories;
  size_t numOfLoops;
  size_t perm[10];
  void setMandatory(Array<Array<IntImm>>);
  Item idxToItem(size_t index) const override;
  TVM_DECLARE_FINAL_OBJECT_INFO(PermuteSpaceNode, SearchSpaceNode);
};

class PermuteSpace : public SearchSpace {
public:
  TVM_DLL PermuteSpace(size_t numOfLoops);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PermuteSpace, SearchSpace,
                                        PermuteSpaceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PermuteSpaceNode);
};

class FusionSpaceNode : public SearchSpaceNode {
public:
  PermuteSpace firstOpPermute, secondOpPermute;
  AttachSpace attach;
  TilingSpace firstOpTiling, secondOpTiling;
  Item idxToItem(size_t idx) const override;
  void VisitAttrs(AttrVisitor *v) {
    v->Visit("firstOpPermute", &firstOpPermute);
    v->Visit("secondOpPermute", &secondOpPermute);
    v->Visit("attachOp", &attach);
    v->Visit("fristOpTiling", &firstOpTiling);
    v->Visit("secondOpTiling", &secondOpTiling);
  }
  void updateCardinal() {
    cardinal = firstOpTiling->cardinal * secondOpTiling->cardinal *
               firstOpPermute->cardinal * secondOpPermute->cardinal *
               attach->cardinal;
    // std::cout << "cardinals: " << firstOpTiling->cardinal << " "
    //           << secondOpTiling->cardinal << " " << firstOpPermute->cardinal
    //           << " " << secondOpPermute->cardinal << " " << attach->cardinal
    //           << std::endl;
  }
  void setFirstOpTilingMandatory(Array<IntImm> mand) {
    firstOpTiling->setMandatory(mand);
    updateCardinal();
    reset();
  }
  void setSecondOpTilingMandatory(Array<IntImm> mand) {
    secondOpTiling->setMandatory(mand);
    updateCardinal();
    reset();
  }
  void setFirstOpPermuteMandatory(Array<Array<IntImm>> mand) {
    firstOpPermute->setMandatory(mand);
    updateCardinal();
    reset();
  }
  void setSecondOpPermuteMandatory(Array<Array<IntImm>> mand) {
    secondOpPermute->setMandatory(mand);
    updateCardinal();
    reset();
  }
  void setAttacchMandatory(Array<IntImm> mand) {
    attach->setMandatory(mand);
    updateCardinal();
    reset();
  }
  static constexpr const char *_type_key = "ditto.auto_tensorize.FusionSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(FusionSpaceNode, SearchSpaceNode);
};

class FusionSpace : public SearchSpace {
public:
  TVM_DLL FusionSpace(PermuteSpace firstOpPermute, PermuteSpace secondOpPermute,
                      TilingSpace firstOpTiling, TilingSpace secondOpTiling,
                      AttachSpace attach);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FusionSpace, SearchSpace,
                                        FusionSpaceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FusionSpaceNode);
};
inline FusionItem buildFusionItem(Array<IntImm> firstOpTiling,
                                  Array<IntImm> secondOpTiling,
                                  Array<IntImm> firstOpPermute,
                                  Array<IntImm> secondOpPermute,
                                  size_t attachPos);
} // namespace auto_tensorize

} // namespace ditto