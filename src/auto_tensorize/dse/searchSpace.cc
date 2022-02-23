#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <algorithm>
#include <vector>
#include <auto_tensorize/dse/searchSpace.h>


using namespace tvm;
namespace ditto{
    namespace auto_tensorize{
        
        size_t product(Array<IntImm> nums){
            size_t ret = 1;
            for (auto i: nums)
                ret *= i->value;
            return ret;
        }

        TilingSpace::TilingSpace(Array<IntImm> extents){
            auto n = make_object<TilingSpaceNode>();
            n->extents = extents;
            n->cardinal = product(extents);
            n->index = 0;
            n->name = "Tiling";
            n->mandatory = false;
            data_ = n;
        }
        Item TilingSpaceNode::idxToItem(size_t idx) const{
            Array<IntImm> ret;
            for (size_t i = 0; i < extents.size(); i++){
                if (mandatory && mandatories[i]->value > 0){
                    ret.push_back(IntImm(DataType::Int(32), mandatories[i]->value));
                }
                else{
                    ret.push_back(IntImm(DataType::Int(32), idx % extents[i]->value + 1));
                    idx /= extents[i]->value;
                }
            }
            CHECK(idx == 0) << "Tiling cardinal error";
            return TilingItem(ret);
        }
        void TilingSpaceNode::setMandatory(Array<IntImm> mandatories_){
            mandatory = true;
            mandatories = mandatories_;
            cardinal = 1;
            for (size_t i = 0; i < mandatories.size(); i++)
                if (mandatories[i]->value < 0){
                    cardinal *= extents[i]->value;
                }
            reset();
        }
        PermuteSpace::PermuteSpace(size_t numOfLoops){
            auto n = make_object<PermuteSpaceNode>();
            n->numOfLoops = numOfLoops;
            size_t card = 1;
            for(size_t i = 1; i <= numOfLoops; i++) card *= i;
            n->cardinal = card; 
            n->index = 0;
            n->name = "Permute";
            data_ = n;
        }
        Item PermuteSpaceNode::idxToItem(size_t idx) const{
            if (mandatory)
                return PermuteItem(mandatories[idx]);
            std::vector<size_t> partial_rank;
            size_t product = 1;
            for(size_t i = 0; i < numOfLoops; i++){
                partial_rank.push_back(idx % product);
                product *= (i+1);
            }
            std::vector<size_t> perm;
            Array<IntImm> ret;
            for(size_t i = 0; i < numOfLoops; i++)
                perm.push_back(i);
            for(int i = numOfLoops - 1; i >= 0; i--){
                ret.push_back(IntImm(DataType::Int(32), perm[partial_rank[i]]));
                perm.erase(perm.begin() + partial_rank[i]);
            }
            return PermuteItem(ret);
        }
        void PermuteSpaceNode::setMandatory(Array<Array<IntImm>> mands){
            mandatories = mands;
            mandatory = true;
            cardinal = mandatories.size();
            reset();
        }
        AttachSpace::AttachSpace(size_t numOfLoops){
            auto n = make_object<AttachSpaceNode>();
            n->numOfLoops = numOfLoops;
            n->cardinal = numOfLoops;
            n->name = "Attach";
            data_ = n;
        }
        Item AttachSpaceNode::idxToItem(size_t idx) const{\
            if (mandatory)
                return AttachItem((mandatories[idx])->value);
            CHECK(idx <= numOfLoops) << "attach range exceeded";
            return AttachItem(idx);
        }
        void AttachSpaceNode::setMandatory(Array<IntImm> mands){
            mandatories = mands;
            mandatory = true;
            cardinal = mandatories.size();
            reset();
        }
        FusionSpace::FusionSpace(PermuteSpace firstOpPermute, PermuteSpace secondOpPermute,\
                                TilingSpace firstOpTiling, TilingSpace secondOpTiling,\
                                AttachSpace attach){
            auto n = make_object<FusionSpaceNode>();
            n->firstOpPermute = firstOpPermute;
            n->secondOpPermute = secondOpPermute;
            n->attach = attach;
            n->firstOpTiling = firstOpTiling;
            n->secondOpTiling = secondOpTiling;
            n->name = "Fusion";
            std::cout << firstOpPermute->cardinal << " " << secondOpPermute ->cardinal << " " << firstOpTiling->cardinal << " " << secondOpTiling ->cardinal << " " << attach->cardinal << std::endl;
            n->cardinal = firstOpPermute->cardinal * secondOpPermute ->cardinal * \
                            firstOpTiling->cardinal * secondOpTiling ->cardinal * attach->cardinal;
            n->index = 0;
            data_ = n;
        }
        Item FusionSpaceNode::idxToItem(size_t idx) const{
            TilingItem firstOpTilingItem, secondOpTilingItem;
            PermuteItem firstOpPermuteItem, secondOpPermuteItem;
            AttachItem attachItem;

            firstOpPermuteItem = Downcast<PermuteItem, Item>(firstOpPermute->idxToItem(idx % firstOpPermute->cardinal));
            idx /= firstOpPermute->cardinal;

            secondOpPermuteItem = Downcast<PermuteItem, Item>(secondOpPermute->idxToItem(idx % secondOpPermute->cardinal));
            idx /= secondOpPermute->cardinal;
            
            firstOpTilingItem = Downcast<TilingItem, Item>(firstOpTiling->idxToItem(idx % firstOpTiling->cardinal));
            idx /= firstOpTiling->cardinal;
            
            secondOpTilingItem = Downcast<TilingItem, Item>(secondOpTiling->idxToItem(idx % secondOpTiling->cardinal));
            idx /= secondOpTiling->cardinal;
            
            attachItem = Downcast<AttachItem, Item>(attach->idxToItem(idx));

            return FusionItem(firstOpTilingItem, secondOpTilingItem,\
            firstOpPermuteItem, secondOpPermuteItem, attachItem);
        }
        TilingItem::TilingItem(Array<IntImm> factors){
            auto n = make_object<TilingItemNode>();
            n->factors = factors;
            data_ = n;
        }
        PermuteItem::PermuteItem(Array<IntImm> permute){
            auto n = make_object<PermuteItemNode>();
            n->permute = permute;
            data_ = n;
        }
        AttachItem::AttachItem(size_t attachPos){
            auto n = make_object<AttachItemNode>();
            n->attachPos = attachPos;
            data_ = n;
        }
        FusionItem::FusionItem(TilingItem firstOpTiling, TilingItem secondOpTiling,
        PermuteItem firstOpPermute, PermuteItem secondOpPermute,\
        AttachItem attachPos){
            auto n = make_object<FusionItemNode>();
            n->firstOpPermute = firstOpPermute;
            n->firstOpTiling = firstOpTiling;
            n->secondOpPermute = secondOpPermute;
            n->secondOpTiling = secondOpTiling;
            n->attachPos = attachPos;
            data_ = n;
        }
        FusionItem buildFusionItem(Array<IntImm> firstOpTiling, Array<IntImm> secondOpTiling,\
  Array<IntImm> firstOpPermute, Array<IntImm> secondOpPermute, size_t attachPos){
            return FusionItem(TilingItem(firstOpTiling), TilingItem(secondOpTiling),\
            PermuteItem(firstOpPermute), PermuteItem(secondOpPermute), 
            AttachItem(attachPos));
        }
        

        TVM_REGISTER_NODE_TYPE(ItemNode);
        TVM_REGISTER_NODE_TYPE(FusionItemNode);
        TVM_REGISTER_NODE_TYPE(SearchSpaceNode);
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<SearchSpaceNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const SearchSpaceNode *>(node.get());
            p->stream << "searchSpace(";
            p->stream << op->name << ", ";
            p->stream << "cardinal: "<< op->cardinal << ")";
        });
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<TilingItemNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const TilingItemNode *>(node.get());
            p->Print(op->factors);
        });
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<PermuteItemNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const PermuteItemNode *>(node.get());
            p->Print(op->permute);
        });
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<AttachItemNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const AttachItemNode *>(node.get());
            p->stream << op->attachPos;
        });
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<FusionItemNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const FusionItemNode *>(node.get());
            p->stream << "-----FusionItem---------\n";
            p->stream << "firstOpTiling:\t";
            p->Print(op->firstOpTiling);
            p->stream << "\n";
            p->stream << "secondOpTiling:\t";
            p->Print(op->secondOpTiling);
            p->stream << "\n";
            p->stream << "firstOpPermute:\t";
            p->Print(op->firstOpPermute);
            p->stream << "\n";
            p->stream << "secondOpPermute:\t";
            p->Print(op->firstOpPermute);
            p->stream << "\n";
            p->stream << "attachPos:\t";
            p->Print(op->attachPos);
            p->stream << "\n";
            p->stream << "-------------------------";
        });
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<FusionSpaceNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const FusionSpaceNode *>(node.get());
            p->stream << "FusionSpace(";
            p->stream << "cardinal: "<< op->cardinal << ")";
        });
        TVM_REGISTER_NODE_TYPE(FusionSpaceNode);
        TVM_REGISTER_NODE_TYPE(TilingSpaceNode);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setMandatory").\
        set_body_method<TilingSpace>(&TilingSpaceNode::setMandatory);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setFirstOpTilingMandatory").\
        set_body_method<FusionSpace>(&FusionSpaceNode::setFirstOpTilingMandatory);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setSecondOpTilingMandatory").\
        set_body_method<FusionSpace>(&FusionSpaceNode::setSecondOpTilingMandatory);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setFirstOpPermuteMandatory").\
        set_body_method<FusionSpace>(&FusionSpaceNode::setFirstOpPermuteMandatory);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setSecondOpPermuteMandatory").\
        set_body_method<FusionSpace>(&FusionSpaceNode::setSecondOpPermuteMandatory);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.setAttachMandatory").\
        set_body_method<FusionSpace>(&FusionSpaceNode::setAttacchMandatory);
        TVM_REGISTER_GLOBAL("ditto.auto_tensorize.buildFusionItem").
        set_body_typed(buildFusionItem);
    }// auto_tensorize
} //ditto`