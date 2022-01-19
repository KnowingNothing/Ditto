#include <tvm/arith/analyzer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <auto_tensorize/dse/evaluator.h>
using namespace tvm;
namespace ditto{
    namespace auto_tensorize{
        FusionResult::FusionResult(Map<tir::Var, IntImm> bounds,\
                      int op1MemVisit,\
                      int op1WorkLoad,\
                      int op1Buffer,\
                      int op2MemVisit,\
                      int op2WorkLoad,\
                      int op2Buffer,\
                      int parallelism){
            auto n = make_object<FusionResultNode>();
            n->bounds = bounds;
            n->op1.memVisit = op1MemVisit;
            n->op1.workLoad = op1WorkLoad;
            n->op1.bufferSize = op1Buffer;
            n->op2.memVisit = op2MemVisit;
            n->op2.workLoad = op2WorkLoad;
            n->op2.bufferSize = op2Buffer;
            n->parallelism = parallelism;
            data_ = n;
        }
        loss_t FusionResultNode::loss() const{
            // the arithmetic intensity;
            return - (op1.workLoad + op2.workLoad) / loss_t(op1.memVisit + op2.memVisit);
        }
        StaticAnalysis::StaticAnalysis(IterGraph ig){
            auto n = make_object<StaticAnalysisNode>();
            n->tag = "static analysis";
            n->iterGraph = ig;
            data_ = n;
        }

        Result StaticAnalysisNode::eval(Item it) const{
            auto fusionItem = Downcast<FusionItem, Item>(it);
            iterGraph->setSchedule(fusionItem);
            FusionResult result =  iterGraph->getAnalyticalResult();
            return result;
        }
        loss_t StaticAnalysisNode::loss(Item it) const{
            auto fusionItem = Downcast<FusionItem, Item>(it);
            iterGraph->setSchedule(fusionItem);
            FusionResult result =  iterGraph->getAnalyticalResult();
            return result->loss();
        }
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<EvaluatorNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const EvaluatorNode *>(node.get());
            p->stream << op->tag;
        });
        TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
        .set_dispatch<FusionResultNode>([](const ObjectRef& node, ReprPrinter * p){
            auto* op = static_cast<const FusionResultNode *>(node.get());
            p->stream << "FusionResult(\n";
            p->stream << "op1:\n";
            p->stream << "\tMemVisit: " << op->op1.memVisit << "\n";
            p->stream << "\tworkLoad: " << op->op1.workLoad << "\n";
            p->stream << "\tbufferSize: " << op->op1.bufferSize << "\n";
            p->stream << "op2:\n";
            p->stream << "\tMemVisit: " << op->op2.memVisit << "\n";
            p->stream << "\tworkLoad: " << op->op2.workLoad << "\n";
            p->stream << "\tbufferSize: " << op->op2.bufferSize << "\n";
            p->stream << "parallelism: " << op->parallelism << " )";
        });
    }// auto_tensorize
} //ditto
