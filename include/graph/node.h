#pragma once

#include <memory>
#include <tvm/tir/expr.h>

namespace nast {


namespace graph {

/*
 * base class for all the ops
*/
class BaseOpNode {

};


/*
 * ops that has no state
*/
class StatelessOpNode : public BaseOpNode {

};


/*
 * ops that has states such as weights
*/
class StateOpNode : public BaseOpNode {

};


///////////////////////////////////////////////////
//  define the stateless ops:
//  1. constant op for constant tensors
//  2. function op for side-effect-free computation
//////////////////////////////////////////////////
class ConstantOpNode : public StatelessOpNode {

};


using ConstantOp = std::shared_ptr<ConstantOpNode>;


class FunctionOpNode : public StatelessOpNode {

};


using FunctionOp = std::shared_ptr<FunctionOpNode>;


///////////////////////////////////////////////////
//  define the state ops:
//  1. placeholder op for common tensors
//  2. compute op for computation with weights involved
//////////////////////////////////////////////////
class PlaceholderOpNode : public StateOpNode {

};


using PlaceholderOp = std::shared_ptr<PlaceholderOpNode>;


class ComputeOpNode : public StateOpNode {

};


using ComputeOp = std::shared_ptr<ComputeOpNode>;

}  // namespace graph

}  // namespace nast