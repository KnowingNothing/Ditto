#pragma once

#include <auto_compute/graph.h>

namespace ditto {
using namespace auto_compute;
namespace autograd {

Layer grad_layer(const Layer& layer);

}  // namespace autograd

}  // namespace ditto