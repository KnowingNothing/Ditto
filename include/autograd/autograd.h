#pragma once

#include <auto_compute/graph.h>

namespace ditto {
using namespace auto_compute;
namespace autograd {

Layer grad_layer(const Layer &layer);

Graph grad_graph(const Graph &graph, bool reserve_forward = false);

} // namespace autograd

} // namespace ditto