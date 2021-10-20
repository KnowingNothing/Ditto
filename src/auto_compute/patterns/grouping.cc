#include <auto_compute/patterns/grouping.h>
#include <auto_compute/patterns/utils.h>

namespace ditto {

namespace auto_compute {


Array<Pattern> FindGroupingPattern(const te::Operation &op) {
  std::vector<Pattern> ret;
  const te::ComputeOpNode *cop = op.as<te::ComputeOpNode>();
  if (cop) {
    Array<te::Tensor> inputs = cop->InputTensors();
    Array<te::IterVar> axis = cop->axis;
    Array<te::IterVar> reduce_axis = cop->reduce_axis;
    CHECK(cop->body.size() == 1U)
        << "Don't know how to handle multiple body.\n";
    int count_riv = 0;
    int count_siv = 0;
    int count_weight = 0;
    int count_data = 0;
    if (reduce_axis.size()) {
      count_riv = 0;
      for (auto riv : reduce_axis) {
        count_siv = 0;
        for (auto siv : axis) {
          count_weight = 0;
          for (auto to_be_weight : inputs) {
            count_data = 0;
            for (auto to_be_data : inputs) {
              if (to_be_weight == to_be_data) {
                count_data += 1;
                continue;
              }
              PureIndexIn rv_exist_w(to_be_weight->op, riv->var);
              PureIndexIn sv_exist_w(to_be_weight->op, siv->var);
              PureIndexIn rv_exist_d(to_be_data->op, riv->var);
              PureIndexIn sv_exist_d(to_be_data->op, siv->var);
              bool cond = true;
              cond &= rv_exist_w.check_single(cop->body[0]);
              int rv_access_dim = rv_exist_w.get_dim();
              cond &= sv_exist_w.check_single(cop->body[0]);
              cond &= rv_exist_d.check_single(cop->body[0]);
              cond &= !sv_exist_d.check_exist(cop->body[0]);

              if (cond) {
                Array<IntImm> tensors = {make_int(count_data), make_int(count_weight)};
                Array<IntImm> iter_vars = {make_int(count_siv), make_int(count_riv)};
                Array<IntImm> access_dim = {make_int(rv_access_dim)};
                Array<Array<IntImm>> iter_vars_array = {iter_vars, access_dim};
                Pattern p(tensors, iter_vars_array);
                ret.push_back(p);
              }

              count_data += 1;
            }
            count_weight += 1;
          }
          count_siv += 1;
        }
        count_riv += 1;
      }
    }
  }

  return Array<Pattern>(ret);
}

TVM_REGISTER_GLOBAL("ditto.auto_compute.FindGroupingPattern")
    .set_body_typed(FindGroupingPattern);

} // namespace auto_compute

} // namespace ditto