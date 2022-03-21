#pragma once

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/registry.h>
#include <utils/logging.h>

namespace ditto {
using namespace tvm;
using namespace tvm::runtime;
namespace utils {

template <typename Function, typename T> class CallFunc {
public:
  void call_func_0(Function f) { f(); }

  void call_func_1(Function f, std::vector<T> v) { f(v[0]); }

  void call_func_2(Function f, std::vector<T> v) { f(v[0], v[1]); }

  void call_func_3(Function f, std::vector<T> v) { f(v[0], v[1], v[2]); }

  void call_func_4(Function f, std::vector<T> v) { f(v[0], v[1], v[2], v[3]); }

  void call_func_5(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4]);
  }

  void call_func_6(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5]);
  }

  void call_func_7(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
  }

  void call_func_8(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  }

  void call_func_9(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
  }

  void call_func_10(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
  }

  void call_func_11(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
  }

  void call_func_12(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
  }

  void call_func_13(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12]);
  }

  void call_func_14(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13]);
  }

  void call_func_15(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14]);
  }

  void call_func_16(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14], v[15]);
  }

  void call_func_17(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14], v[15], v[16]);
  }

  void call_func_18(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14], v[15], v[16], v[17]);
  }

  void call_func_19(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14], v[15], v[16], v[17], v[18]);
  }

  void call_func_20(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19]);
  }

  void call_func_21(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
      v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], v[20]);
  }

  void call_func_any(Function f, std::vector<T> v) {
    const auto *call_unpack =
        tvm::runtime::Registry::Get("runtime.call_unpack");
    ASSERT(call_unpack != nullptr) << "Should prepare call_unpack function.";
    (*call_unpack)(f, Array<T>(v));
  }

  void operator()(Function f, std::vector<T> v) {
    int num_args = (int)v.size();
    switch (num_args) {
    case 0:
      call_func_0(f);
      break;
    case 1:
      call_func_1(f, v);
      break;
    case 2:
      call_func_2(f, v);
      break;
    case 3:
      call_func_3(f, v);
      break;
    case 4:
      call_func_4(f, v);
      break;
    case 5:
      call_func_5(f, v);
      break;
    case 6:
      call_func_6(f, v);
      break;
    case 7:
      call_func_7(f, v);
      break;
    case 8:
      call_func_8(f, v);
      break;
    case 9:
      call_func_9(f, v);
      break;
    case 10:
      call_func_10(f, v);
      break;
    case 11:
      call_func_11(f, v);
      break;
    case 12:
      call_func_12(f, v);
      break;
    case 13:
      call_func_13(f, v);
      break;
    case 14:
      call_func_14(f, v);
      break;
    case 15:
      call_func_15(f, v);
      break;
    case 16:
      call_func_16(f, v);
      break;
    case 17:
      call_func_17(f, v);
      break;
    case 18:
      call_func_18(f, v);
      break;
    case 19:
      call_func_19(f, v);
      break;
    case 20:
      call_func_20(f, v);
      break;
    case 21:
      call_func_21(f, v);
      break;
    default:
      std::cout << "need " << num_args << "arguments.\n";
      call_func_any(f, v);
    }
  }
};

} // namespace utils

} // namespace ditto