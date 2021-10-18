#pragma once 

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
using namespace tvm;

namespace ditto{
namespace hybrid{
    class testObj: public Object{
        public:
        int data;
        static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
        static constexpr const char* _type_key = "test.testObj";
        TVM_DECLARE_BASE_OBJECT_INFO(testObj, Object);
    };
    class test: public ObjectRef{
        public:
        test(int n = 0){
            auto node = make_object<testObj>();
            node->data = n;
            data_ = node;
        }
        explicit test(ObjectPtr<Object> n) : ObjectRef(n){
        }
        std::string str() const{
            return std::to_string(operator->()->data);
        }
        inline testObj* operator->() const{
            return static_cast<testObj*> (data_.get()); 
        }
        using ContainerType = testObj;
    };
}
}