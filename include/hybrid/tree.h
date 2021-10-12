#pragma once 

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>

#include <algorithm>
#include <iostream>
#include <queue>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <functional>

using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{

template<class T> 
class TreeUnitNode;
template <typename T> // T is a objRef

class TreeNode{
public:
    const bool is_subTree = false;
    TreeUnitNode<T>* base;
    /*!
    *   \brief get the node with value of v. 
    */
    TreeUnitNode<T>* getUnit(const T & v);
    /*!
    *   \brief get inplace subtree with root_ as root; 
    */
    TreeNode<T> getSubTree(TreeUnitNode<T>*root_);
    /*!
    *   \brief check whether child is in the subtree of parent.  
    */
    bool is_parent(TreeUnitNode<T>*parent, TreeUnitNode<T>*child);
    bool is_parent(const T& parent, const T&child);
    /*!
     * \brief deepCopy the tree. Use f to deep copy value.
     */
    TreeNode<T> deepCopy(std::function<const T&(const T&)>const &f);
    /*!
    * \brief apply f to each node in the tree in mode traverse.
    */
    void apply(std::function<void (TreeUnitNode<T>*)> const & f, std::string mode = "RootFirst");
    TreeNode(bool is_subTree_ = false):is_subTree(is_subTree_){base = new TreeUnitNode<T>();}; // changed
    TreeNode(TreeUnitNode<T>*root_, bool is_subTree_ = false):is_subTree(is_subTree_){
        base = new TreeUnitNode<T>();
        base->pChild = root_;
    }
    ~TreeNode();
    TreeUnitNode<T>* getRoot();
    TreeUnitNode<T>* getBase(){return base;};
    /*!
    * \brief insert child at pos. If pos = NULL, insert at root.
    */
    TreeUnitNode<T>* insertChild(const T& pos, const T & childvalue) {
        return insertChild(getUnit(pos), childvalue);
    };
    TreeUnitNode<T>* insertChild(TreeUnitNode<T>*ptr, const T&childValue) {
        TreeUnitNode<T>* newUnit = new TreeUnitNode<T> (childValue);
        return insertChild(ptr, newUnit);
    };
    TreeUnitNode<T>* insertChild(TreeUnitNode<T>*ptr, TreeUnitNode<T>* child) {
        ptr->insertChild(child);
        return child;
    };
    /*!
    *   \brief insert value v at ptr. v is child of ptr. v has children of ptr.
    */
    TreeUnitNode<T>* insert(TreeUnitNode<T>*ptr, const T&v);
    TreeUnitNode<T>* insert(const T& pos, const T&v);
    /*!
    *   \brief erase ptr. ptr's father succeed ptr's child.
    */
    TreeUnitNode<T>* erase(TreeUnitNode<T> *ptr);
    TreeUnitNode<T>* erase(const T&pos, const T&v);
    bool isEmpty();
    /*!
    *   \brief get parent of current node.
    */
    TreeUnitNode<T>* Parent(TreeUnitNode<T> *current);
    void DeleteSubTree(TreeUnitNode<T>* subroot);
    /*!
    *   \brief rootfirst, rootlast, width traverse.
    */
    void RootFirstTraverse(TreeUnitNode<T>* root, std::function<void (TreeUnitNode<T>*)> const & f);
    void RootLastTraverse(TreeUnitNode<T>*root, std::function<void (TreeUnitNode<T>*)> const & f);
    void WidthTraverse(std::function<void (TreeUnitNode<T>*)> const & f);
    void display(){
        std::cout <<"--------------display begin----------------\n";
        std::queue<TreeUnitNode<T> *> q;
        q.push(base->pChild);
        while(!q.empty()){
            TreeUnitNode<T>* father = q.front();
            q.pop();
            std::cout << father->Value() << " has children: ";
            auto son = father -> pChild;
            while(son){
                
                std::cout << son->Value() << " ";
                q.push(son);
                son = son -> pSibling;
            }
            std::cout << ";\n";
        }
        std::cout <<"---------------display end---------------\n";
        return;
    }
};
template <typename T>
class TreeUnitNode{
public:
    // the dataRef is a ref to the reference of the data 
    T dataRef;
    TreeUnitNode<T>*pChild;
    TreeUnitNode<T>*pSibling;
    TreeUnitNode(){};
    TreeUnitNode(const T&value);
    virtual ~TreeUnitNode(){};
    T Value();
    TreeUnitNode<T>* deepCopy(std::function<const T &(const T&)> const & f);
    void setParent(TreeUnitNode<T>*p);
    void setValue(const T & value){dataRef = value;};
    void insertChild(TreeUnitNode<T>* child);
    void deleteChild(TreeUnitNode<T>* child);
    void display(){
        std::cout <<"------------------------------\n";
        std::queue<TreeUnitNode<T> *> q;
        q.push(this);
        while(!q.empty()){
            TreeUnitNode<T>* father = q.front();
            q.pop();
            std::cout << father->Value() << " has children: ";
            auto son = father -> pChild;
            while(son){
                std::cout << son->Value() << " ";
                q.push(son);
                son = son -> pSibling;
            }
            std::cout << ";\n";
        }
        std::cout <<"------------------------------\n";
        return;
    }
};
} //end hybrid 
} //end ditto
