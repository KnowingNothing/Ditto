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
#include <stack>

using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{
template<typename T>
class TreeBaseNode;
template<typename T>
class TreeUnitNode;
template <typename T>
class TreeNode: public Object, public TreeBaseNode<T>{
    public:
    TreeNode(){
    }
    TreeNode(TreeBaseNode<T> & t): TreeBaseNode<T>(t){
    }
    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
    static constexpr const char * _type_key = "test.TreeNode";
    TVM_DECLARE_BASE_OBJECT_INFO(TreeNode, Object);
};
template <typename T>
class Tree: public ObjectRef{
    public:
    Tree(){
        auto node = make_object<TreeNode<T> >();
        data_ = node;
    }
    explicit Tree(ObjectPtr<Object> n) : ObjectRef(n){
    }
    /*!
    *   \brief the default copy construtor
    *   \param f the function to copy T object, default shallow copy. 
    */
    Tree(const Tree<T> & tree, std::function<T (const T &)>const & f = [](const T & t)->T{
        ObjectPtr<typename T::ContainerType> n = make_object<typename T::ContainerType>(*t.operator->());
        T tmp(n);
        return tmp;
    }){
        TreeBaseNode<T> node = tree.operator->()->deepCopy(f);
        node.is_subTree = 0;
        auto node_ = make_object<TreeNode<T> >(node);
        node.is_subTree = 1;
        data_ = node_;
    }
    inline TreeNode<T>* operator->() const;
    int is_subTree(){
        return operator->()->is_subTree;
    }
    /*!
    * \brief insertChild at root
    * \param v the value to insert at root
    */
    void insertChild(const T& v){
        operator->()->insertChild(v);}
    /*!
    *   \brief insertChild
    *   \param p the parent position
    *   \param v the child value
    */
    void insertChild(const T & p, const T & v){
        operator->()->insertChild(v, p);}
    /*!
    *   \brief insertTree
    *   \param p the parent
    *   \param tree the tree to insert
    */
    void insertTree(const T & p, Tree<T> & tree){
        operator->()->insertTree(p, *tree.operator->());
    }
    /*!
    *   \brief insertTree at root 
    *   \param tree the tree to insert
    */
    void insertTree(Tree<T>&tree){
        operator->()->insertTree(*tree.operator->());
    }
    /*!
    *    \brief these 2 functions are used for debug 
    */
    TreeUnitNode<T>* getUnit(const T & t){return operator->()->getUnit(t);}
    TreeUnitNode<T>* getBase(){return operator->()->getBase();}
    /*!
    *   \brief change the node
    *   \param pos the original value
    *   \param v    the new value 
    */
    void setValue(const T & pos, const T & v){
        auto ptr = operator->()->getUnit(pos);
        ICHECK(ptr) << "reference value not found in tree.";
        ptr->setValue(v);
    }
    /*!
    * \brief insert at root
    * \param v the value to insert
    */
    void insert(const T& v){operator->()->insert(v);}
    /*!
    *   \brief insert at p
    *   \param  p the parant value
    *   \param  v the value to insert
    */
    void insert(const T & p, const T & v){operator->()->insert(p, v);}
    /*!
    *   \brief erase
    *   \param  v value to erase
    */
    void erase(const T& v){operator->()->erase(v);}
    /*!
    * \brief display the tree structure 
    */
    void display(std::string name = ""){operator->()->display(name);}
    /*!
    *   \brief get the SubTree, note that this use the old tree, not a new one. 
    *   \param t, the root node on the tree
    */
    Tree<T> getSubTree(const T& t){
        TreeBaseNode<T> node = operator->()->getSubTree(t);
        auto node_ = make_object<TreeNode<T> >(node);
        node.is_subTree = 1;
        Tree<T>ret = Tree<T>(node_);
        node_->is_subTree = 2;
        return ret;
    }
    /*!
    *   \brief erase the subTree with root t;
    */
    void eraseTree(const T& t){
        operator->()->eraseTree(t);
    }
    /*!
    *   \brief apply a function to all nodes in tree; 
    */
    void apply(std::function<void (T & )> const & f){
        operator->()->apply(f);
    }
    /*!
    *   \brief check where child is in the subtree of parent.
    */
    bool is_parent(const T& parent, const T& child){
        return operator->()->is_parent(parent, child);
    }
    /*!
    * \brief access the internal node container
    * \return the pointer to the internal node container
    */
    using ContainerType = TreeNode<T>;
};
template<typename T>
inline TreeNode<T>* Tree<T>::operator->() const{
    return static_cast<TreeNode<T>*> (data_.get()); 
}


template<class T> 
class TreeUnitNode;
template <typename T> // T is a objRef
class TreeBaseNode{
public:
    int is_subTree;
    TreeUnitNode<T>* base;
    TreeBaseNode(int is_subTree_ = 0):is_subTree(is_subTree_){
        base = new TreeUnitNode<T>();
    }; // changed
    TreeBaseNode(TreeUnitNode<T>*root_, int is_subTree_ = 0):is_subTree(is_subTree_){
        base = new TreeUnitNode<T>();
        base->pChild = root_;
    }
    TreeBaseNode(const TreeBaseNode<T> & t): is_subTree(false){
        base = new TreeUnitNode<T>();
        base->pChild = t.base->pChild;
    }
    ~TreeBaseNode(){
        if(is_subTree == 0)
            apply([](TreeUnitNode<T>* node){
                    delete node;
                }, "RootLast");
        if(is_subTree != 2)
            delete base;
    }
    /*!
    *   \brief get the node with value of v. 
    */
    TreeUnitNode<T>* getUnit(const T & v){
        std::queue <TreeUnitNode<T>* > Q;
        Q.push(base->pChild);
        while(!Q.empty()){
            auto p = Q.front();
            Q.pop();
            if(p == NULL) continue;
            if(p->data_ptr->same_as(v))
                return p;
            Q.push(p->pChild);
            Q.push(p->pSibling);
        }
        return NULL;
    }

    TreeUnitNode<T>* getUnit(TreeUnitNode<T>* v){
        std::queue <TreeUnitNode<T>* > Q;
        Q.push(base->pChild);
        while(!Q.empty()){
            auto p = Q.front();
            Q.pop();
            if(p == NULL) continue;
            if(p->data_ptr->same_as(*v->data_ptr))
                return p;
            Q.push(p->pChild);
            Q.push(p->pSibling);
        }
        return NULL;
    }
    /*!
    *   \brief get inplace subtree with root_ as root; 
    */
    TreeBaseNode<T> getSubTree(const T&root_){
        TreeUnitNode<T> * ptr = getUnit(root_);
        ICHECK(ptr) << "root node not in tree";
        TreeBaseNode<T> subTree(ptr, 1);
        return subTree;
    }
    /*!
    *   \brief check whether child is in the subtree of parent.  
    */
    bool is_parent(TreeUnitNode<T>*parent, TreeUnitNode<T>*child){
        if(!parent||!child){
            return false;
        }
        bool isParent = false;
        auto func = [&isParent, &child](TreeUnitNode<T>* p){
            if(p == child){
                isParent = true; 
            }
        };
        getSubTree(parent).apply(func);
        return isParent;
    }
    bool is_parent(const T& parent, const T&child){
        return is_parent(getUnit(parent), getUnit(child));
    }
    /*!
     * \brief deepCopy the tree. Use f to deep copy value.
     */
    TreeBaseNode<T> deepCopy(std::function<T(const T&)>const &f){
        TreeUnitNode<T> * root_ = NULL;
        if(base->pChild)
            root_ = base->pChild->deepCopy(f);
        return TreeBaseNode<T>(root_);
    }

    void apply(std::function<void (T &)> const & f, std::string Mode = "RootFirst"){
        if (Mode == "RootFirst"){
            RootFirstTraverse(base->pChild, f);
        }
        else if(Mode == "RootLast"){
            RootLastTraverse(base->pChild, f);
        }
        else if(Mode == "Width"){
            WidthTraverse(f);
        }
        else{
            LOG(FATAL) << "no traverse matched";
        }
    }
    /*!
    * \brief apply f to each node in the tree in mode traverse.
    */
    void apply(std::function<void (TreeUnitNode<T>*)> const & f, std::string Mode = "RootFirst"){
        if (Mode == "RootFirst"){
            RootFirstTraverse(base->pChild, f);
        }
        else if(Mode == "RootLast"){
            RootLastTraverse(base->pChild, f);
        }
        else if(Mode == "Width"){
            WidthTraverse(f);
        }
        else{
            LOG(FATAL) << "no traverse matched";
        }
    }
    TreeUnitNode<T>* getRoot(){return base->pChild;}
    TreeUnitNode<T>* getBase(){return base;}
    /*!
    * \brief insert child at pos. If pos = NULL, insert at root.
    */
    TreeUnitNode<T>* insertChild(const T & childValue){
        return insertChild(childValue, base);
    }
    TreeUnitNode<T>* insertChild(const T & childvalue, const T& pos) {
        auto ptr = getUnit(pos);
        ICHECK(ptr != NULL) << "insert pos not in the tree. pos: " << pos.str() << "child: " << childvalue.str();
        ICHECK(getUnit(childvalue) == NULL) << "cannot insert inserted node";
        return insertChild(childvalue, getUnit(pos));
    };
    TreeUnitNode<T>* insertChild(const T&childValue, TreeUnitNode<T>*ptr) {
        TreeUnitNode<T>* newUnit = new TreeUnitNode<T> (childValue);
        return insertChild(newUnit, ptr);
    };
    TreeUnitNode<T>* insertChild(TreeUnitNode<T>* child, TreeUnitNode<T>*ptr) {
        ptr->insertChild(child);
        return child;
    };
    void insertTree(TreeBaseNode<T>&tree){
        bool have_same_ele = false;
        tree.apply([&have_same_ele, this](TreeUnitNode<T>* e){
            if(this->getUnit(e))
                have_same_ele = true;
        });
        ICHECK(have_same_ele == false) << "cannot insert tree with identical element.";
        insertChild(tree.base->pChild, base);
        tree.is_subTree = 1;
    }
    void insertTree(const T & pos, TreeBaseNode<T> & tree){
        bool have_same_ele = false;
        tree.apply([&have_same_ele, this](TreeUnitNode<T>* e){
            if(this->getUnit(e))
                have_same_ele = true;
        });
        ICHECK(have_same_ele == false) << "cannot insert tree with identical element.";
        insertChild(tree.base->pChild, pos);
        tree.is_subTree = 1;
    }
    /*!
    *   \brief insert value v at ptr. v is child of ptr. v has children of ptr.
    */
    TreeUnitNode<T>* insert(TreeUnitNode<T>*ptr, const T&v){
        if (getUnit(v)!=NULL){
            LOG(FATAL) << "cannot insert a inserted object\n";
            return NULL;
        }
        TreeUnitNode<T> * node_ = new TreeUnitNode<T>(v);
        node_->pChild = ptr->pChild;
        ptr->pChild = node_;
        return node_;
    }
    TreeUnitNode<T>* insert(const T& pos, const T&v){
        return insert(getUnit(pos), v);
    }
    TreeUnitNode<T>* insert(const T& v){
        return insert(base, v);
    }
    /*!
    *   \brief erase ptr. ptr's father succeed ptr's child.
    */
    TreeUnitNode<T>* erase(TreeUnitNode<T> *pos){
        if(pos == NULL){
            return NULL;
        }
        TreeUnitNode<T>*parent = Parent(pos);
        if(parent->pChlid == pos){
            TreeUnitNode<T>*lastChild = pos->pChild;
            if (lastChild == NULL)
                parent->pChild = pos->pSibling;
            else
            {
                parent->pChild = pos->pChild;
                while(lastChild->pSibling){
                    lastChild = lastChild->pSibling;
                }
                lastChild->pSibling = pos->pSibling;
            }
        }
        else{
            TreeUnitNode<T>* prev;
            for(prev = parent->pChild; prev -> pSibling != pos; prev = prev->pSibling);
            TreeUnitNode<T>*lastChild = pos->pChild;
            if (lastChild == NULL)
                prev-> pSibling= pos->pSibling;
            else
            {
                prev->pSibling = pos->pChild;
                while(lastChild->pSibling){
                    lastChild = lastChild->pSibling;
                }
                lastChild->pSibling = pos->pSibling;
            }
        }
        delete pos;
        return parent;
    }
    TreeUnitNode<T>* erase(const T&v){
        return erase(getUnit(v));
    }
    void eraseTree(const T& v){
        // deconstruct the tree.
        // TreeBaseNode<T> tmp = getSubTree(v);
        // tmp.is_subTree = 0;
        TreeUnitNode<T>* ptr = getUnit(v);
        TreeUnitNode<T>* parent = Parent(ptr);
        if(parent->pChild == ptr){
            parent->pChild = ptr->pSibling;
        }
        else{
            for(TreeUnitNode<T>*iter = parent->pChild; iter->pSibling != NULL; iter = iter->pSibling){
                if(iter->pSibling == ptr){
                    iter->pSibling = ptr->pSibling;
                    break;
                }
            }
        }
        std::queue<TreeUnitNode<T>* > Q;
        Q.push(ptr->pChild);
        while(!Q.empty()){
            TreeUnitNode<T> * tmp = Q.front();
            Q.pop();
            if(tmp->pChild) Q.push(tmp->pChild);
            if(tmp->pSibling) Q.push(tmp->pSibling);
            delete tmp;
        }
        delete ptr;
    }
    bool isEmpty(){return base->pChild == NULL;}
    /*!
    *   \brief get parent of current node.
    */
    TreeUnitNode<T>* Parent(TreeUnitNode<T> *current){
        std::queue<TreeUnitNode<T>*> q;
        q.push(base);
        while(!q.empty()){
            TreeUnitNode<T>* father = q.front();
            q.pop();
            TreeUnitNode<T>* child = father->pChild;
            while(child){
                if (child == current)
                    return father;
                q.push(child);
                child = child->pSibling;
            }
        }
        return NULL; 
    }
    /*!
    *   \brief rootfirst, rootlast, width traverse.
    */
    void RootFirstTraverse(TreeUnitNode<T>* root, std::function<void (TreeUnitNode<T>*)> const & f){
        if (root == NULL) return;
        auto ptr = root->pChild;
        f(root);
        while(ptr){
            RootFirstTraverse(ptr, f);
            ptr = ptr->pSibling;
        } 
    }
    void RootFirstTraverse(TreeUnitNode<T>* root, std::function<void (T &)> const & f){
        if (root == NULL) return;
        auto ptr = root->pChild;
        if(root-> data_ptr)
            f(*root->data_ptr);
        while(ptr){
            RootFirstTraverse(ptr, f);
            ptr = ptr->pSibling;
        } 
    }
    void RootLastTraverse(TreeUnitNode<T>*root, std::function<void (TreeUnitNode<T>*)> const & f){
        if (root == NULL) return;
        auto ptr = root->pChild;
        while(ptr){
            RootLastTraverse(ptr, f);
            ptr = ptr->pSibling;
        } 
        f(root);
    }
    void RootLastTraverse(TreeUnitNode<T>*root, std::function<void (T &)> const & f){
        if (root == NULL) return;
        auto ptr = root->pChild;
        while(ptr){
            RootLastTraverse(ptr, f);
            ptr = ptr->pSibling;
        } 
        if(root->data_ptr)
            f(*root->data_ptr);
    }
    void WidthTraverse(std::function<void (TreeUnitNode<T>*)> const & f){
        std::queue<TreeUnitNode<T>* > q;
        q.push(base->pChild);
        while(!q.empty()){
            auto ptr = q.front();
            q.pop();
            if(!ptr) continue;
            for(; ptr; ptr = ptr->pSibling){
                f(ptr);
                q.push(ptr->pChild);
            }
        }
    }
    void WidthTraverse(std::function<void (T &)> const & f){
        std::queue<TreeUnitNode<T>* > q;
        q.push(base->pChild);
        while(!q.empty()){
            auto ptr = q.front();
            q.pop();
            if(!ptr) continue;
            for(; ptr; ptr = ptr->pSibling){
                if(ptr->data_ptr)
                    f(*ptr->data_ptr);
                q.push(ptr->pChild);
            }
        }
    }
    void display(std::string name = ""){
        std::cout <<"--------------display begin----------------\n";
        std::cout << "name: " << name << std::endl;
        std::cout << "is_subTree: " << is_subTree << std::endl;
        std::queue<TreeUnitNode<T> *> q;
        q.push(base);
        while(!q.empty()){
            TreeUnitNode<T>* father = q.front();
            q.pop();
            if(father == base)
                std::cout << "the root is: ";
            else 
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
    T* data_ptr;
    TreeUnitNode<T>*pChild;
    TreeUnitNode<T>*pSibling;
    // TreeUnitNode():data_ptr(NULL), pChild(NULL), pSibling(NULL){
    // }
    TreeUnitNode(): data_ptr(NULL), pChild(NULL), pSibling(NULL){}
    TreeUnitNode(const T & data): pChild(NULL), pSibling(NULL){
        data_ptr = new T(data);
    }
    ~TreeUnitNode(){
        if(data_ptr != NULL)
        { 
            data_ptr->T::~T();
        }
    }

    std::string Value(){return data_ptr->str();}
    TreeUnitNode<T>* deepCopy(std::function<T(const T&)> const & f){
        TreeUnitNode<T> * node_ = new TreeUnitNode<T>(f(*data_ptr));
        if(pChild)
            node_->pChild = pChild->deepCopy(f);
        if(pSibling)
            node_->pSibling = pSibling->deepCopy(f);
        return node_;
    }
    void setValue(const T & value){
        delete data_ptr;
        *data_ptr = value;
    }
    void insertChild(TreeUnitNode<T>* node){
        if(pChild)
            node->pSibling = pChild;
        pChild = node;
    }
    void deleteChild(TreeUnitNode<T>* node){
        if (node == pChild){
            pChild = node -> pSibling;
            return;
        }
        for(auto p = pChild; p->pSibling != NULL; p = p->pSibling){
            if (p->pSibling == node){
                p->pSibling = p->pSibling-> pSibling;
                return;
            }
        }
        LOG(WARNING) << "delete node not found\n";
        return; 
    }
};
} //end hybrid 
} //end ditto
