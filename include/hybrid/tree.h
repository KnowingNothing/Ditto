#pragma once 

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>
#include <hybrid/test.h>

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
    TreeNode(TreeBaseNode<T> & t, int is_subTree = 0): TreeBaseNode<T>(t, is_subTree){
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
        // std::cout << operator->()->check_parent() << std::endl;
    }
    explicit Tree(ObjectPtr<Object> n) : ObjectRef(n){
        // std::cout << operator->()->check_parent() << std::endl;
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
        // the TreeBaseNode object created by deepCopy is directly the node; no TreeBaseNode copy called
        TreeBaseNode<T> node = tree.operator->()->deepCopy(f);
        node.is_subTree = 0;
        // node_ has the different base but same child as node
        auto node_ = make_object<TreeNode<T> >(node);
        // only delete base in node;
        node.is_subTree = 1;
        data_ = node_;
        // std::cout << "deep copy: parent relationship right? "; 
        // std::cout << operator->()->check_parent() << std::endl;
    }
    inline TreeNode<T>* operator->() const;
    int is_subTree(){
        return operator->()->is_subTree;
    }
    bool check_parent(bool modify = true) const {
        return operator->()->check_parent(modify);
    }
    TreeUnitNode<T>* getRoot(){
        return operator->()->getRoot();
    }
    /*!
    * \brief insertChild at root
    * \param v the value to insert at root
    */
    void insertChild(TreeUnitNode<T> * t){
        operator->()->insertChild(t);
    }
    void insertChild(const T& v){
        operator->()->insertChild(v);}
    /*!
    *   \brief insertChild
    *   \param p the parent position
    *   \param v the child value
    */
   void insertChild(TreeUnitNode<T> * p, TreeUnitNode<T> * v){
        operator->()->insertChild(v, p);
    }
    void insertChild(const T & p, const T & v){
        operator->()->insertChild(v, p);}
    /*!
    *   \brief insertTree
    *   \param p the parent
    *   \param tree the tree to insert
    */
   void insertTree(TreeUnitNode<T> * p, Tree<T> & tree){
        operator->()->insertTree(p, *tree.operator->());
    }
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
    void insert(TreeUnitNode<T>* v){operator->()->insert(v);}
    void insert(const T& v){operator->()->insert(v);}
    /*!
    *   \brief insert at p
    *   \param  p the parant value
    *   \param  v the value to insert
    */
    void insert(TreeUnitNode<T>* p, TreeUnitNode<T>* v){operator->()->insert(p, v);}
    void insert(const T & p, const T & v){operator->()->insert(p, v);}
    /*!
    *   \brief erase
    *   \param  v value to erase
    */
    /*!
    *   \brief replace the old node with the new node
    */
    void replace(const T & old, const T & now){
        operator->() -> replace(old, now);
    }
    void erase(TreeUnitNode<T>* v){operator->()->erase(v);}
    void erase(const T& v){operator->()->erase(v);}
    /*!
    * \brief display the tree structure 
    */
    void display(std::string name = "", std::string attr = ""){operator->()->display(name, attr);}
    /*!
    *   \brief get the SubTree, note that this use the old tree, not a new one. 
    *   \param t, the root node on the tree
    */
    Tree<T> getSubTree(TreeUnitNode<T>* t){
        TreeBaseNode<T> node = operator->()->getSubTree(t);
        // delete nothing
        auto node_ = make_object<TreeNode<T> >(node, 1);
        // delete root
        Tree<T>ret = Tree<T>(node_);

        // node_->is_subTree = 2;
        return ret;
    }
    Tree<T> getSubTree(const T& t){
        TreeBaseNode<T> node = operator->()->getSubTree(t);
        // delete nothing
        auto node_ = make_object<TreeNode<T> >(node, 1);
        // delete root
        Tree<T>ret = Tree<T>(node_);
        // node_->is_subTree = 2;

        return ret;
    }
    /*!
    *   \brief erase the subTree with root t;
    */
   void eraseTree(TreeUnitNode<T>* t){
        operator->()->eraseTree(t);
    }
    void eraseTree(const T& t){
        operator->()->eraseTree(t);
    }
    /*!
    *   \brief apply a function to all nodes in tree; 
    */
    void apply(std::function<void (T & )> const & f, std::string mode="RootLast"){
        operator->()->apply(f, mode);
    }
    /*!
    *   \brief check where child is in the subtree of parent.
    */
    bool is_ancestor(TreeUnitNode<T>* parent, TreeUnitNode<T>* child){
        return operator->()->is_ancestor(parent, child);
    }
    bool is_ancestor(const T& parent, const T& child){
        return operator->()->is_ancestor(parent, child);
    }
    /*!
    *   \brief check whether parent is an immediate parent of child.
    */
    bool is_immediate_parent(const T& parent, const T& child){
        return operator->()->is_immediate_parent(parent, child);
    }
    /*!
    *   \brief count the number of children.
    */
    int count_children(const T& parent){
        return operator->()->count_child(parent);
    }
    T * get_parent_ptr(const T& child){
        return operator->()->get_parent_ptr(child);
    }
    TreeUnitNode<T> * get_parent_unit(const T & child){
        return operator->()->Parent(child);
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
        if(is_subTree == 0){
            for(TreeUnitNode<T> * temp = root_; temp != NULL; temp = temp->pSibling)
                temp->pParent = base;
        }
    }
    TreeBaseNode(const TreeBaseNode<T> & t, int is_subTree_ = 0): is_subTree(is_subTree_){
        base = new TreeUnitNode<T>();
        base->pChild = t.base->pChild;
        if (is_subTree == 0){
            for(TreeUnitNode<T> * temp = base->pChild; temp != NULL; temp = temp->pSibling){
                temp->pParent = base;
            }
        }
    }
    ~TreeBaseNode(){
        if(is_subTree == 0)
            apply([](TreeUnitNode<T>* node){
                    delete node;
                }, "RootLast");
        if(is_subTree != 2)
            delete base;
    }
    bool check_parent(bool modify = true){
        bool is_correct = true;
        std::queue<TreeUnitNode<T>* > Q;
        Q.push(base);
        while(!Q.empty()){
            TreeUnitNode<T> * father = Q.front();
            Q.pop();
            for(TreeUnitNode<T> * child = father->pChild; child != NULL; child = child->pSibling){
                if(child->pParent != father){
                    if(is_subTree == 0 || base != father){
                        is_correct = false;         
                        std::cout << "child value " << child ->Value() << std::endl;
                        if(father == base)
                            std::cout << "father is base" << std::endl;
                        else std::cout << "father is " << father ->Value() << std::endl;
                        if(child->pParent->data_ptr == NULL)
                            std::cout << "wrong father is base" << std::endl;
                        else std::cout << "wrong father is " << child ->pParent -> Value() << std::endl;
                        if (modify)
                            child->pParent = father;
                                       
                    }
                }
                Q.push(child);
            }
        }
        return is_correct;
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
        return TreeBaseNode<T>(ptr, 1);
    }
    TreeBaseNode<T> getSubTree(TreeUnitNode<T> * root_){
        return TreeBaseNode<T>(root_, 1);
    }
    bool is_immediate_parent(const T & parent, const T & child){
        TreeUnitNode<T> * child_ptr = getUnit(child);
        ICHECK(child_ptr) << "child not in the tree";
        TreeUnitNode<T> * p = Parent(child_ptr);
        TreeUnitNode<T> * p_ = getUnit(parent);
        ICHECK(p_) << "parent not in the tree";
        return p_ == p;
    }
    int count_child(TreeUnitNode<T>* ptr){
        return ptr->count_child();
    }
    int count_child(const T & parent){
        TreeUnitNode<T> * ptr = getUnit(parent);
        ICHECK(ptr) << "parent node not in the tree";
        return count_child(ptr);
    }
    /*!
    *   \brief check whether child is in the subtree of parent.  
    */
    bool is_ancestor(TreeUnitNode<T>*parent, TreeUnitNode<T>*child){
        ICHECK(parent) << "parent not in tree.";
        ICHECK(child) << "child not in tree.";
        while(child && child != parent){
            child = child->pParent;
        }
        return child == parent;
    }
    bool is_ancestor(const T& parent, const T&child){
        return is_ancestor(getUnit(parent), getUnit(child));
    }
    /*!
     * \brief deepCopy the tree. Use f to deep copy value.
     */
    TreeBaseNode<T> deepCopy(std::function<T(const T&)>const &f){
        TreeUnitNode<T> * root_ = NULL;
        if(base->pChild)
            root_ = base->pChild->deepCopy(f);
        return TreeBaseNode<T>(root_, 0);
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
        ICHECK(is_subTree == 0) << "cannot insert child at a subtree' s root";
        TreeUnitNode<T>* newUnit = new TreeUnitNode<T> (childValue);
        return insertChild(newUnit, ptr);
    };
    TreeUnitNode<T>* insertChild(TreeUnitNode<T>* child, TreeUnitNode<T>*ptr) {
        ICHECK(is_subTree == 0) << "cannot insert child at a subtree' s root";
        ptr->insertChild(child);
        return child;
    };
    void insertTree(TreeUnitNode<T> * pos, TreeBaseNode<T> & tree){
        bool have_same_ele = false;
        tree.apply([&have_same_ele, this](TreeUnitNode<T>* e){
            if(this->getUnit(e))
                have_same_ele = true;
        });
        ICHECK(have_same_ele == false) << "cannot insert tree with identical element.";
        pos->insertChild(tree.base->pChild);
        tree.is_subTree = 1;
    }
    void insertTree(TreeBaseNode<T>&tree){
        insertTree(base, tree);
    }
    void insertTree(const T & pos, TreeBaseNode<T> & tree){
        TreeUnitNode<T> * ptr = getUnit(pos);
        ICHECK(ptr) << "node to insert at is not in tree.";
        insertTree(ptr, tree);
    }
    TreeUnitNode<T>* insert(TreeUnitNode<T>* ptr, TreeUnitNode<T>* child){
        child->pChild = ptr->pChild;
        child->pParent = ptr;
        if(child->pChild){ 
            child->pChild->pParent = child;
        }
        ptr->pChild = child;
        return child;
    }
    TreeUnitNode<T>* insert(TreeUnitNode<T>*child){
        return insert(base, child);
    }
    /*!
    *   \brief insert value v at ptr. v is child of ptr. v has children of ptr.
    */
    TreeUnitNode<T>* insert(TreeUnitNode<T>*ptr, const T&v){
        ICHECK(is_subTree==0)<<"cannot insert in a subtree";
        if (getUnit(v)!=NULL){
            LOG(FATAL) << "cannot insert a inserted object\n";
            return NULL;
        }
        TreeUnitNode<T> * node_ = new TreeUnitNode<T>(v);
        return insert(ptr, node_);
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
        if(parent->pChild == pos){
            TreeUnitNode<T>*lastChild = pos->pChild;
            if (lastChild == NULL)
                parent->pChild = pos->pSibling;
            else
            {
                parent->pChild = pos->pChild;
                while(lastChild->pSibling){
                    lastChild = lastChild->pSibling;
                    lastChild->pParent = parent;
                }
                lastChild->pParent = parent;
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
                    lastChild->pParent = parent;
                }
                lastChild->pParent = parent;
                lastChild->pSibling = pos->pSibling;
            }
        }
        delete pos;
        return parent;
    }
    TreeUnitNode<T>* erase(const T&v){
        return erase(getUnit(v));
    }
    void eraseTree(TreeUnitNode<T>* ptr){
        if(ptr == NULL)
            return;
        TreeUnitNode<T>* parent = ptr->pParent;
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
        Q.push(ptr);
        while(!Q.empty()){
            TreeUnitNode<T> * tmp = Q.front();
            Q.pop();
            if(!tmp) continue;
            if(tmp->pChild) Q.push(tmp->pChild);
            if(tmp != ptr && tmp->pSibling) Q.push(tmp->pSibling);
            delete tmp;
        }
    }
    void eraseTree(const T& v){
        // deconstruct the tree.
        // TreeBaseNode<T> tmp = getSubTree(v);
        // tmp.is_subTree = 0;
        std::cout << "in TreeBaseNode::eraseTree(const T & v)\n";
        TreeUnitNode<T>* ptr = getUnit(v);
        ICHECK(ptr) << "erase node not in Tree";
        eraseTree(ptr);
    }
    void replace(const T & old, const T & now){
        TreeUnitNode<T> * self = getUnit(old);
        TreeUnitNode<T> * parent = Parent(self);
        ICHECK(self != NULL) << "the old node not in tree.";
        ICHECK(getUnit(now) == NULL) << " the new node is in the tree.";
        TreeUnitNode<T>* newUnit = new TreeUnitNode<T> (now);
        if(parent->pChild == self)
            parent -> pChild = newUnit;
        else{
            TreeUnitNode<T> * tmp = parent->pChild;
            while(tmp -> pSibling != self){
                tmp = tmp->pSibling;
            }
            tmp->pSibling = newUnit;
        }            
        newUnit->pChild = self->pChild;
        newUnit->pSibling = self->pSibling;
        newUnit->pParent = self->pParent;
        delete self;
    }
    bool isEmpty(){return base->pChild == NULL;}
    T * get_parent_ptr(const T & e){
        return getUnit(e)->pParent->data_ptr;
    }
    /*!
    *   \brief get parent of current node.
    */
    TreeUnitNode<T>* Parent(TreeUnitNode<T> *current){
        ICHECK(current) << "node not in tree.";
        return current->pParent;
        /*
        std::queue<TreeUnitNode<T>*> q;
        q.push(base);
        while(!q.empty()){
            TreeUnitNode<T>* father = q.front();
            q.pop();
            TreeUnitNode<T>* child = father->pChild;
            while(child){
                if (child == current){
                    if(father != current->pParent && (is_subTree == 0 || father != base))
                    {
                        std::cout << "pParent implemented incorrect." 
                        << std::endl;
                        std::cout << "child is" << child->Value() << std::endl;
                        if (father == base)
                            std::cout << "father is base." << std::endl;
                        else std::cout << "father is " << father->Value() << std::endl;
                        if (current -> pParent -> data_ptr)
                            std::cout << "wrong father is " << current -> pParent -> Value() << std::endl;
                        else std::cout << "wrong father is base" << std::endl;
                    }
                    return father;
                }
                q.push(child);
                child = child->pSibling;
            }
        }*/
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
    void display(std::string name = "", std::string attr = ""){
        std::cout <<"--------------display begin----------------\n";
        std::cout << "opName: " << name << std::endl;
        std::cout << "decription:   " << attr << std::endl;
        std::cout << "is_subTree: " << is_subTree << std::endl;
        std::cout << "parent correct? " << check_parent(false) << std::endl;
        struct format{
          TreeUnitNode<T>* p;
          int n_tab;  
          format(TreeUnitNode<T>* p_, int n_tab_): p(p_), n_tab(n_tab_){}
        };
        std::stack<format> s;
        s.push({base, -1});
        while(!s.empty()){
            format father = s.top();
            s.pop();
            if(father.p != base){
                for(int i = 0; i < father.n_tab; i++)
                    std::cout << "\t";
                std::cout <<"for " <<  father.p->Value();
            }
            std::cout << std::endl;
            auto son = father.p -> pChild;
            while(son){
                s.push({son, father.n_tab + 1});
                son = son -> pSibling;
            }
        }
        std::cout <<"---------------display end---------------\n";
        return;
    }
};

std::string to_string__(IterVar i);

std::string to_string__(test i);

template <typename T>
class TreeUnitNode{
public:
    // the dataRef is a ref to the reference of the data 
    T* data_ptr;
    TreeUnitNode<T>*pChild;
    TreeUnitNode<T>*pSibling;
    TreeUnitNode<T>*pParent;
    // TreeUnitNode():data_ptr(NULL), pChild(NULL), pSibling(NULL){
    // }
    TreeUnitNode(): data_ptr(NULL), pChild(NULL), pSibling(NULL), pParent(NULL){}
    TreeUnitNode(const T & data): pChild(NULL), pSibling(NULL), pParent(NULL){
        data_ptr = new T(data);
    }
    ~TreeUnitNode(){
        if(data_ptr != NULL){ 
            data_ptr->T::~T();
        }
    }

    std::string Value(){
        return to_string__(*data_ptr);
    }
    TreeUnitNode<T>* deepCopy(std::function<T(const T&)> const & f, TreeUnitNode<T> * parent = NULL){
        TreeUnitNode<T> * node_ = new TreeUnitNode<T>(f(*data_ptr));
        node_ ->pParent = parent;
        if(pChild)
            node_->pChild = pChild->deepCopy(f, node_);
        if(pSibling)
            node_->pSibling = pSibling->deepCopy(f, parent);
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
        node->pParent = this;
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
    int count_child(){
        TreeUnitNode<T> * child = pChild;
        int ret = 0;
        while(child){
            child = child->pSibling;
            ret += 1;
        }
        return ret;
    }
};
} //end hybrid 
} //end ditto
