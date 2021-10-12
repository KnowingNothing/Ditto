#include <hybrid/tree.h>



using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{
template<class T>
TreeUnitNode<T>* TreeUnitNode<T>::deepCopy(std::function<const T&(const T&)> const & f){
    TreeUnitNode<T> * node_ = new TreeUnitNode<T>(f(dataRef));
    if(pChild)
        node_->pChild = pChild->deepCopy(f);
    if(pSibling)
        node_->pSibling = pSibling->deepCopy(f);
    return node_;
}
template<class T>
TreeUnitNode<T>::TreeUnitNode(const T & value):dataRef(value), pChild(NULL), pSibling(NULL){
}
template<class T>
T TreeUnitNode<T>::Value(){
    return dataRef;
}
template<class T>
void TreeUnitNode<T>::setParent(TreeUnitNode<T>*pointer){
    pointer->InsertChild(this);
}
template<class T> 
void TreeUnitNode<T>::insertChild(TreeUnitNode<T>* node){
    if(pChild)
        node->pSibling = pChild;
    pChild = node;
}
template<class T>
void TreeUnitNode<T>::deleteChild(TreeUnitNode<T>* node){
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
    std::cout << "Warn: node not found\n";
    return; 
} 
template<class T>
TreeNode<T>::~TreeNode(){
    if(!is_subTree)
        apply([](TreeUnitNode<T>* node){delete node;}, "RootLast");
    delete base;
}
template<class T>
TreeUnitNode<T>* TreeNode<T>::getRoot(){
    return base->pChild;
};
template<class T>
TreeUnitNode<T>* TreeNode<T>::getUnit(const T & value){
    std::queue <TreeUnitNode<T>* > Q;
    Q.push(base->pChild);
    while(!Q.empty()){
        auto p = Q.front();
        Q.pop();
        if(p == NULL) continue;
        if(p->dataRef == value)
            return p;
        Q.push(p->pChild);
        Q.push(p->pSibling);
    }
    return NULL;
}

template<class T>
bool TreeNode<T>::is_parent(TreeUnitNode<T>*parent, TreeUnitNode<T>*child){
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
template<class T>
bool TreeNode<T>::is_parent(const T&parent, const T&child){
    return is_parent(getUnit(parent), getUnit(child));
}
template<class T>
TreeNode<T> TreeNode<T>::deepCopy(std::function<const T&(const T&)>const &f){
    TreeUnitNode<T> * root_ = base->pChild->deepCopy(f);
    return TreeNode<T>(root_);
}
template<class T> 
void TreeNode<T>::RootFirstTraverse(TreeUnitNode<T>* root, std::function<void (TreeUnitNode<T>*)> const & f){
    if (root == NULL) return;
    auto ptr = root->pChild;
    f(root);
    while(ptr){
        RootFirstTraverse(ptr, f);
        ptr = ptr->pSibling;
    } 
    
}
template<class T>
void TreeNode<T>::RootLastTraverse(TreeUnitNode<T>* root, std::function<void (TreeUnitNode<T>*)> const & f){
    if (root == NULL) return;
    auto ptr = root->pChild;
    while(ptr){
        RootLastTraverse(ptr, f);
        ptr = ptr->pSibling;
    } 
    f(root);
}
template<class T>
void TreeNode<T>::WidthTraverse(std::function<void (TreeUnitNode<T>*)> const & f){
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
template<class T>
void TreeNode<T>::apply(std::function<void (TreeUnitNode<T>*)> const & f, std::string Mode){
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
        std::cout << "no traverse matched";
    }
}
template<class T>
TreeUnitNode<T>* TreeNode<T>::Parent(TreeUnitNode<T>*current){
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
template<class T>
TreeNode<T> TreeNode<T>::getSubTree(TreeUnitNode<T>* root_){
    TreeNode<T> subTree(root_, true);
    return subTree; 
}
/*
TreeUnitNode<T>* insert(TreeUnitNode<T>*ptr, const T&v);
TreeUnitNode<T>* insert(const T& pos, const T&v);
TreeUnitNode<T>* erase(TreeUnitNode<T> *ptr);
TreeUnitNode<T>* erase(const T&pos, const T&v);
*/
template<class T>
TreeUnitNode<T>* TreeNode<T>::insert(TreeUnitNode<T>*ptr, const T&v){
    if (is_subTree && ptr == getRoot()){
        std::cout << "Error: cannot insert in a subTree's root\n";
        return NULL;
    }
    if (getUnit(v)!=NULL){
        std::cout << "Error: cannot insert a inserted object\n";
        return NULL;
    }
    TreeUnitNode<T> * node_ = new TreeUnitNode<T>(v);
    if (ptr == NULL)
        ptr = base;
    node_->pChild = ptr->pChild;
    ptr->pChild = node_;
    return node_;
}
template<class T>
TreeUnitNode<T>* insert(const T & pos, const T & v){
    return insert(getUnit(pos), v);
}
template<class T>
TreeUnitNode<T> *erase(TreeUnitNode<T>* pos){
    if(pos == NULL){
        std::cout << "Error: pos not found\n";
        return NULL;
    }
    TreeUnitNode<T>*parent = Parent(pos);
    TreeUnitNode<T>*lastChild = pos -> pChild;
    if (lastChild == NULL)
        parent->pChild = pos->pSibling;
    else
        parent->pChild = pos->pChild;
    while(lastChild->pSibling){
        lastChild = lastChild->pSibling;
    }
    lastChild->pSibling = pos->pSibling;
    return parent;
}
template<class T>
TreeUnitNode<T> *erase(const T& v){
    return erase(getUnit(v));
}
} // end hybrid 
} // end ditto