#include <hybrid/tree.h>
#include <hybrid/test.h>


using namespace tvm;
using namespace te;

namespace ditto{
namespace hybrid{

std::string to_string__(IterVar i){
    return i->var->name_hint.operator std::string();
}

std::string to_string__(test i){
    return i.str();
}

}  // namespace hybrid
}  // namespace ditto

// /*!
// * \brief class tree
// */
// // template<typename T>
// // Tree<T>::Tree(){
// //     auto node = make_object<TreeNode<T> >();
// //     data_ = node;
// // }
// /*! 
// * \brief copy with function
// */
// // template<typename T>
// // Tree<T>::Tree(const Tree<T> & tree, std::function<T (const T &)>const & f){
// //     TreeBaseNode<T> node = tree.operator->()->deepCopy(f);
// //     auto node_ = make_object<TreeNode<T> >(node);
// //     data_ = node;
// // }
// /*!
// * \brief insertChild
// */
// template<typename T>
// void Tree<T>::insertChild(const T & p, const T & v){operator->()->insertChild(p, v);}
// template<typename T>
// void Tree<T>::insertChild(const T &p, const Tree<T> & tree){}
// /*!
// * \brief insert
// */
// template<typename T>
// void Tree<T>::insert(const T& v){operator->()->insert(v);}
// /*!
// * \brief erase
// */
// template<typename T>
// void Tree<T>::erase(const T& v){operator->()->erase(v);}
// /*!
// * \brief display
// */
// template<typename T>
// void Tree<T>::display(){operator->()->display();}
// /*!
// *   \brief getSubTree
// *   \brief t, the root node on the tree
// */
// template<typename T>
// Tree<T> Tree<T>::getSubTree(const T& t){return operator->()->getSubTree(t);}
// /*!
// *   \brief check where child is in the subtree of parent.
// */
// template<typename T>
// bool Tree<T>::is_parent(const T& parent, const T& child){
//     return operator->()->is_parent(parent, child);
// }

// /*!
//     \brief TreeNode
// */



// // template<class T>
// // TreeUnitNode<T>* TreeUnitNode<T>::deepCopy(std::function<T(const T&)> const & f){
// //     TreeUnitNode<T> * node_ = new TreeUnitNode<T>(f(*data_ptr));
// //     if(pChild)
// //         node_->pChild = pChild->deepCopy(f);
// //     if(pSibling)
// //         node_->pSibling = pSibling->deepCopy(f);
// //     return node_;
// // }
// template<class T>
// T TreeUnitNode<T>::Value(){
//     return *data_ptr;
// }
// template<class T> 
// void TreeUnitNode<T>::insertChild(TreeUnitNode<T>* node){
//     if(pChild)
//         node->pSibling = pChild;
//     pChild = node;
// }
// template<class T>
// void TreeUnitNode<T>::deleteChild(TreeUnitNode<T>* node){
//     if (node == pChild){
//         pChild = node -> pSibling;
//         return;
//     }
//     for(auto p = pChild; p->pSibling != NULL; p = p->pSibling){
//         if (p->pSibling == node){
//             p->pSibling = p->pSibling-> pSibling;
//             return;
//         }
//     }
//     LOG(WARNING) << "delete node not found\n";
//     return; 
// } 
// // template<class T>
// // TreeBaseNode<T>::~TreeBaseNode(){
// //     if(!is_subTree)
// //         apply([](TreeUnitNode<T>* node){
// //             node->data_ptr->T::~T();
// //             delete node;
// //             }, "RootLast");
// //     base->data_ptr->T::~T();
// //     delete base;
// // }
// template<class T>
// TreeUnitNode<T>* TreeBaseNode<T>::getRoot(){
//     return base->pChild;
// };
// template<class T>
// TreeUnitNode<T>* TreeBaseNode<T>::getUnit(const T & value){
//     std::queue <TreeUnitNode<T>* > Q;
//     Q.push(base->pChild);
//     while(!Q.empty()){
//         auto p = Q.front();
//         Q.pop();
//         if(p == NULL) continue;
//         if(p->dataRef->same_as(value))
//             return p;
//         Q.push(p->pChild);
//         Q.push(p->pSibling);
//     }
//     return NULL;
// }

// template<class T>
// bool TreeBaseNode<T>::is_parent(TreeUnitNode<T>*parent, TreeUnitNode<T>*child){
//     if(!parent||!child){
//         return false;
//     }
//     bool isParent = false;
//     auto func = [&isParent, &child](TreeUnitNode<T>* p){
//         if(p == child){
//             isParent = true; 
//         }
//     };
//     getSubTree(parent).apply(func);
//     return isParent;
// }
// template<class T>
// bool TreeBaseNode<T>::is_parent(const T&parent, const T&child){
//     return is_parent(getUnit(parent), getUnit(child));
// }
// // template<class T>
// // TreeBaseNode<T> TreeBaseNode<T>::deepCopy(std::function<T(const T&)>const &f){
// //     TreeUnitNode<T> * root_ = base->pChild->deepCopy(f);
// //     return TreeBaseNode<T>(root_);
// // }
// // template<class T> 
// // void TreeBaseNode<T>::RootFirstTraverse(TreeUnitNode<T>* root, std::function<void (TreeUnitNode<T>*)> const & f){
// //     if (root == NULL) return;
// //     auto ptr = root->pChild;
// //     f(root);
// //     while(ptr){
// //         RootFirstTraverse(ptr, f);
// //         ptr = ptr->pSibling;
// //     } 
    
// // }
// // template<class T>
// // void TreeBaseNode<T>::RootLastTraverse(TreeUnitNode<T>* root, std::function<void (TreeUnitNode<T>*)> const & f){
// //     if (root == NULL) return;
// //     auto ptr = root->pChild;
// //     while(ptr){
// //         RootLastTraverse(ptr, f);
// //         ptr = ptr->pSibling;
// //     } 
// //     f(root);
// // }
// // template<class T>
// // void TreeBaseNode<T>::WidthTraverse(std::function<void (TreeUnitNode<T>*)> const & f){
// //     std::queue<TreeUnitNode<T>* > q;
// //     q.push(base->pChild);
// //     while(!q.empty()){
// //         auto ptr = q.front();
// //         q.pop();
// //         if(!ptr) continue;
// //         for(; ptr; ptr = ptr->pSibling){
// //             f(ptr);
// //             q.push(ptr->pChild);
// //         }
// //     }
// // }
// // template<class T>
// // void TreeBaseNode<T>::apply(std::function<void (TreeUnitNode<T>*)> const & f, std::string Mode){
// //     if (Mode == "RootFirst"){
// //         RootFirstTraverse(base->pChild, f);
// //     }
// //     else if(Mode == "RootLast"){
// //         RootLastTraverse(base->pChild, f);
// //     }
// //     else if(Mode == "Width"){
// //         WidthTraverse(f);
// //     }
// //     else{
// //         std::cout << "no traverse matched";
// //     }
// // }
// template<class T>
// TreeUnitNode<T>* TreeBaseNode<T>::Parent(TreeUnitNode<T>*current){
//     std::queue<TreeUnitNode<T>*> q;
//     q.push(base);
//     while(!q.empty()){
//         TreeUnitNode<T>* father = q.front();
//         q.pop();
//         TreeUnitNode<T>* child = father->pChild;
//         while(child){
//             if (child == current)
//                 return father;
//             q.push(child);
//             child = child->pSibling;
//         }
//     }
//     return NULL; 
// }
// template<class T>
// TreeBaseNode<T> TreeBaseNode<T>::getSubTree(TreeUnitNode<T>* root_){
//     TreeBaseNode<T> subTree(root_, true);
//     return subTree; 
// }
// template <typename T>
// void Tree<T>::setValue(const T & pos, const T & v){
//     auto ptr = operator->().getUnit(pos);
//     ICHECK(ptr) << "reference value not found in tree.";
//     ptr->setValue(v);
// }
// /*
// TreeUnitNode<T>* insert(TreeUnitNode<T>*ptr, const T&v);
// TreeUnitNode<T>* insert(const T& pos, const T&v);
// TreeUnitNode<T>* erase(TreeUnitNode<T> *ptr);
// TreeUnitNode<T>* erase(const T&pos, const T&v);
// */
// template<class T>
// TreeUnitNode<T>* TreeBaseNode<T>::insert(TreeUnitNode<T>*ptr, const T&v){
//     if (is_subTree && ptr == getRoot()){
//         LOG(FATAL) << "Error: cannot insert in a subTree's root\n";
//         return NULL;
//     }
//     if (getUnit(v)!=NULL){
//         LOG(FATAL) << "cannot insert a inserted object\n";
//         return NULL;
//     }
//     TreeUnitNode<T> * node_ = new TreeUnitNode<T>(v);
//     if (ptr == NULL)
//         ptr = base;
//     node_->pChild = ptr->pChild;
//     ptr->pChild = node_;
//     return node_;
// }
// template<class T>
// TreeUnitNode<T>* insert(const T & pos, const T & v){
//     return insert(getUnit(pos), v);
// }
// template<class T>
// TreeUnitNode<T> *erase(TreeUnitNode<T>* pos){
//     if(pos == NULL){
//         std::cout << "Error: pos not found\n";
//         return NULL;
//     }
//     TreeUnitNode<T>*parent = Parent(pos);
//     TreeUnitNode<T>*lastChild = pos -> pChild;
//     if (lastChild == NULL)
//         parent->pChild = pos->pSibling;
//     else
//         parent->pChild = pos->pChild;
//     while(lastChild->pSibling){
//         lastChild = lastChild->pSibling;
//     }
//     lastChild->pSibling = pos->pSibling;
//     return parent;
// }
// template<class T>
// TreeUnitNode<T> *erase(const T& v){
//     return erase(getUnit(v));
// }
// } // end hybrid 
// } // end ditto