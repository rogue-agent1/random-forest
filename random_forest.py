#!/usr/bin/env python3
"""Random Forest classifier — zero-dep implementation."""
import random, math
from collections import Counter

def gini(y):
    c=Counter(y); n=len(y)
    return 1-sum((v/n)**2 for v in c.values()) if n else 0

def best_split(X, y, features):
    best_g=float('inf'); best_f=None; best_v=None
    for f in features:
        vals=sorted(set(r[f] for r in X))
        for i in range(len(vals)-1):
            v=(vals[i]+vals[i+1])/2
            left=[yi for xi,yi in zip(X,y) if xi[f]<=v]
            right=[yi for xi,yi in zip(X,y) if xi[f]>v]
            if not left or not right: continue
            g=(len(left)*gini(left)+len(right)*gini(right))/len(y)
            if g<best_g: best_g=g; best_f=f; best_v=v
    return best_f, best_v

def build_tree(X, y, max_depth=10, max_features=None, depth=0):
    if depth>=max_depth or len(set(y))<=1: return Counter(y).most_common(1)[0][0]
    d=len(X[0]); feats=random.sample(range(d),min(max_features or int(math.sqrt(d)),d))
    f,v=best_split(X,y,feats)
    if f is None: return Counter(y).most_common(1)[0][0]
    li=[(xi,yi) for xi,yi in zip(X,y) if xi[f]<=v]
    ri=[(xi,yi) for xi,yi in zip(X,y) if xi[f]>v]
    if not li or not ri: return Counter(y).most_common(1)[0][0]
    return {"f":f,"v":v,
            "l":build_tree([x for x,_ in li],[y for _,y in li],max_depth,max_features,depth+1),
            "r":build_tree([x for x,_ in ri],[y for _,y in ri],max_depth,max_features,depth+1)}

def predict_tree(tree, x):
    if not isinstance(tree,dict): return tree
    return predict_tree(tree["l"] if x[tree["f"]]<=tree["v"] else tree["r"], x)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees=n_trees; self.max_depth=max_depth; self.trees=[]
    def fit(self, X, y):
        n=len(X)
        for _ in range(self.n_trees):
            idx=[random.randint(0,n-1) for _ in range(n)]
            self.trees.append(build_tree([X[i] for i in idx],[y[i] for i in idx],self.max_depth))
    def predict(self, X):
        return [Counter(predict_tree(t,x) for t in self.trees).most_common(1)[0][0] for x in X]

if __name__=="__main__":
    random.seed(42)
    X=[[random.gauss(0,1),random.gauss(0,1)] for _ in range(50)]+[[random.gauss(2,1),random.gauss(2,1)] for _ in range(50)]
    y=[0]*50+[1]*50
    rf=RandomForest(n_trees=20,max_depth=5); rf.fit(X,y)
    acc=sum(p==a for p,a in zip(rf.predict(X),y))/len(y)
    print(f"Random Forest accuracy: {acc:.0%} ({rf.n_trees} trees)")
