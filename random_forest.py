#!/usr/bin/env python3
"""random_forest - Ensemble classifier."""
import sys,argparse,json,math,random
from collections import Counter
def entropy(labels):
    n=len(labels);freq=Counter(labels)
    return -sum((c/n)*math.log2(c/n) for c in freq.values() if c>0)
def build_tree(X,y,max_depth=3,n_features=None,depth=0):
    if depth>=max_depth or len(set(y))<=1:return Counter(y).most_common(1)[0][0]
    d=len(X[0]);feats=random.sample(range(d),min(n_features or int(math.sqrt(d)),d))
    best_gain=-1;best_f=0;best_v=0
    base=entropy(y)
    for f in feats:
        vals=sorted(set(x[f] for x in X))
        for v in vals:
            l=[yi for xi,yi in zip(X,y) if xi[f]<=v];r=[yi for xi,yi in zip(X,y) if xi[f]>v]
            if not l or not r:continue
            g=base-(len(l)/len(y)*entropy(l)+len(r)/len(y)*entropy(r))
            if g>best_gain:best_gain=g;best_f=f;best_v=v
    if best_gain<=0:return Counter(y).most_common(1)[0][0]
    lX,ly,rX,ry=[],[],[],[]
    for xi,yi in zip(X,y):
        if xi[best_f]<=best_v:lX.append(xi);ly.append(yi)
        else:rX.append(xi);ry.append(yi)
    return {"f":best_f,"v":best_v,"l":build_tree(lX,ly,max_depth,n_features,depth+1),"r":build_tree(rX,ry,max_depth,n_features,depth+1)}
def predict_tree(tree,x):
    if not isinstance(tree,dict):return tree
    return predict_tree(tree["l"] if x[tree["f"]]<=tree["v"] else tree["r"],x)
def main():
    p=argparse.ArgumentParser(description="Random forest")
    p.add_argument("--trees",type=int,default=10);p.add_argument("--depth",type=int,default=4)
    p.add_argument("--samples",type=int,default=200)
    args=p.parse_args()
    random.seed(42)
    X=[];y=[]
    for _ in range(args.samples//3):
        X.append([random.gauss(1,1),random.gauss(1,1)]);y.append("A")
        X.append([random.gauss(4,1),random.gauss(1,1)]);y.append("B")
        X.append([random.gauss(2.5,1),random.gauss(4,1)]);y.append("C")
    split=int(len(X)*0.8)
    forest=[]
    for _ in range(args.trees):
        idx=[random.randint(0,split-1) for _ in range(split)]
        bX=[X[i] for i in idx];by=[y[i] for i in idx]
        forest.append(build_tree(bX,by,args.depth))
    correct=0
    for xi,yi in zip(X[split:],y[split:]):
        votes=Counter(predict_tree(t,xi) for t in forest)
        if votes.most_common(1)[0][0]==yi:correct+=1
    print(json.dumps({"trees":args.trees,"max_depth":args.depth,"accuracy":round(correct/len(X[split:]),4),"test_size":len(X)-split},indent=2))
if __name__=="__main__":main()
