#!/usr/bin/env python3
"""random_forest - Simple random forest classifier."""
import sys,math,random
from collections import Counter
def gini(groups,classes):
    total=sum(len(g) for g in groups);score=0
    for g in groups:
        if not g:continue
        s=len(g);counts=Counter(row[-1] for row in g)
        score+=sum((c/s)**2 for c in counts.values())*s/total
    return 1-score
def split(data,idx,val):
    left=[r for r in data if r[idx]<val];right=[r for r in data if r[idx]>=val]
    return left,right
def best_split(data,n_features):
    features=random.sample(range(len(data[0])-1),min(n_features,len(data[0])-1))
    best={"score":1};
    for f in features:
        for row in data:
            l,r=split(data,f,row[f]);g=gini([l,r],set(row[-1] for row in data))
            if g<best["score"]:best={"idx":f,"val":row[f],"score":g,"left":l,"right":r}
    return best
def build_tree(data,max_depth,min_size,n_features,depth=0):
    if depth>=max_depth or len(data)<=min_size:return Counter(r[-1] for r in data).most_common(1)[0][0]
    node=best_split(data,n_features)
    if not node.get("left") or not node.get("right"):return Counter(r[-1] for r in data).most_common(1)[0][0]
    node["left"]=build_tree(node["left"],max_depth,min_size,n_features,depth+1)
    node["right"]=build_tree(node["right"],max_depth,min_size,n_features,depth+1)
    return node
def predict_tree(node,row):
    if not isinstance(node,dict):return node
    return predict_tree(node["left"] if row[node["idx"]]<node["val"] else node["right"],row)
def random_forest(data,n_trees=10,max_depth=5,sample_ratio=0.8):
    n=len(data);n_features=int(math.sqrt(len(data[0])-1))+1;trees=[]
    for _ in range(n_trees):
        sample=[data[random.randint(0,n-1)] for _ in range(int(n*sample_ratio))]
        trees.append(build_tree(sample,max_depth,2,n_features))
    return trees
def predict_rf(trees,row):
    votes=[predict_tree(t,row) for t in trees];return Counter(votes).most_common(1)[0][0]
if __name__=="__main__":
    random.seed(42);data=[]
    for _ in range(50):data.append([random.gauss(-1,1),random.gauss(-1,1),0])
    for _ in range(50):data.append([random.gauss(1,1),random.gauss(1,1),1])
    trees=random_forest(data)
    correct=sum(1 for r in data if predict_rf(trees,r)==r[-1])
    print(f"Trees: {len(trees)}, Accuracy: {correct/len(data):.1%}")
