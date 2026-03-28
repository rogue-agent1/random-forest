#!/usr/bin/env python3
"""Random Forest classifier from scratch."""
import sys,json,random,math
from collections import Counter

def gini(groups, classes):
    total = sum(len(g) for g in groups)
    score = 0.0
    for g in groups:
        if not g: continue
        s = len(g)
        for c in classes:
            p = sum(1 for r in g if r[-1]==c)/s
            score += p*(1-p)*s/total
    return score

def split(data, idx, val):
    l,r=[],[]
    for row in data:
        (l if row[idx]<val else r).append(row)
    return l,r

def best_split(data, n_features):
    classes = list(set(r[-1] for r in data))
    features = random.sample(range(len(data[0])-1), min(n_features, len(data[0])-1))
    best_idx,best_val,best_score,best_groups = 0,0,float('inf'),None
    for idx in features:
        for row in data:
            groups = split(data, idx, row[idx])
            g = gini(groups, classes)
            if g < best_score:
                best_idx,best_val,best_score,best_groups = idx,row[idx],g,groups
    return {'index':best_idx,'value':best_val,'groups':best_groups}

def to_terminal(group):
    outcomes = [r[-1] for r in group]
    return Counter(outcomes).most_common(1)[0][0]

def do_split(node, max_depth, min_size, n_features, depth):
    l,r = node['groups']; del node['groups']
    if not l or not r:
        node['left']=node['right']=to_terminal(l+r); return
    if depth >= max_depth:
        node['left'],node['right']=to_terminal(l),to_terminal(r); return
    node['left'] = best_split(l, n_features) if len(l)>min_size else to_terminal(l)
    if isinstance(node['left'],dict): do_split(node['left'],max_depth,min_size,n_features,depth+1)
    node['right'] = best_split(r, n_features) if len(r)>min_size else to_terminal(r)
    if isinstance(node['right'],dict): do_split(node['right'],max_depth,min_size,n_features,depth+1)

def build_tree(data, max_depth, min_size, n_features):
    root = best_split(data, n_features)
    do_split(root, max_depth, min_size, n_features, 1)
    return root

def predict_tree(node, row):
    key = 'left' if row[node['index']]<node['value'] else 'right'
    if isinstance(node[key],dict): return predict_tree(node[key],row)
    return node[key]

def random_forest(train, n_trees=10, max_depth=5, min_size=2, sample_ratio=0.7):
    n_features = max(1, int(math.sqrt(len(train[0])-1)))
    trees = []
    for _ in range(n_trees):
        sample = [random.choice(train) for _ in range(int(len(train)*sample_ratio))]
        trees.append(build_tree(sample, max_depth, min_size, n_features))
    return trees

def predict_rf(trees, row):
    preds = [predict_tree(t, row) for t in trees]
    return Counter(preds).most_common(1)[0][0]

def main():
    if "--demo" in sys.argv:
        random.seed(42)
        data = [[random.gauss(0,1),random.gauss(0,1),0] for _ in range(30)]
        data += [[random.gauss(3,1),random.gauss(3,1),1] for _ in range(30)]
        random.shuffle(data)
        train,test = data[:48],data[48:]
        trees = random_forest(train, n_trees=15)
        preds = [predict_rf(trees, r) for r in test]
        acc = sum(1 for p,r in zip(preds,test) if p==r[-1])/len(test)
        print(f"Random Forest: {len(trees)} trees, test accuracy: {acc:.1%}")
    else:
        d = json.loads(sys.stdin.read())
        trees = random_forest(d["train"], d.get("n_trees",10))
        preds = [predict_rf(trees, r) for r in d["test"]]
        print(json.dumps(preds))
if __name__=="__main__": main()
