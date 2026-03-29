#!/usr/bin/env python3
"""Random forest classifier from scratch."""
import sys, math, random, csv
from collections import Counter

def gini(labels):
    n = len(labels)
    if n == 0: return 0
    counts = Counter(labels)
    return 1 - sum((c/n)**2 for c in counts.values())

class TreeNode:
    def __init__(self): self.feature=None; self.threshold=None; self.left=None; self.right=None; self.label=None

def build_tree(X, y, max_depth=10, min_samples=2, max_features=None, depth=0):
    node = TreeNode(); node.label = Counter(y).most_common(1)[0][0]
    if depth >= max_depth or len(set(y)) == 1 or len(y) < min_samples: return node
    n_features = len(X[0])
    features = random.sample(range(n_features), min(max_features or n_features, n_features))
    best_gain, best_f, best_t = 0, None, None
    parent_gini = gini(y); n = len(y)
    for f in features:
        vals = sorted(set(row[f] for row in X))
        for i in range(len(vals)-1):
            t = (vals[i]+vals[i+1])/2
            ly = [y[j] for j in range(n) if X[j][f] <= t]
            ry = [y[j] for j in range(n) if X[j][f] > t]
            if not ly or not ry: continue
            gain = parent_gini - (len(ly)/n*gini(ly) + len(ry)/n*gini(ry))
            if gain > best_gain: best_gain=gain; best_f=f; best_t=t
    if best_f is None: return node
    node.feature=best_f; node.threshold=best_t
    li = [i for i in range(n) if X[i][best_f] <= best_t]
    ri = [i for i in range(n) if X[i][best_f] > best_t]
    node.left = build_tree([X[i] for i in li],[y[i] for i in li],max_depth,min_samples,max_features,depth+1)
    node.right = build_tree([X[i] for i in ri],[y[i] for i in ri],max_depth,min_samples,max_features,depth+1)
    return node

def tree_predict(node, x):
    if node.feature is None: return node.label
    return tree_predict(node.left if x[node.feature] <= node.threshold else node.right, x)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, max_features=None):
        self.n_trees=n_trees; self.max_depth=max_depth; self.max_features=max_features; self.trees=[]
    def fit(self, X, y):
        n = len(y); mf = self.max_features or max(1, int(math.sqrt(len(X[0]))))
        for _ in range(self.n_trees):
            idx = [random.randint(0, n-1) for _ in range(n)]
            bX = [X[i] for i in idx]; by = [y[i] for i in idx]
            self.trees.append(build_tree(bX, by, self.max_depth, max_features=mf))
    def predict(self, x):
        votes = Counter(tree_predict(t, x) for t in self.trees)
        return votes.most_common(1)[0][0]
    def score(self, X, y):
        return sum(1 for xi,yi in zip(X,y) if self.predict(xi)==yi) / len(y)

def main():
    random.seed(42)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        with open(sys.argv[1]) as f:
            reader=csv.reader(f); next(reader); data=list(reader)
        X=[[float(v) for v in r[:-1]] for r in data]; y=[r[-1] for r in data]
        rf=RandomForest(n_trees=20); rf.fit(X,y)
        print(f"Accuracy: {rf.score(X,y)*100:.1f}%"); return
    X=[[random.gauss(cx,1),random.gauss(cy,1)] for cx,cy in [(0,0),(3,3),(0,3)] for _ in range(40)]
    y=[c for c in ["A","B","C"] for _ in range(40)]
    idx=list(range(len(y))); random.shuffle(idx)
    X=[X[i] for i in idx]; y=[y[i] for i in idx]
    s=int(len(y)*0.8)
    for n in [1,5,10,20]:
        rf=RandomForest(n_trees=n); rf.fit(X[:s],y[:s])
        print(f"Trees={n:2d}: accuracy={rf.score(X[s:],y[s:])*100:.1f}%")

if __name__ == "__main__": main()
