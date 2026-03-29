#!/usr/bin/env python3
"""Random forest classifier from scratch."""
import sys, math, random
from collections import Counter

def entropy(labels):
    n = len(labels)
    if n == 0: return 0
    counts = Counter(labels)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())

class TreeNode:
    def __init__(self, feat=None, thresh=None, left=None, right=None, label=None):
        self.feat, self.thresh, self.left, self.right, self.label = feat, thresh, left, right, label

def build_tree(X, y, max_depth, n_features):
    if max_depth == 0 or len(set(y)) == 1 or len(y) < 2:
        return TreeNode(label=Counter(y).most_common(1)[0][0])
    d = len(X[0])
    feats = random.sample(range(d), min(n_features, d))
    best_g, best_f, best_t = 0, 0, 0
    for f in feats:
        vals = sorted(set(x[f] for x in X))
        for i in range(len(vals)-1):
            t = (vals[i]+vals[i+1])/2
            left_y = [y[k] for k in range(len(y)) if X[k][f] <= t]
            right_y = [y[k] for k in range(len(y)) if X[k][f] > t]
            if not left_y or not right_y: continue
            g = entropy(y) - len(left_y)/len(y)*entropy(left_y) - len(right_y)/len(y)*entropy(right_y)
            if g > best_g: best_g, best_f, best_t = g, f, t
    if best_g == 0:
        return TreeNode(label=Counter(y).most_common(1)[0][0])
    li = [k for k in range(len(y)) if X[k][best_f] <= best_t]
    ri = [k for k in range(len(y)) if X[k][best_f] > best_t]
    return TreeNode(best_f, best_t,
                    build_tree([X[k] for k in li], [y[k] for k in li], max_depth-1, n_features),
                    build_tree([X[k] for k in ri], [y[k] for k in ri], max_depth-1, n_features))

def predict_tree(node, x):
    if node.label is not None: return node.label
    return predict_tree(node.left if x[node.feat] <= node.thresh else node.right, x)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, n_features=None):
        self.n_trees, self.max_depth, self.n_features = n_trees, max_depth, n_features
        self.trees = []
    def fit(self, X, y):
        n = len(X)
        nf = self.n_features or max(1, int(math.sqrt(len(X[0]))))
        for _ in range(self.n_trees):
            idx = [random.randint(0, n-1) for _ in range(n)]
            self.trees.append(build_tree([X[i] for i in idx], [y[i] for i in idx], self.max_depth, nf))
    def predict(self, x):
        votes = [predict_tree(t, x) for t in self.trees]
        return Counter(votes).most_common(1)[0][0]

def test():
    random.seed(42)
    X = [[random.gauss(0,1), random.gauss(0,1)] for _ in range(30)] +         [[random.gauss(3,1), random.gauss(3,1)] for _ in range(30)]
    y = ["A"]*30 + ["B"]*30
    rf = RandomForest(n_trees=10, max_depth=4)
    rf.fit(X, y)
    correct = sum(1 for i in range(60) if rf.predict(X[i]) == y[i])
    assert correct >= 45, f"Accuracy: {correct}/60"
    print("  random_forest: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Random forest classifier")
