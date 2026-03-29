#!/usr/bin/env python3
"""Random forest — ensemble of decision stumps."""
import math, random, sys
from collections import Counter

class Stump:
    def __init__(self): self.feat = 0; self.threshold = 0; self.left_class = 0; self.right_class = 0
    def fit(self, X, y, features=None):
        features = features or list(range(len(X[0])))
        best_gini = float('inf')
        for f in features:
            vals = sorted(set(row[f] for row in X))
            for v in vals:
                left = [y[i] for i in range(len(X)) if X[i][f] <= v]
                right = [y[i] for i in range(len(X)) if X[i][f] > v]
                if not left or not right: continue
                gini = len(left)/len(y)*self._gini(left) + len(right)/len(y)*self._gini(right)
                if gini < best_gini:
                    best_gini = gini; self.feat = f; self.threshold = v
                    self.left_class = Counter(left).most_common(1)[0][0]
                    self.right_class = Counter(right).most_common(1)[0][0]
    def _gini(self, labels):
        n = len(labels); counts = Counter(labels)
        return 1 - sum((c/n)**2 for c in counts.values())
    def predict(self, x):
        return self.left_class if x[self.feat] <= self.threshold else self.right_class

class RandomForest:
    def __init__(self, n_trees=10, max_features=None):
        self.n_trees = n_trees; self.max_features = max_features; self.trees = []
    def fit(self, X, y):
        n, d = len(X), len(X[0])
        mf = self.max_features or max(1, int(math.sqrt(d)))
        for _ in range(self.n_trees):
            idx = [random.randint(0, n-1) for _ in range(n)]
            Xb, yb = [X[i] for i in idx], [y[i] for i in idx]
            feats = random.sample(range(d), min(mf, d))
            stump = Stump(); stump.fit(Xb, yb, feats); self.trees.append(stump)
    def predict(self, x):
        votes = [t.predict(x) for t in self.trees]
        return Counter(votes).most_common(1)[0][0]
    def accuracy(self, X, y):
        return sum(self.predict(x)==yi for x,yi in zip(X,y))/len(y)

if __name__ == "__main__":
    random.seed(42); X, y = [], []
    for _ in range(200):
        x = [random.uniform(0,10) for _ in range(4)]
        X.append(x); y.append("A" if sum(x[:2]) > sum(x[2:]) else "B")
    rf = RandomForest(n_trees=20); rf.fit(X[:160], y[:160])
    print(f"Random Forest (20 trees): accuracy={rf.accuracy(X[160:],y[160:])*100:.0f}%")
