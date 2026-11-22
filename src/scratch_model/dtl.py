import json

import numpy as np
import pandas as pd

class _TreeNode:
    def __init__(self, class_weights, n_samples, depth, impurity):
        self.class_weights = class_weights          
        self.weight = float(class_weights.sum())    
        self.n_samples = int(n_samples)             
        self.depth = depth
        self.impurity = float(impurity)
        self.feature = None                         
        self.threshold = None                       
        self.children = {}                          
        self.branch_weights = {}                    
        self.gain = 0.0                             

    @property
    def is_leaf(self):
        return self.feature is None

    @property
    def proba(self):
        return self.class_weights / self.weight

    def make_leaf(self):
        self.feature = None
        self.threshold = None
        self.children = {}
        self.branch_weights = {}
        self.gain = 0.0


class DecisionTreeClassifier:
    def __init__(self, criterion="gain_ratio", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain_ratio=1e-4, max_features=None,
                 feature_names=None, random_state=None, ccp_alpha=0.0,
                 class_weight=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.max_features = max_features
        self.feature_names = feature_names
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight

    _PARAM_NAMES = ("criterion", "max_depth", "min_samples_split", "min_samples_leaf",
                    "min_gain_ratio", "max_features", "feature_names", "random_state",
                    "ccp_alpha", "class_weight")

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self._PARAM_NAMES:
                raise ValueError(f"Parameter '{key}' tidak valid untuk DecisionTreeClassifier.")
            setattr(self, key, value)
        return self

    def _validate_params(self):
        if self.criterion not in ("gain_ratio", "entropy", "gini"):
            raise ValueError("criterion harus 'gain_ratio', 'entropy', atau 'gini'.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth harus >= 1 atau None.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split harus >= 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf harus >= 1.")
        if self.ccp_alpha is not None and self.ccp_alpha < 0:
            raise ValueError("ccp_alpha harus >= 0.")

    def _check_fitted(self):
        if not hasattr(self, "root_"):
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")
    
    def _check_fitted(self):
        if not hasattr(self, "root_"):
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")

    def _frame(self, X, fitting):
        if isinstance(X, pd.DataFrame):
            df = X
            if not fitting:
                names = list(self.feature_names_)
                cols = [str(c) for c in df.columns]
                if set(names).issubset(cols):
                    df = df.iloc[:, [cols.index(n) for n in names]]
                elif df.shape[1] != self.n_features_in_:
                    raise ValueError(f"Jumlah fitur ({df.shape[1]}) != saat fit ({self.n_features_in_}).")
            return df
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X harus 2 dimensi.")
        if not fitting and arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Jumlah fitur ({arr.shape[1]}) != saat fit ({self.n_features_in_}).")
        df = pd.DataFrame(arr)
        if fitting and arr.dtype == object:
            for j in df.columns:
                try:
                    df[j] = pd.to_numeric(df[j])
                except (ValueError, TypeError):
                    pass
        return df

    def _setup_features(self, df):
        self.n_features_in_ = df.shape[1]
        if isinstance(df.columns, pd.RangeIndex):
            default = [f"x{i}" for i in range(df.shape[1])]
            names = list(self.feature_names) if self.feature_names is not None else default
        else:
            names = [str(c) for c in df.columns]
        if len(names) != self.n_features_in_:
            raise ValueError("feature_names tidak sesuai jumlah kolom X.")
        self.feature_names_ = names

        self.is_numeric_feature_, self.categories_, self._cat_maps = [], [], []
        for j in range(self.n_features_in_):
            s = df.iloc[:, j]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                self.is_numeric_feature_.append(True)
                self.categories_.append(None)
                self._cat_maps.append(None)
            else:
                values = list(pd.unique(s.dropna()))
                try:
                    values = sorted(values)
                except TypeError:
                    pass
                self.is_numeric_feature_.append(False)
                self.categories_.append(values)
                self._cat_maps.append({v: i for i, v in enumerate(values)})

    def _encode_column(self, s, j):
        if self.is_numeric_feature_[j]:
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        mapped = pd.to_numeric(s.astype(object).map(self._cat_maps[j]), errors="coerce")
        codes = mapped.to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isnan(codes), -1, codes).astype(np.int64)

    def _sample_weights(self, sample_weight, y_idx):
        n = len(y_idx)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float).copy()
        if len(w) != n:
            raise ValueError("sample_weight harus sepanjang y.")
        if (w < 0).any():
            raise ValueError("sample_weight tidak boleh negatif.")
        if self.class_weight is not None:
            counts = np.bincount(y_idx, minlength=self._n_classes)
            if isinstance(self.class_weight, str):
                if self.class_weight != "balanced":
                    raise ValueError("class_weight string hanya mendukung 'balanced'.")
                cw = n / (self._n_classes * np.maximum(counts, 1))
            else:
                cw = np.array([self.class_weight.get(c, self.class_weight.get(c.item() if hasattr(c, "item") else c, 1.0))
                               for c in self.classes_], dtype=float)
            w = w * cw[y_idx]
        return w

    def _resolve_max_features(self):
        n, mf = self.n_features_in_, self.max_features
        if mf is None:
            return n
        if mf == "sqrt":
            return max(1, int(np.sqrt(n)))
        if mf == "log2":
            return max(1, int(np.log2(n)))
        if isinstance(mf, (int, np.integer)):
            return int(min(max(mf, 1), n))
        if isinstance(mf, float):
            return int(min(max(1, int(mf * n)), n))
        raise ValueError("max_features tidak valid.")

    def _impurity(self, counts):
        counts = np.asarray(counts, dtype=float)
        total = counts.sum(axis=-1, keepdims=True)
        p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
        if self.criterion == "gini":
            return 1.0 - np.sum(p * p, axis=-1)
        logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
        return -np.sum(p * logp, axis=-1)

    @staticmethod
    def _entropy_of_probs(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _best_split(self, idx, w):
        W = w.sum()
        n_feat = self.n_features_in_
        if self._n_candidate_features >= n_feat:
            feats = range(n_feat)
        else:
            feats = self._rng.choice(n_feat, self._n_candidate_features, replace=False)

        cands = []
        for f in feats:
            if self.is_numeric_feature_[f]:
                res = self._eval_numeric(f, idx, w, W)
            else:
                res = self._eval_categorical(f, idx, w, W)
            if res is not None:
                cands.append(res)
        if not cands:
            return None

        if self.criterion == "gain_ratio":
            mean_gain = np.mean([c["gain"] for c in cands])
            eligible = [c for c in cands if c["gain"] >= mean_gain - 1e-12]
            best = max(eligible, key=lambda c: c["ratio"])
            score = best["ratio"]
        else:
            best = max(cands, key=lambda c: c["gain"])
            score = best["gain"]
        if best["gain"] <= 1e-12 or score < self.min_gain_ratio:
            return None
        return best

    def _eval_numeric(self, f, idx, w, W):
        v = self._cols[f][idx]
        known = ~np.isnan(v)
        n_known = int(known.sum())
        if n_known < 2 * self._min_leaf:
            return None
        vk, yk, wk = v[known], self._y[idx][known], w[known]
        Wk = wk.sum()
        if Wk <= 0:
            return None

        order = np.argsort(vk, kind="mergesort")
        vs = vk[order]
        pos = np.nonzero(vs[:-1] < vs[1:])[0]             
        pos = pos[((pos + 1) >= self._min_leaf) & ((n_known - pos - 1) >= self._min_leaf)]
        if len(pos) == 0:
            return None

        onehot = np.zeros((n_known, self._n_classes))
        onehot[np.arange(n_known), yk[order]] = wk[order]
        cum = np.cumsum(onehot, axis=0)
        total = cum[-1]
        left = cum[pos]
        right = np.maximum(total - left, 0.0)
        wl, wr = left.sum(axis=1), right.sum(axis=1)

        child_imp = (wl * self._impurity(left) + wr * self._impurity(right)) / Wk
        gains = self._impurity(total) - child_imp
        b = int(np.argmax(gains))
        gain = (Wk / W) * float(gains[b])

        i = pos[b]
        thr = (vs[i] + vs[i + 1]) / 2.0
        if not (vs[i] <= thr < vs[i + 1]):
            thr = vs[i]

        split_info = self._entropy_of_probs([wl[b] / W, wr[b] / W, (W - Wk) / W])
        ratio = gain / split_info if split_info > 1e-12 else 0.0
        return {"feature": f, "gain": gain, "ratio": ratio, "threshold": float(thr)}

    def _eval_categorical(self, f, idx, w, W):
        c = self._cols[f][idx]
        known = c >= 0
        if not known.any():
            return None
        cc, yk, wk = c[known], self._y[idx][known], w[known]
        Wk = wk.sum()
        if Wk <= 0:
            return None

        n_cat = len(self.categories_[f])
        rows = np.bincount(cc, minlength=n_cat)
        present = np.nonzero(rows > 0)[0]
        if len(present) < 2 or (rows[present] < self._min_leaf).any():
            return None

        K = self._n_classes
        M = np.bincount(cc * K + yk, weights=wk, minlength=n_cat * K).reshape(n_cat, K)[present]
        wc = M.sum(axis=1)
        gain_k = float(self._impurity(M.sum(axis=0)) - np.sum(wc * self._impurity(M)) / Wk)
        gain = (Wk / W) * gain_k

        split_info = self._entropy_of_probs(np.append(wc / W, (W - Wk) / W))
        ratio = gain / split_info if split_info > 1e-12 else 0.0
        return {"feature": f, "gain": gain, "ratio": ratio, "categories": present}

    def _descend(self, node, idx, wts, cols, out):
        if len(idx) == 0:
            return
        if node.is_leaf:
            out[idx] += wts[:, None] * node.proba
            return
        col = cols[node.feature][idx]
        if self.is_numeric_feature_[node.feature]:
            miss = np.isnan(col)
            masks = {0: ~miss & (col <= node.threshold), 1: ~miss & (col > node.threshold)}
        else:
            keys = np.fromiter(node.children.keys(), dtype=np.int64)
            miss = ~np.isin(col, keys)
            masks = {k: col == k for k in node.children}
        for key, child in node.children.items():
            m = masks[key]
            p = node.branch_weights[key]
            self._descend(child,
                          np.concatenate([idx[m], idx[miss]]),
                          np.concatenate([wts[m], wts[miss] * p]),
                          cols, out)