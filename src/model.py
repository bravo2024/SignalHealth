
"""model.py - FFT-feature signal classifier (NumPy logistic; prod: 1D-CNN in torch)."""
import numpy as np
from src.core import Standardizer,LogisticRegression,train_test_split,roc_auc_score,accuracy_score
PREDICT_KIND="signal"
def _feat(X):
    F=np.abs(np.fft.rfft(np.asarray(X,float),axis=1))
    return np.hstack([F,X.mean(1,keepdims=True),X.std(1,keepdims=True)])
def fit_and_evaluate(data):
    X=_feat(data["X"]); y=np.asarray(data["y"],int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,0.25,7); sc=Standardizer().fit(Xtr)
    clf=LogisticRegression(lr=0.2,epochs=400).fit(sc.transform(Xtr),ytr)
    proba=clf.predict_proba(sc.transform(Xte)); pred=(proba>=0.5).astype(int)
    return {"scaler":sc,"clf":clf},{"n_train":int(len(Xtr)),"n_test":int(len(Xte)),"roc_auc":roc_auc_score(yte,proba),"accuracy":accuracy_score(yte,pred)}
