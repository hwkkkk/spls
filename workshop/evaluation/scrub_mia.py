import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def entropy(p, dim=-1, keepdim=False, eps=1e-30):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model, num_classes=10):
    if data_loader is None:
        device = next(model.parameters()).device
        return (
            torch.zeros([0, num_classes], device=device),
            torch.zeros([0], dtype=torch.long, device=device),
        )

    prob, targets = [], []

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in data_loader:
            try:
                data, target = batch
                data, target = data.to(device), target.to(device)
            except Exception:
                data, target = get_x_y_from_data_dict(batch, device)

            output = model(data)
            prob.append(F.softmax(output, dim=-1))
            targets.append(target)

    return torch.cat(prob, dim=0), torch.cat(targets, dim=0)


def get_scrub_mia_data(retain_loader, forget_loader, test_loader, model):
    """
    SCRUB-style MIA:
    - attack train:
        retain -> member(1)
        test   -> non-member(0)
    - attack eval:
        forget only
    """
    retain_prob, _ = collect_prob(retain_loader, model)
    forget_prob, _ = collect_prob(forget_loader, model)
    test_prob, _ = collect_prob(test_loader, model)

    retain_ent = entropy(retain_prob).detach().cpu().numpy().reshape(-1, 1)
    forget_ent = entropy(forget_prob).detach().cpu().numpy().reshape(-1, 1)
    test_ent = entropy(test_prob).detach().cpu().numpy().reshape(-1, 1)

    # Attack model training data
    X_attack = np.concatenate([retain_ent, test_ent], axis=0)
    y_attack = np.concatenate([
        np.ones(len(retain_ent), dtype=np.int64),
        np.zeros(len(test_ent), dtype=np.int64)
    ])

    # Forget-only evaluation
    X_forget = forget_ent
    forget_mean_entropy = float(forget_ent.mean()) if len(forget_ent) > 0 else float("nan")
    retain_mean_entropy = float(retain_ent.mean()) if len(retain_ent) > 0 else float("nan")
    test_mean_entropy = float(test_ent.mean()) if len(test_ent) > 0 else float("nan")

    return X_attack, y_attack, X_forget, forget_mean_entropy, retain_mean_entropy, test_mean_entropy


def scrub_mia(retain_loader, forget_loader, test_loader, model):
    copy_model = copy.deepcopy(model)

    X_attack, y_attack, X_forget, forget_mean_entropy, retain_mean_entropy, test_mean_entropy = get_scrub_mia_data(
        retain_loader, forget_loader, test_loader, copy_model
    )

    # clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf.fit(X_attack, y_attack)

    # 1. Forget 평가
    pred_forget = clf.predict(X_forget)
    forget_member_ratio = float(pred_forget.mean())
    
    return {
        "forget_member_ratio": forget_member_ratio,  # Target: test_fpr과 같아져야 함
    }
