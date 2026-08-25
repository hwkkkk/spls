import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def entropy(p, dim=-1, keepdim=False, eps=1e-30):
    # 안정화: log(0) 방지
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False, eps=1e-30):
    """
    Modified entropy:
    - for the true label y: use (1 - p_y) instead of p_y
    - and swap the corresponding log terms consistently
    """
    p = p.clamp_min(eps)
    log_p = p.log()

    rp = (1.0 - p).clamp_min(eps)          # reverse_prob
    log_rp = rp.log()                      # log_reverse_prob  (FIXED)

    # row-wise index
    idx = torch.arange(p.size(0), device=p.device)

    modified_probs = p.clone()
    modified_probs[idx, labels] = rp[idx, labels]             # (FIXED indexing)

    modified_log_probs = log_p.clone()
    modified_log_probs[idx, labels] = log_rp[idx, labels]     # swap log term (FIXED)

    return -(modified_probs * modified_log_probs).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model, num_classes=10):
    if data_loader is None:
        return torch.zeros([0, num_classes]), torch.zeros([0], dtype=torch.long)

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
            prob.append(F.softmax(output, dim=-1))   # no .data (FIXED)
            targets.append(target)

    return torch.cat(prob, dim=0), torch.cat(targets, dim=0)

def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test  = shadow_test.shape[0]

    # Shadow -> train membership classifier
    X_shadow = torch.cat([shadow_train, shadow_test]).detach().cpu().numpy()
    X_shadow = X_shadow.reshape(n_shadow_train + n_shadow_test, -1)
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    out = {}

    # target_train: ratio predicted as member(1)
    if target_train is not None and target_train.shape[0] > 0:
        X_target_train = target_train.detach().cpu().numpy().reshape(target_train.shape[0], -1)
        pred_train = clf.predict(X_target_train)  # 0/1
        out["train_member_ratio"] = float(pred_train.mean())
    else:
        out["train_member_ratio"] = float("nan")

    # target_test: ratio predicted as non-member(0)
    if target_test is not None and target_test.shape[0] > 0:
        X_target_test = target_test.detach().cpu().numpy().reshape(target_test.shape[0], -1)
        pred_test = clf.predict(X_target_test)  # 0/1
        out["test_nonmember_ratio"] = float((1 - pred_test).mean())
    else:
        out["test_nonmember_ratio"] = float("nan")

    # (선택) 평균이 필요하면 따로 명시적으로 계산
    vals = []
    if not np.isnan(out["train_member_ratio"]):
        vals.append(out["train_member_ratio"])
    if not np.isnan(out["test_nonmember_ratio"]):
        vals.append(out["test_nonmember_ratio"])
    out["avg"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")

    return out

def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob,  shadow_test_labels  = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob,  target_test_labels  = collect_prob(target_test, model)

    # shadow_train_corr = (shadow_train_prob.argmax(dim=1) == shadow_train_labels).int()
    # shadow_test_corr  = (shadow_test_prob.argmax(dim=1)  == shadow_test_labels).int()
    # target_train_corr = (target_train_prob.argmax(dim=1) == target_train_labels).int()
    # target_test_corr  = (target_test_prob.argmax(dim=1)  == target_test_labels).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf  = torch.gather(shadow_test_prob,  1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf  = torch.gather(target_test_prob,  1, target_test_labels[:, None])

    # shadow_train_entr = entropy(shadow_train_prob)
    # shadow_test_entr  = entropy(shadow_test_prob)
    # target_train_entr = entropy(target_train_prob)
    # target_test_entr  = entropy(target_test_prob)

    # shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    # shadow_test_m_entr  = m_entropy(shadow_test_prob,  shadow_test_labels)
    # target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    # target_test_m_entr  = m_entropy(target_test_prob,  target_test_labels)

    import time
    results = {}

    # t0 = time.time()
    # results["correctness"] = SVC_fit_predict(shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr)
    # print("acc_corr time:", time.time() - t0, results["correctness"])

    t0 = time.time()
    results["confidence"] = SVC_fit_predict(shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf)
    print("acc_conf time:", time.time() - t0, results["confidence"])

    # t0 = time.time()
    # results["entropy"] = SVC_fit_predict(shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr)
    # print("acc_entr time:", time.time() - t0, results["entropy"])

    # t0 = time.time()
    # results["m_entropy"] = SVC_fit_predict(shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr)
    # print("acc_m_entr time:", time.time() - t0, results["m_entropy"])

    # t0 = time.time()
    # results["prob"] = SVC_fit_predict(shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob)
    # print("acc_prob time:", time.time() - t0, results["prob"])

    return results