import numpy as np
import sklearn.metrics as metrics


def cost_delay(k, break_point):
    # lc_o on the original erde equation.
    return 1 - (1 / (1 + np.exp(k - break_point)))


def erde_user(label, true_label, delay, _c_tp, _c_fn, _c_fp, _o):
    if label == 1 and true_label == 1:
        return cost_delay(k=delay, break_point=_o) * _c_tp
    elif label == 1 and true_label == 0:
        return _c_fp
    elif label == 0 and true_label == 1:
        return _c_fn
    elif label == 0 and true_label == 0:
        return 0


def erde_final(labels_list, true_labels_list, delay_list, c_fp, c_tp=1, c_fn=1, o=50):
    erde_list = [
        erde_user(
            label=l,
            true_label=true_labels_list[i],
            delay=delay_list[i],
            _c_tp=c_tp,
            _c_fn=c_fn,
            _c_fp=c_fp,
            _o=o,
        )
        for i, l in enumerate(labels_list)
    ]
    return np.mean(erde_list)


def value_p(k):
    return -(np.log(1 / 3) / (k - 1))


def f_penalty(k, _p):
    return -1 + (2 / (1 + np.exp((-_p) * (k - 1))))


def speed(y_pred, y_true, d, p):
    penalty_list = [
        f_penalty(k=d[i], _p=p)
        for i in range(len(y_pred))
        if y_pred[i] == 1 and y_true[i] == 1
    ]

    if len(penalty_list) != 0:
        return 1 - np.median(penalty_list)
    else:
        return 0.0


def f_latency(labels, true_labels, delays, penalty):
    f1_score = metrics.f1_score(y_pred=labels, y_true=true_labels, average="binary")
    speed_value = speed(y_pred=labels, y_true=true_labels, d=delays, p=penalty)

    return f1_score * speed_value


def precision_at_k(scores, y_true, k=10):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    idx = np.argsort(-scores)
    scores_sorted = scores[idx]
    y_true_sorted = y_true[idx]
    if len(scores_sorted) > k:
        y_true_sorted = y_true_sorted[:k]

    return np.sum(y_true_sorted) / k


def dcg(relevance, rank):
    relevance = np.asarray(relevance)[:rank]
    n_relevances = len(relevance)
    if n_relevances == 0:
        return 0.0

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum((2 ** relevance - 1) / discounts)


def ndcg(scores, y_true, p):
    best_dcg = dcg(relevance=sorted(y_true, reverse=True), rank=p)
    current_dcg = dcg(relevance=sorted(scores, reverse=True), rank=p)

    return current_dcg / best_dcg
