from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

def std(metric_score):
    return np.std(metric_score)
def acc(human_score, metric_score):
    acc_list = []
    for h, m in zip(human_score, metric_score):
        if abs(h - m) <= 0.5:
            acc_list.append(1.)
        else:
            acc_list.append(0.)
    return np.mean(acc_list)

def _safe_div(numerator, denominator):
    return np.array([n/d if d>0 else 0. for n, d in zip(numerator, denominator)])

def _ece_from_bins(bin_counts, bin_true_sum, bin_preds_sum):
    bin_accuracies = _safe_div(bin_true_sum, bin_counts)
    bin_confidences = _safe_div(bin_preds_sum, bin_counts)
    abs_bin_errors = np.abs(bin_accuracies - bin_confidences)
    bin_weights = bin_counts / np.sum(bin_counts)
    return np.sum(abs_bin_errors * bin_weights)

def expected_calibration_error(human_score, metric_score, nbins=20, savepath=None):
    bin_counts, bin_true_sum, bin_preds_sum = [0 for _ in range(nbins)], [0 for _ in range(nbins)], [0 for _ in range(nbins)]
    bin_width = 1. / nbins
    for h, m in zip(human_score, metric_score):
        bin_ids = int(m / bin_width) if m < 1.0 else nbins-1
        bin_counts[bin_ids] += 1
        bin_true_sum[bin_ids] += h
        bin_preds_sum[bin_ids] += m
    bin_counts, bin_true_sum, bin_preds_sum = np.array(bin_counts), np.array(bin_true_sum), np.array(bin_preds_sum)
    ece = _ece_from_bins(bin_counts=bin_counts, bin_true_sum=bin_true_sum, bin_preds_sum=bin_preds_sum)
    avg_confidence = np.mean(metric_score)
    plt.figure()
    x = bin_width * np.arange(nbins) + bin_width / 2.
    plt.bar(x, bin_counts/np.sum(bin_counts), width=bin_width)
    plt.plot([avg_confidence for _ in range(11)], [0.1*d for d in range(11)], "r--", label="Avg. Confidence")
    avg_acc = np.sum(bin_true_sum) / np.sum(bin_counts)
    plt.plot([avg_acc for _ in range(11)], [0.1*d for d in range(11)], "b--", label="Accuracy")
    plt.legend()
    plt.xlabel("Confidence")
    plt.ylabel("\% of Examples")
    plt.title("Number Plot")
    plt.savefig(savepath.replace(".pdf", "_number.pdf"))

    plt.figure()
    x = bin_width * np.arange(nbins) + bin_width / 2.
    plt.bar(x, _safe_div(bin_true_sum, bin_counts), width=bin_width)
    plt.plot(x, x, "r--")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("ECE Plot")
    plt.savefig(savepath)
    return ece
    
def brier_score(human_score, metric_score):
    return brier_score_loss(human_score, metric_score)

def correlation(human_score, metric_score):
    return {
        "Pearson's Correlation": pearsonr(human_score, metric_score),
        "Spearman's Correlation": spearmanr(human_score, metric_score),
        "Kendall's Correlation": kendalltau(human_score, metric_score),
    }