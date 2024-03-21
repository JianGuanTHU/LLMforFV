import dataclasses
import enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

sns.set_theme(style='darkgrid', font_scale=1.1)


class Labels(enum.IntEnum):
  GROUNDED = 1
  UNGROUNDED = 0


@dataclasses.dataclass
class PrecisionRecallValues():
  grounded_precision: float
  grounded_recall: float
  grounded_f1: float
  ungrounded_precision: float
  ungrounded_recall: float
  ungrounded_f1: float


@dataclasses.dataclass
class MetaEvalValues():
  metric: List[str]
  accuracy: List[float]
  roc_auc: List[float]
  grounded_precision: List[float]
  grounded_recall: List[float]
  grounded_f1: List[float]
  ungrounded_precision: List[float]
  ungrounded_recall: List[float]
  ungrounded_f1: List[float]


def get_predicted_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
  return np.array([
      Labels.UNGROUNDED if score <= threshold else Labels.GROUNDED
      for score in scores
  ])


def get_optimal_gmeans_threshold(scores: np.ndarray,
                                 labels: np.ndarray) -> Tuple[float, float]:
  fpr, tpr, thresholds = metrics.roc_curve(
      y_true=(1 - labels), y_score=(1 - scores))
  gmeans = np.sqrt(tpr * (1 - fpr))
  max_gmeans_idx = np.argmax(gmeans)
  return (1 - thresholds[max_gmeans_idx], gmeans[max_gmeans_idx])


def compute_precision_recall_per_threshold(
    predictions: np.ndarray, labels: np.ndarray) -> PrecisionRecallValues:
  grounded_precision = metrics.precision_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.GROUNDED)
  grounded_recall = metrics.recall_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.GROUNDED)
  grounded_f1 = metrics.f1_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.GROUNDED)
  ungrounded_precision = metrics.precision_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.UNGROUNDED)
  ungrounded_recall = metrics.recall_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.UNGROUNDED)
  ungrounded_f1 = metrics.f1_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.UNGROUNDED)
  result = PrecisionRecallValues(
      grounded_precision=grounded_precision,
      grounded_recall=grounded_recall,
      grounded_f1=grounded_f1,
      ungrounded_precision=ungrounded_precision,
      ungrounded_recall=ungrounded_recall,
      ungrounded_f1=ungrounded_f1)
  return result


def evaluate_metrics(data: pd.DataFrame, scores_columns: List[str],
                     thresholds: List[float]) -> pd.DataFrame:
  accuracies, roc_aucs = [], []
  grounded_precision_vals, grounded_recall_vals = [], []
  ungrounded_precision_vals, ungrounded_recall_vals = [], []
  grounded_f1_vals, ungrounded_f1_vals = [], []
  labels = data['label'].to_numpy()

  for i, metric_name in enumerate(scores_columns):
    print(metric_name)
    scores = data[metric_name].to_numpy()
    predictions = get_predicted_labels(scores, thresholds[i])
    accuracy = metrics.accuracy_score(y_true=labels, y_pred=predictions)
    accuracies.append(accuracy)

    fpr, tpr, _ = metrics.roc_curve(y_true=labels, y_score=scores)
    roc_aucs.append(metrics.auc(fpr, tpr))

    precision_recall_vals = compute_precision_recall_per_threshold(
        predictions, labels)
    grounded_precision_vals.append(precision_recall_vals.grounded_precision)
    grounded_recall_vals.append(precision_recall_vals.grounded_recall)
    grounded_f1_vals.append(precision_recall_vals.grounded_f1)
    ungrounded_precision_vals.append(precision_recall_vals.ungrounded_precision)
    ungrounded_recall_vals.append(precision_recall_vals.ungrounded_recall)
    ungrounded_f1_vals.append(precision_recall_vals.ungrounded_f1)

  meta_eval_scores = MetaEvalValues(
      metric=scores_columns,
      accuracy=accuracies,
      roc_auc=roc_aucs,
      grounded_precision=grounded_precision_vals,
      grounded_recall=grounded_recall_vals,
      grounded_f1=grounded_f1_vals,
      ungrounded_precision=ungrounded_precision_vals,
      ungrounded_recall=ungrounded_recall_vals,
      ungrounded_f1=ungrounded_f1_vals)
  meta_eval_df = pd.DataFrame(dataclasses.asdict(meta_eval_scores))
  return meta_eval_df


def plot_precision_recall_comparison(data: pd.DataFrame,
                                     scores_columns: List[str],
                                     grounded_positive: bool,
                                     output_path: str) -> None:

  all_metrics_precision = np.array([])
  all_metrics_recall = np.array([])
  metrics_names = []

  labels = data['label'].to_numpy()

  for metric_name in scores_columns:
    scores = data[metric_name].to_numpy()

    if grounded_positive:
      pos_label = 1
    else:
      pos_label = 0
      scores = 1 - scores

    precision, recall, _ = metrics.precision_recall_curve(
        y_true=labels, probas_pred=scores, pos_label=pos_label)
    auc = metrics.auc(recall, precision)

    all_metrics_precision = np.append(all_metrics_precision, precision)
    all_metrics_recall = np.append(all_metrics_recall, recall)
    metrics_names.extend([f'{metric_name}, auc:{auc:.2f}'] * len(precision))

  plt.figure(figsize=(16, 9))
  precision_recall_values_df = pd.DataFrame({
      'Recall': all_metrics_recall,
      'Precision': all_metrics_precision,
      'Metric': metrics_names
  })

  plot_type = 'grounded' if grounded_positive else 'ungrounded'
  sns_plot = sns.lineplot(
      x='Recall', y='Precision', hue='Metric',
      data=precision_recall_values_df).set_title(
          f'Precision vs. Recall for detecting {plot_type} text')
  with open(output_path, 'wb') as f:
    sns_plot.figure.savefig(f)
    return


def plot_roc_comparison(data: pd.DataFrame, scores_columns: List[str],
                        grounded_positive: bool, output_path: str) -> None:
  all_metrics_tpr = np.array([])
  all_metrics_fpr = np.array([])
  metrics_names = []

  labels = data['label'].to_numpy()

  for metric_name in scores_columns:
    scores = data[metric_name].to_numpy()

    if grounded_positive:
      pos_label = 1
    else:
      pos_label = 0
      scores = 1 - scores

    fpr, tpr, _ = metrics.roc_curve(
        y_true=labels, y_score=scores, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    all_metrics_tpr = np.append(all_metrics_tpr, tpr)
    all_metrics_fpr = np.append(all_metrics_fpr, fpr)
    metrics_names.extend([f'{metric_name}, auc:{auc:.2f}'] * len(tpr))

  plt.figure(figsize=(16, 9))
  roc_values_df = pd.DataFrame({
      'FPR': all_metrics_fpr,
      'TPR': all_metrics_tpr,
      'Metric': metrics_names
  })

  plot_type = 'grounded' if grounded_positive else 'ungrounded'
  sns_plot = sns.lineplot(
      x='FPR', y='TPR', hue='Metric', data=roc_values_df).set_title(
          f'TPR vs. FPR for detecting {plot_type} text')
  sns_plot.figure.savefig(output_path)
  plt.close()
  return
