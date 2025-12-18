import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_metrics(preds, labels, num_classes):
    """
    Вычисляет метрики качества: accuracy, precision, recall, F1-score
    """
    accuracy = accuracy_score(labels, preds)
    
    cm = confusion_matrix(labels, preds, labels=np.arange(num_classes))
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def calculate_iou(preds, labels, num_classes):
    """
    Вычисляет Intersection over Union (IoU) для каждого класса
    """
    iou = {}
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        
        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()
        
        if union > 0:
            iou[cls] = intersection / union
        else:
            iou[cls] = None
    
    return iou


def calculate_mean_iou(preds, labels, num_classes):
    """
    Вычисляет средний IoU по всем классам
    """
    iou = calculate_iou(preds, labels, num_classes)
    valid_ious = [iou[i] for i in range(num_classes) if iou[i] is not None]
    
    if len(valid_ious) > 0:
        return np.mean(valid_ious)
    else:
        return 0.0

