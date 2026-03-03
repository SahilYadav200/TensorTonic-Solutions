import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using the trapezoidal rule.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    # Calculate (FPR_{i+1} - FPR_i)
    # np.diff computes the difference between adjacent elements in the array
    d_fpr = np.diff(fpr)
    
    # Calculate (TPR_i + TPR_{i+1})
    # We slice the array to add the current elements to the next elements
    sum_tpr = tpr[:-1] + tpr[1:]
    
    # Apply the trapezoidal formula: Sum( 0.5 * (TPR_i + TPR_{i+1}) * (FPR_{i+1} - FPR_i) )
    area = np.sum(0.5 * sum_tpr * d_fpr)
    
    return area