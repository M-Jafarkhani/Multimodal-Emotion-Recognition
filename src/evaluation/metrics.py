import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth.

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def eval_mosei_senti_return(results, truths, exclude_zero=False):
    """Evaluate MOSEI and return metric list.

    Args:
        results (np.array): List of predicated values.
        truths (np.array): List of true values.
        exclude_zero (bool, optional): Whether to exclute zero. Defaults to False.

    Returns:
        tuple(mae, corr, mult_a7, f_score, accuracy): Return statistics for MOSEI.
    """
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    f_score = f1_score(
        (test_preds[non_zeros] > 0),
        (test_truth[non_zeros] > 0),
        average="weighted",
    )
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0

    return (
        mae,
        corr,
        mult_a7,
        f_score,
        accuracy_score(binary_truth, binary_preds),
    )
