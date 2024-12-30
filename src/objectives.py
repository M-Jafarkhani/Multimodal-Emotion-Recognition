import torch


def _criterioning(pred, truth, criterion):
    """Handle criterion ideosyncracies."""
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        truth = (
            truth.squeeze() if len(truth.shape) == len(pred.shape) else truth
        )
        return criterion(
            pred,
            truth.long().to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ),
        )
    if isinstance(
        criterion,
        (
            torch.nn.modules.loss.BCEWithLogitsLoss,
            torch.nn.MSELoss,
            torch.nn.L1Loss,
        ),
    ):
        return criterion(
            pred,
            truth.float().to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            ),
        )


def recon_weighted_sum(modal_loss_funcs, weights):
    """Create wrapper function that computes the weighted model reconstruction loss."""

    def _actualfunc(recons, origs):
        totalloss = 0.0
        for i in range(len(recons)):
            trg = (
                origs[i].view(recons[i].shape[0], recons[i].shape[1])
                if len(recons[i].shape) != len(origs[i].shape)
                else origs[i]
            )
            totalloss += modal_loss_funcs[i](recons[i], trg) * weights[i]
        return torch.mean(totalloss)

    return _actualfunc


def MFM_objective(
    ce_weight,
    modal_loss_funcs,
    recon_weights,
    criterion=torch.nn.CrossEntropyLoss(),
):
    """Define objective for MFM.

    :param ce_weight: the weight of simple supervised loss
    :param model_loss_funcs: list of functions that takes in reconstruction and input of each modality and compute reconstruction loss
    :param recon_weights: list of float values indicating the weight of reconstruction loss of each modality
    :param criterion: the loss function for supervised loss (default CrossEntropyLoss)
    """
    recon_loss_func = recon_weighted_sum(modal_loss_funcs, recon_weights)

    def _actualfunc(pred, truth, args):
        ints = args["intermediates"]
        reps = args["reps"]
        fused = args["fused"]
        decoders = args["decoders"]
        inps = args["inputs"]
        recons = []
        for i in range(len(reps)):
            recons.append(
                decoders[i](torch.cat([ints[i](reps[i]), fused], dim=1))
            )
        ce_loss = _criterioning(pred, truth, criterion)
        inputs = [
            i.float().to(
                torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
            )
            for i in inps
        ]
        recon_loss = recon_loss_func(recons, inputs)
        return ce_loss * ce_weight + recon_loss

    return _actualfunc
