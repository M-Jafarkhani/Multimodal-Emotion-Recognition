import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from evaluation.performance import AUPRC
from evaluation.performance import eval_affect
from evaluation.complexity import all_in_one_train, all_in_one_test

softmax = nn.Softmax()


def train(
    encoder,
    head,
    train_dataloader,
    valid_dataloader,
    total_epochs,
    early_stop=False,
    optimtype=torch.optim.RMSprop,
    lr=0.001,
    weight_decay=0.0,
    criterion=nn.CrossEntropyLoss(),
    auprc=False,
    save_encoder="encoder.pt",
    save_head="head.pt",
    modalnum=0,
    task="classification",
    track_complexity=True,
):
    """Train unimodal module.

    Args:
        encoder (nn.Module): Unimodal encodder for the modality
        head (nn.Module): Takes in the unimodal encoder output and produces the final prediction.
        train_dataloader (torch.utils.data.DataLoader): Training data dataloader
        valid_dataloader (torch.utils.data.DataLoader): Validation set dataloader
        total_epochs (int): Total number of epochs
        early_stop (bool, optional): Whether to apply early-stopping or not. Defaults to False.
        optimtype (torch.optim.Optimizer, optional): Type of optimizer to use. Defaults to torch.optim.RMSprop.
        lr (float, optional): Learning rate. Defaults to 0.001.
        weight_decay (float, optional): Weight decay of optimizer. Defaults to 0.0.
        criterion (nn.Module, optional): Loss module. Defaults to nn.CrossEntropyLoss().
        auprc (bool, optional): Whether to compute AUPRC score or not. Defaults to False.
        save_encoder (str, optional): Path of file to save model with best validation performance. Defaults to 'encoder.pt'.
        save_head (str, optional): Path fo file to save head with best validation performance. Defaults to 'head.pt'.
        modalnum (int, optional): Which modality to apply encoder to. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        track_complexity (bool, optional): Whether to track the model's complexity or not. Defaults to True.
    """
    train_losses = []
    valid_losses = []

    def _trainprocess():
        model = nn.Sequential(encoder, head)
        op = optimtype(model.parameters(), lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0
        for epoch in range(total_epochs):
            print(f"Epoch [{epoch + 1}/{total_epochs}]")
            totalloss = 0.0
            totals = 0
            for j in train_dataloader:
                op.zero_grad()
                out = model(
                    j[modalnum]
                    .float()
                    .to(
                        torch.device(
                            "cuda:0" if torch.cuda.is_available() else "cpu"
                        )
                    )
                )

                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss = criterion(
                        out,
                        j[-1]
                        .float()
                        .to(
                            torch.device(
                                "cuda:0" if torch.cuda.is_available() else "cpu"
                            )
                        ),
                    )
                else:
                    loss = criterion(
                        out,
                        j[-1].to(
                            torch.device(
                                "cuda:0" if torch.cuda.is_available() else "cpu"
                            )
                        ),
                    )
                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
                op.step()
            print(f"Train Loss: {totalloss.item()/totals:.4f}")
            train_losses.append(totalloss.item() / totals)
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    out = model(
                        j[modalnum]
                        .float()
                        .to(
                            torch.device(
                                "cuda:0" if torch.cuda.is_available() else "cpu"
                            )
                        )
                    )
                    if (
                        type(criterion)
                        == torch.nn.modules.loss.BCEWithLogitsLoss
                    ):
                        loss = criterion(
                            out,
                            j[-1]
                            .float()
                            .to(
                                torch.device(
                                    "cuda:0"
                                    if torch.cuda.is_available()
                                    else "cpu"
                                )
                            ),
                        )
                    else:
                        loss = criterion(
                            out,
                            j[-1].to(
                                torch.device(
                                    "cuda:0"
                                    if torch.cuda.is_available()
                                    else "cpu"
                                )
                            ),
                        )
                    totalloss += loss * len(j[-1])
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        sm = softmax(out)
                        pts += [
                            (sm[i][1].item(), j[-1][i].item())
                            for i in range(j[-1].size(0))
                        ]
            if pred:
                pred = torch.cat(pred, 0).cpu().numpy()
            true = torch.cat(true, 0).cpu().numpy()
            totals = true.shape[0]
            valloss = totalloss / totals
            valid_losses.append(valloss.item())
            if task == "classification":
                acc = accuracy_score(true, pred)
                print(
                    f"Valid Loss: {valloss.item():.4f} | Accuracy: {acc*100:.4f}%"
                )
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print(
                    f"Valid Loss: {valloss.item():.4f} | f1_micro: {f1_micro:.4f} | f1_macro: {f1_macro:.4f}"
                )
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            elif task == "regression":
                print(f"Valid Loss: {valloss.item():.4f}")
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(encoder, save_encoder)
                    torch.save(head, save_head)
                else:
                    patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: " + str(AUPRC(pts)))
            print("")

    if track_complexity:
        all_in_one_train(_trainprocess, [encoder, head])
    else:
        _trainprocess()
    return train_losses, valid_losses


def single_test(
    encoder,
    head,
    test_dataloader,
    auprc=False,
    modalnum=0,
    task="classification",
    criterion=None,
):
    """Test unimodal model on one dataloader.

    Args:
        encoder (nn.Module): Unimodal encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloader (torch.utils.data.DataLoader): Data Loader for test set.
        auprc (bool, optional): Whether to output AUPRC or not. Defaults to False.
        modalnum (int, optional): Index of modality to consider for the test with the given encoder. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.

    Returns:
        dict: Dictionary of (metric, value) relations.
    """
    model = nn.Sequential(encoder, head)
    with torch.no_grad():
        pred = []
        true = []
        totalloss = 0
        pts = []
        for j in test_dataloader:
            out = model(
                j[modalnum]
                .float()
                .to(
                    torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )
                )
            )
            if criterion is not None:
                loss = criterion(
                    out,
                    j[-1].to(
                        torch.device(
                            "cuda:0" if torch.cuda.is_available() else "cpu"
                        )
                    ),
                )
                totalloss += loss * len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                sm = softmax(out)
                pts += [
                    (sm[i][1].item(), j[-1][i].item())
                    for i in range(j[-1].size(0))
                ]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        if auprc:
            print("AUPRC: " + str(AUPRC(pts)))
        if criterion is not None:
            print(f"Loss: {(totalloss/totals).item():.4f}")
        if task == "classification":
            print(f"Accuracy: {accuracy_score(true, pred):.4f}%")
            return {"Accuracy": accuracy_score(true, pred)}
        elif task == "multilabel":
            print(
                " f1_micro: "
                + str(f1_score(true, pred, average="micro"))
                + " f1_macro: "
                + str(f1_score(true, pred, average="macro"))
            )
            return {
                "F1 score (micro)": f1_score(true, pred, average="micro"),
                "F1 score (macro)": f1_score(true, pred, average="macro"),
            }
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print(f"Recall: {accs*100:.4f}% | Total Accuracy: {acc2*100:.4f}%")
            return {"Accuracy": accs}
        else:
            return {"MSE": (totalloss / totals).item()}


def test(
    encoder,
    head,
    test_dataloaders_all,
    auprc=False,
    modalnum=0,
    task="classification",
    criterion=None,
):
    """Test unimodal model on all provided dataloaders.

    Args:
        encoder (nn.Module): Encoder module
        head (nn.Module): Module which takes in encoded unimodal input and predicts output.
        test_dataloaders_all (dict): Dictionary of noisetype, dataloader to test.
        auprc (bool, optional): Whether to output AUPRC scores or not. Defaults to False.
        modalnum (int, optional): Index of modality to test on. Defaults to 0.
        task (str, optional): Type of task to try. Supports "classification", "regression", or "multilabel". Defaults to 'classification'.
        criterion (nn.Module, optional): Loss module. Defaults to None.
    """

    def _testprocess():
        single_test(
            encoder,
            head,
            test_dataloaders_all,
            auprc,
            modalnum,
            task,
            criterion,
        )

    all_in_one_test(_testprocess, [encoder, head])
