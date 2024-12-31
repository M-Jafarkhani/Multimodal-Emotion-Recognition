import time
from utils import format_time


def get_all_params(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


def all_in_one(trainprocess, trainmodules):
    starttime = time.time()
    train_losses, valid_losses = trainprocess()
    endtime = time.time()

    print("Training Time: " + format_time(endtime - starttime))
    print("Training Params: " + str(get_all_params(trainmodules)))
    return train_losses, valid_losses


def all_in_one_train(trainprocess, trainmodules):
    starttime = time.time()
    trainprocess()
    endtime = time.time()

    print("Training Time: " + format_time(endtime - starttime))
    print("Training Params: " + str(get_all_params(trainmodules)))


def all_in_one_test(testprocess, testmodules):
    teststart = time.time()
    testprocess()
    testend = time.time()
    print("Inference Time: " + format_time(testend - teststart))
    print("Inference Params: " + str(get_all_params(testmodules)))
