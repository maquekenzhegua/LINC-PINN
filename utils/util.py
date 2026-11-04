from sklearn import metrics
import numpy as np
import logging

def get_logger(log_name='log.txt'):
    logger = logging.getLogger('mylogger')
    # Reuse of the shared named logger would otherwise accumulate handlers
    # (one per PINN instance), duplicating every log line in old files.
    for _h in tuple(logger.handlers):
        logger.removeHandler(_h)
        _h.close()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - function:%(funcName)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name is not None:
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))

    return [MAE,MAPE,MSE,RMSE]

def battery_metrics(true_label, pred_label, gap):
    """Split a run back into batteries and average the per-battery errors.

    Adjacent rows are treated as a battery boundary when the true SOH jumps by
    more than ``gap``; the reported MAE/MAPE/RMSE is then the mean over
    batteries, which is the aggregation used in the manuscript.
    """
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    splits = np.where(np.diff(true_label) > gap)[0]
    starts = [0] + (splits + 1).tolist()
    ends = splits.tolist() + [len(true_label) - 1]
    maes, mapes, rmses = [], [], []
    for s, e in zip(starts, ends):
        t = true_label[s:e + 1]
        q = pred_label[s:e + 1]
        if len(t) < 2:
            continue
        mae, mape, _mse, rmse = eval_metrix(t, q)
        maes.append(mae); mapes.append(mape); rmses.append(rmse)
    if not mapes:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(maes)), float(np.mean(mapes)), float(np.mean(rmses))


def read_run(folder):
    """Read one Experiment folder and return (true, pred) arrays."""
    import os
    true_label = np.load(os.path.join(folder, "true_label.npy"))
    pred_label = np.load(os.path.join(folder, "pred_label.npy"))
    return true_label, pred_label

def write_to_txt(txt_name,txt):
    with open(txt_name,'a') as f:
        f.write(txt)
        f.write('\n')

if __name__ == '__main__':
    for i in range(3):
        logger_name = f'log_{i}.txt'
        if i == 1:
            logger_name = None
        logger = get_logger(log_name=logger_name)
        logger.info(f'time: {i}, This is a log info')
        logger.removeHandler(logger.handlers[0])
        logger.handlers.clear()

