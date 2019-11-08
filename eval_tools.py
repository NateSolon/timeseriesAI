# Based on work by Ignacio Oguiza.
# https://github.com/timeseriesAI/timeseriesAI

from sklearn.linear_model import RidgeClassifierCV
from fastai_timeseries import *


def evaluate(model,
             iterations = 5,
             datasets = sorted(get_UCR_univariate_list())
            ):
    # user provides function that takes x_train, y_train, and x_valid and returns y_pred.

    ds_ = []
    ds_nl_ = []
    ds_di_ = []
    ds_type_ = []
    iters_ = []
    means_ = []
    stds_ = []
    times_ = []

    datasets = listify(datasets)
    for dsid in datasets:
        try:
            print(f'\nProcessing {dsid} dataset...\n')
            if dsid in get_UCR_univariate_list(): scale_type=None
            else: scale_type='standardize'
            X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)
        except:
            ds_nl_.append(dsid)
            print(f'\n...{dsid} cannot be loaded\n')
            continue
        if np.isinf(X_train).sum() + np.isnan(X_train).sum() > 0:
            ds_di_.append(dsid)
            print(f'...{dsid} contains inf or nan\n')
            continue

        if X_train.shape[1] == 1: ds_type_.append('univariate')
        else: ds_type_.append('multivariate')

        score_ = []
        elapsed_times_ = []
        for i in range(iterations):
            start_time = time.time()
            y_pred = model(X_train, y_train, X_valid)
            score = get_score(y_valid, y_pred)
            score_.append(score)
            elapsed_time = time.time() - start_time
            elapsed_times_.append(elapsed_time)
            print('   {:2} - score: {:.5f}  time(s): {:4.0f}'.\
                  format(i + 1, score, elapsed_time))
        ds_.append(dsid)
        iters_.append(iterations)
        means_.append('{:.6f}'.format(np.mean(score_)))
        stds_.append('{:.6f}'.format(np.std(score_)))
        times_.append('{:.0f}'.format(np.mean(elapsed_times_)))
    df = pd.DataFrame(np.stack((ds_, ds_type_, iters_,  means_, stds_, times_)).T,
             columns=['dataset', 'type', 'iterations',  'mean_accuracy', 'std_accuracy',  'time(s)'])
    if ds_nl_ != []: print('\n(*) datasets not loaded      :', ds_nl_)
    if ds_di_ != []:print('(*) datasets with data issues:', ds_di_)

    return df


def random_choice(X_train, y_train, X_valid):
    c = np.unique(y_train)
    y_pred = np.random.choice(c, size=X_valid.shape[0])
    return y_pred


def get_score(y, yhat):
    return (yhat==y).sum()/len(y)


def rocket_vanilla(X_train, y_train, X_valid):
    seq_len = X_train.shape[-1]
    X_train = X_train[:, 0].astype(np.float64)
    X_valid = X_valid[:, 0].astype(np.float64)
    X_train = (X_train - X_train.mean(axis = 1, keepdims = True)) / (X_train.std(axis = 1, keepdims = True) + 1e-8)
    X_valid = (X_valid - X_valid.mean(axis = 1, keepdims = True)) / (X_valid.std(axis = 1, keepdims = True) + 1e-8)
    kernels = generate_kernels(seq_len, 10000)
    X_train_tfm = apply_kernels(X_train, kernels)
    X_valid_tfm = apply_kernels(X_valid, kernels)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7), normalize=True)
    classifier.fit(X_train_tfm, y_train)
    preds = classifier.predict(X_valid_tfm)
    return preds