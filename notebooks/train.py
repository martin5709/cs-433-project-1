from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cols import DROP_COLS, NOMINAL_CATEGORICAL
from tqdm import tqdm


def split_data(X: np.ndarray, y: np.ndarray, tr_rate: float = 0.8):
    np.random.seed(12345)

    y_flat = y.flatten()
    pos_ids = np.where(y_flat > 0)[0]
    neg_ids = np.where(y_flat < 0)[0]

    pos_perm = np.random.permutation(pos_ids)
    neg_perm = np.random.permutation(neg_ids)
    n_pos_tr = int(tr_rate * len(pos_perm))
    n_neg_tr = int(tr_rate * len(neg_perm))

    tr_ids = np.concatenate([pos_perm[:n_pos_tr], neg_perm[:n_neg_tr]])
    va_ids = np.concatenate([pos_perm[n_pos_tr:], neg_perm[n_neg_tr:]])

    return (X[tr_ids], y[tr_ids]), (X[va_ids], y[va_ids])


def k_fold(X: np.ndarray, y: np.ndarray, K: int = 5):
    np.random.seed(12345)
    N = y.shape[0]

    y_flat = y.flatten()
    pos_ids = np.where(y_flat > 0)[0]
    neg_ids = np.where(y_flat < 0)[0]

    pos_perm = np.random.permutation(pos_ids)
    neg_perm = np.random.permutation(neg_ids)
    n_pos_tr = len(pos_perm) // K
    n_neg_tr = len(neg_perm) // K

    fold_ids = [
        np.concatenate(
            [
                pos_perm[k * n_pos_tr : (k + 1) * n_pos_tr],
                neg_perm[k * n_neg_tr : (k + 1) * n_neg_tr],
            ]
        )
        for k in range(K)
    ]
    return fold_ids


def acc(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    return np.mean((X @ w) * y > 0)


def f1(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    pred = (X @ w).flatten()
    y = y.flatten()
    tp = np.sum((pred >= 0) & (y > 0))
    fp = np.sum((pred >= 0) & (y < 0))
    fn = np.sum((pred < 0) & (y > 0))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)


def sigmoid(x: np.ndarray):
    x = np.clip(x, -400, 400)
    return 1.0 / (1.0 + np.exp(-x))


def least_squares(X: np.ndarray, y: np.ndarray):
    return np.linalg.solve(X.T @ X + 10 * np.eye(X.shape[1]), X.T @ y)


def compute_loss(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, loss: str = "logistic", lambda_: float = 0
):
    if loss == "logistic":
        return np.mean(np.logaddexp(0, -y * (X @ w)))
    elif loss == "hinge":
        return np.mean(np.maximum(1 - y * (X @ w), 0))
    else:
        raise ValueError(f"{loss} is invalid")


def compute_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, loss: str = "logistic", lambda_: float = 0
):
    if loss == "logistic":
        return -np.expand_dims(np.mean(sigmoid(-y * (X @ w)) * y * X, axis=0), -1) + lambda_ * w
    elif loss == "hinge":
        return np.expand_dims(np.mean((-X * y) * (1 - y * (X @ w) > 0), axis=0), -1) + lambda_ * w
    else:
        raise ValueError(f"{loss} is invalid")


def get_lr(gamma: float, i: int, max_iters: int):
    warm_up_iters = int(0.05 * max_iters)
    if i < warm_up_iters:
        return gamma * ((i + 1) / warm_up_iters)
    else:
        return gamma * (1 - (i - warm_up_iters) / (max_iters - warm_up_iters)) ** 1.5


def gradient_descent(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    loss_fn: str,
    lambda_: float,
    batch_size: int = -1,
    beta: float = 0.9,
    gamma: float = 0.1,
    n_iters: int = 10000,
    valid_every: int = 10,
):
    n_tr, D = X_tr.shape
    pos_ids = np.where(y_tr[:, 0] == 1)[0]
    neg_ids = np.where(y_tr[:, 0] == -1)[0]

    # w = np.random.randn(D, 1) / np.sqrt(D)
    w = least_squares(X_tr, y_tr)

    best_w = w.copy()
    best_id = None
    loss_list = []
    val_loss_list = []
    acc_list = []
    f1_list = []
    loss_list.append(compute_loss(X_tr, y_tr, w, loss_fn, lambda_))
    val_loss_list.append(compute_loss(X_va, y_va, w, loss_fn, lambda_))
    acc_list.append(acc(X_va, y_va, w))
    f1_list.append(f1(X_va, y_va, w))
    momentum = np.zeros_like(w)
    with tqdm(range(n_iters), bar_format="{l_bar}{bar:10}{r_bar}") as pbar:
        for i in pbar:
            if batch_size > 0:
                n_pos = int(batch_size * 0.25)
                batch_ids = np.concatenate(
                    [
                        np.random.choice(pos_ids, n_pos, replace=False),
                        np.random.choice(neg_ids, batch_size - n_pos, replace=False),
                    ],
                    axis=0,
                )
            else:
                batch_ids = np.arange(n_tr)

            grad = compute_gradient(X_tr[batch_ids], y_tr[batch_ids], w, loss_fn, lambda_)

            if i == 0:
                momentum = grad
            else:
                momentum = beta * momentum + grad

            lr = get_lr(gamma, i, n_iters)
            w = w - lr * momentum

            if (i + 1) % valid_every == 0:
                loss_list.append(compute_loss(X_tr, y_tr, w, loss_fn, lambda_))
                val_loss_list.append(compute_loss(X_va, y_va, w, loss_fn, lambda_))
                acc_list.append(acc(X_va, y_va, w))
                f1_list.append(f1(X_va, y_va, w))

                if best_id is None or f1_list[(best_id + 1) // valid_every] < f1_list[-1]:
                    best_id = i
                    best_w = w.copy()

            pbar.set_postfix(
                {
                    "lr": lr,
                    "loss": loss_list[-1],
                    "val_loss": val_loss_list[-1],
                    "acc": acc_list[-1],
                    "f1": f1_list[-1],
                }
            )

    return w, best_id, best_w, loss_list, val_loss_list, acc_list, f1_list


def one_test(x_np, y_tr_np, n_trains, loss_fn: str, lambda_: float = 0.1, gamma: float = 1e-5):
    (X_tr, y_tr), (X_va, y_va) = split_data(x_np, y_tr_np, 0.8)

    w, best_id, best_w, loss_list, val_loss_list, acc_list, f1_list = gradient_descent(
        X_tr,
        y_tr,
        X_va,
        y_va,
        loss_fn=loss_fn,
        lambda_=lambda_,
        batch_size=2**12,
        beta=0.9,
        gamma=gamma,
        n_iters=2000,
        valid_every=100,
    )
    fig = plt.figure()
    ax = fig.add_subplot()
    name = f"save_data/{x_np.shape}_{loss_fn}_{lambda_}_{gamma}"
    _ = ax.set_title(name)
    _ = ax.plot(np.arange(0, len(loss_list)), loss_list)
    _ = ax.plot(np.arange(0, len(loss_list)), val_loss_list)
    fig.savefig(f"{name}.pdf")
    plt.close(fig)
    np.save(f"{name}.npy", best_w)
    return best_w, acc(X_va, y_va, best_w), f1(X_va, y_va, best_w)


def main():
    LOCAL_DIR = Path("../local")
    DATA_DIR = LOCAL_DIR / "dataset"
    X_TRAIN_PATH = DATA_DIR / "x_train.csv"
    Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
    X_TEST_PATH = DATA_DIR / "x_test.csv"

    x_train_df = pd.read_csv(X_TRAIN_PATH, engine="python")
    y_train_df = pd.read_csv(Y_TRAIN_PATH, engine="python")
    x_test_df = pd.read_csv(X_TEST_PATH, engine="python")

    x_tr_np = x_train_df.to_numpy()
    x_te_np = x_test_df.to_numpy()
    # x_np = np.concat([x_tr_np, x_te_np], axis=0)
    x_np = x_tr_np
    y_tr_np = y_train_df.to_numpy()

    n_trains = x_tr_np.shape[0]
    n_tests = x_te_np.shape[0]

    # Preprocessing
    # x_df = pd.concat([x_train_df, x_test_df])
    x_df = x_train_df
    x_df = x_df.drop(columns=DROP_COLS, errors="ignore")
    N_DEGREES = 1
    new_cols = [np.ones(x_np.shape[0])]
    for c in tqdm(x_df.columns[1:], desc="Preprocessing"):
        col = x_df[c].to_numpy().copy()

        # if pd.isnull(col).mean() > 0.75:
        #     continue

        if c in NOMINAL_CATEGORICAL:
            u_values = np.unique(col)
            for v in u_values:
                if not np.isnan(v):
                    new_cols.append((col == v).astype(np.float64))
        else:
            mean = np.mean(col[~np.isnan(col)])
            std = np.std(col[~np.isnan(col)])
            if std > 0:
                norm_col = np.nan_to_num((col - mean) / std, nan=0)
                for d in range(1, N_DEGREES + 1):
                    new_cols.append(norm_col**d)

    print("Stacking...")
    pp_x_np = np.stack(new_cols, axis=-1)
    pp_x_tr_np = pp_x_np[:n_trains]
    pp_x_te_np = pp_x_np[n_trains:]

    print("N_TRAINs = ", pp_x_tr_np.shape[0])
    print("N_COLS = ", pp_x_tr_np.shape[1])

    best_w = None
    best_f1 = None
    for loss_fn in ["hinge", "logistic"]:
        for lambda_ in [1e-3, 1e-6, 1e-9, 0]:
            for gamma in [1, 1e-1, 1e-2, 1e-3]:
                print("\n##")
                print("\tn_degree =", N_DEGREES)
                print("\tloss_fn =", loss_fn)
                print("\tlambda_ =", lambda_)
                print("\tgamma =", gamma)
                w, va_acc, va_f1 = one_test(
                    pp_x_tr_np,
                    y_tr_np[:, 1:],
                    n_trains,
                    loss_fn=loss_fn,
                    lambda_=lambda_,
                    gamma=gamma,
                )
                print("## acc =", va_acc)
                print("## f1 =", va_f1)
                if best_f1 is None or best_f1 < va_f1:
                    best_f1 = va_f1
                    best_w = w.copy()
                    print("## New best")

    np.save(f"best_w_d{N_DEGREES}.npy", best_w)


if __name__ == "__main__":
    main()
