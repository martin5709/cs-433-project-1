from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from cols import BINARY_COLS, DROP_COLS, NOMINAL_CATEGORICAL


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


def acc(X: np.ndarray, y: np.ndarray, w: np.ndarray, t: float = 0):
    return np.mean((X @ w) * y > t)


def f1(X: np.ndarray, y: np.ndarray, w: np.ndarray, t: float = 0):
    pred = (X @ w).flatten()
    y = y.flatten()
    tp = np.sum((pred >= t) & (y > 0))
    fp = np.sum((pred >= t) & (y < 0))
    fn = np.sum((pred < t) & (y > 0))
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
    elif loss == "focal":
        return np.mean(sigmoid(-y * (X @ w)) * np.logaddexp(0, -y * (X @ w)))
    else:
        raise ValueError(f"{loss} is invalid")


def compute_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, loss: str = "logistic", lambda_: float = 0
):
    if loss == "logistic":
        return -np.expand_dims(np.mean(sigmoid(-y * (X @ w)) * y * X, axis=0), -1) + lambda_ * (w)
    elif loss == "hinge":
        return np.expand_dims(np.mean((-X * y) * (1 - y * (X @ w) > 0), axis=0), -1) + lambda_ * (w)
    elif loss == "focal":
        return -np.expand_dims(
            np.mean(
                sigmoid(-y * (X @ w)) * sigmoid(y * (X @ w)) * np.logaddexp(0, -y * (X @ w)) * y * X
                + (sigmoid(-y * (X @ w)) ** 2) * y * X,
                axis=0,
            ),
            -1,
        ) + lambda_ * (w)

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

    w = np.random.randn(D, 1) / np.sqrt(D)
    # w = least_squares(X_tr, y_tr)

    best_w = w.copy()
    best_f1 = None
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

                if best_id is None or best_f1 < f1_list[-1]:
                    best_id = i
                    best_f1 = f1_list[-1]
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

    return w, best_f1, best_id, best_w, loss_list, val_loss_list, acc_list, f1_list


def one_test(x_tr_np, y_tr_np, loss_fn: str, lambda_: float = 0.1, gamma: float = 1e-5):
    # (X_tr, y_tr), (X_va, y_va) = split_data(x_np, y_tr_np, 0.8)
    fold_ids = k_fold(x_tr_np, y_tr_np, 5)

    w_list: list[np.ndarray] = []
    best_acc_list: list[float] = []
    best_f1_list: list[float] = []

    for k, valid_ids in enumerate(fold_ids):
        valid_mask = np.zeros(x_tr_np.shape[0], dtype=np.bool_)
        valid_mask[valid_ids] = True
        X_tr = x_tr_np[~valid_mask]
        y_tr = y_tr_np[~valid_mask]
        X_va = x_tr_np[valid_mask]
        y_va = y_tr_np[valid_mask]
        print(f"Fold {k}")
        print(f"n_trains={X_tr.shape[0]}")
        print(f"n_valids={X_va.shape[0]}")

        w, best_f1, best_id, best_w, loss_list, val_loss_list, acc_list, f1_list = gradient_descent(
            X_tr,
            y_tr,
            X_va,
            y_va,
            loss_fn=loss_fn,
            lambda_=lambda_,
            batch_size=2**12,
            beta=0.9,
            gamma=gamma,
            n_iters=5000,
            valid_every=100,
        )
        fig = plt.figure()
        ax = fig.add_subplot()
        name = f"save_data/{loss_fn}_{lambda_}_{gamma}_fold{k}"
        _ = ax.set_title(name)
        _ = ax.plot(np.arange(0, len(loss_list)), loss_list)
        _ = ax.plot(np.arange(0, len(loss_list)), val_loss_list)
        fig.savefig(f"{name}.pdf")
        plt.close(fig)
        np.save(f"{name}.npy", best_w)

        w_list.append(best_w)
        best_acc_list.append(acc(X_va, y_va, best_w))
        best_f1_list.append(f1(X_va, y_va, best_w))
        print(f"Best: f1={f1_list[-1]:.4f} acc={acc_list[-1]:.4f} at iter={best_id}")

    return w_list, best_acc_list, best_f1_list


# THIS LOGIC IS GENERATED BY GEMINI (TAKE IT WITH A GRAIN OF SALT)
def clean_brfss_column(col: np.ndarray, col_name: str) -> np.ndarray:
    col = col.astype(np.float64)

    # 1. CDC Variables with 2 Implied Decimals (divide by 100)
    if col_name in ["_BMI5", "HTM4", "WTKG3", "_FRUTSUM", "_VEGESUM"]:
        col[np.isin(col, [99999, 99900, 9999])] = np.nan
        return col / 100.0
    # 2. Days of Poor Health (88 = "None" -> 0)
    elif col_name in ["PHYSHLTH", "MENTHLTH", "POORHLTH"]:
        col[np.isin(col, [77, 99])] = np.nan
        col[col == 88] = 0.0
        return col
    # 3. Pain Scale (0-10: 7 and 9 are real pain levels, only 77/99 are missing)
    elif col_name == "JOINPAIN":
        col[np.isin(col, [77, 99])] = np.nan
        return col
    # 4. Employment: 7=Retired (VALID!), 9=Refused
    elif col_name == "EMPLOY1":
        col[col == 9] = np.nan
        return col
    # 5. Age Groups: 1-13 are valid, 14 is Missing
    elif col_name == "_AGEG5YR":
        col[col == 14] = np.nan
        return col
    # 6. Binary Yes/No Questions (7="Don't know", 9="Refused")
    elif col_name in BINARY_COLS:
        col[np.isin(col, [7, 9])] = np.nan
        return col
    # 7. Exercise / Drinks Calculated Variables (Missing code is 99900 or 99000)
    elif col_name in ["MAXVO2_", "FC60_", "_DRNKWEK"]:
        col[np.isin(col, [99900, 99000])] = np.nan
        return col
    # 8. All other columns: only remove multi-digit refusal/don't know codes
    else:
        col[np.isin(col, [77, 99, 777, 999, 7777, 9999, 99900])] = np.nan
        return col


def main(args):
    DATA_DIR = Path(args.data_dir)
    X_TRAIN_PATH = DATA_DIR / "x_train.csv"
    Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
    X_TEST_PATH = DATA_DIR / "x_test.csv"

    x_train_df = pd.read_csv(X_TRAIN_PATH, engine="python")
    y_train_df = pd.read_csv(Y_TRAIN_PATH, engine="python")
    x_test_df = pd.read_csv(X_TEST_PATH, engine="python")

    x_tr_np = x_train_df.to_numpy()
    x_te_np = x_test_df.to_numpy()
    x_np = np.concat([x_tr_np, x_te_np], axis=0)
    y_tr_np = y_train_df.to_numpy()

    n_trains = x_tr_np.shape[0]
    n_tests = x_te_np.shape[0]

    # Preprocessing
    x_df = pd.concat([x_train_df, x_test_df])
    x_df = x_df.drop(columns=DROP_COLS, errors="ignore")
    N_DEGREES = 1
    new_cols = [np.ones(x_np.shape[0])]
    col_names = ["const"]
    for c in tqdm(x_df.columns[1:], desc="Preprocessing"):
        assert "#" not in c
        col = x_df[c].to_numpy().copy()
        col = clean_brfss_column(col, c)

        # if pd.isnull(col).mean() > 0.75:
        #     continue

        if c in NOMINAL_CATEGORICAL:
            u_values = np.unique(col)
            for v in u_values:
                if not np.isnan(v):
                    new_cols.append((col == v).astype(np.float64))
                    col_names.append(f"{c}_C#{v}")
        else:
            mean = np.mean(col[~np.isnan(col)])
            std = np.std(col[~np.isnan(col)])
            if std > 0:
                norm_col = np.nan_to_num((col - mean) / std, nan=0)
                for d in range(1, N_DEGREES + 1):
                    new_cols.append(norm_col**d)
                    col_names.append(f"{c}_N#{d}")

    print("Stacking...")
    assert len(new_cols) == len(col_names)
    pp_x_np = np.stack(new_cols, axis=-1)
    pp_x_tr_np = pp_x_np[:n_trains]
    pp_x_te_np = pp_x_np[n_trains:]

    print("N_TRAINs = ", pp_x_tr_np.shape[0])
    print("N_COLS = ", pp_x_tr_np.shape[1])

    def train():
        for loss_fn in ["focal", "logistic"]:
            best_w = np.zeros(0)
            best_f1 = None
            for lambda_ in [1e-4, 1e-5, 0]:
                for gamma in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
                    print("\n##")
                    print("\tn_degree =", N_DEGREES)
                    print("\tloss_fn =", loss_fn)
                    print("\tlambda_ =", lambda_)
                    print("\tgamma =", gamma)
                    w_list, acc_list, f1_list = one_test(
                        pp_x_tr_np,
                        y_tr_np[:, 1:],
                        loss_fn=loss_fn,
                        lambda_=lambda_,
                        gamma=gamma,
                    )

                    w = np.array(w_list).mean(axis=0)
                    k = 20
                    sorted_ids = np.argsort(np.abs(w), axis=None, descending=True)
                    topk_cols = set()
                    topk_w = []
                    for col_id in sorted_ids:
                        name = col_names[col_id]
                        if name not in topk_cols:
                            topk_cols.add(name)
                            topk_w.append(w[col_id].item())

                        if len(topk_cols) == k:
                            break

                    va_acc = np.mean(np.array(acc_list))
                    va_f1 = np.mean(np.array(f1_list))
                    full_acc = acc(pp_x_tr_np, y_tr_np[:, 1:], w)
                    full_f1 = f1(pp_x_tr_np, y_tr_np[:, 1:], w)

                    print("## valid acc =", va_acc)
                    print("## valid f1 =", va_f1)
                    print("## full acc =", full_acc)
                    print("## full f1 =", full_f1)
                    print(f"## Top-{k} w =", list(zip(list(topk_cols), topk_w)))
                    if best_f1 is None or best_f1 < va_f1:
                        best_f1 = va_f1
                        best_w = np.array(w_list)
                        print("## New best")

            np.save(f"best_w_d{N_DEGREES}_{loss_fn}.npy", best_w)

    def generate_submission():
        w_list = []
        for loss_fn in ["focal"]:
            w = np.load(f"best_w_d{N_DEGREES}_{loss_fn}.npy")
            w_list.append(w)

        best_w = np.concat(w_list, axis=0)
        best_w = best_w.mean(axis=0)

        best_t = None
        best_acc = None
        best_f1 = None
        for t in np.linspace(-100, 100, 201):
            full_acc = acc(pp_x_tr_np, y_tr_np[:, 1:], best_w, t)
            full_f1 = f1(pp_x_tr_np, y_tr_np[:, 1:], best_w, t)
            if best_f1 is None or best_f1 < full_f1:
                best_t = t
                best_acc = full_acc
                best_f1 = full_f1

        print(f"t = {best_t}")
        print(f"Acc = {best_acc}")
        print(f"f1 = {best_f1}")

        logit = (pp_x_te_np @ best_w).flatten()
        y_pred = np.ones(pp_x_te_np.shape[0])
        y_pred[logit < 0] = -1
        y_pred[logit >= 0] = 1
        submission_df = pd.DataFrame(
            {"Id": x_test_df["Id"].astype(int), "Prediction": y_pred.astype(int)}
        )
        submission_df.to_csv("submission.csv", index=False)

    train()
    generate_submission()


def parse_args():
    parser = ArgumentParser()

    _ = parser.add_argument("--data-dir", default="./local/dataset")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
