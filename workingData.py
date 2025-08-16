import torch

def working_data(
    n: int = 200,
    weight: float = 0.7,
    bias: float = 0.3,
    noise_std: float = 0.05,        # zgomot gaussian de bază
    hetero: bool = True,            # zgomot crește cu X (heteroscedastic)
    nonlinearity: float = 0.15,     # adaugă un mic termen X^2
    outlier_frac: float = 0.08,     # ~8% puncte outlier
    outlier_scale: float = 6.0,     # cât de mari sunt outlier-urile
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str | torch.device = "cpu",
):
    torch.manual_seed(seed)
    device = torch.device(device)

    # X în [0, 1], neuniform (mai dens spre capete, de ex.)
    X = torch.sort(torch.rand(n) ** 1.2).values.unsqueeze(1).to(device)

    # țintă “aproape” liniară cu un mic termen neliniar
    y_clean = weight * X + bias + nonlinearity * (X ** 2)

    # zgomot gaussian; dacă hetero, cresc odată cu X
    base_noise = torch.randn_like(y_clean) * noise_std
    if hetero:
        base_noise = base_noise * (1.0 + 2.0 * X)  # mai mult zgomot la X mare

    y = y_clean + base_noise

    # outlieri: câteva puncte cu deviații mari
    k = int(outlier_frac * n)
    if k > 0:
        idx = torch.randperm(n)[:k]
        signs = torch.sign(torch.randn(k, 1).to(device))
        y[idx] = y[idx] + signs * outlier_scale * noise_std

    # amestecă înainte de split (altfel train/test ar fi “ordonate” pe X)
    perm = torch.randperm(n)
    X, y = X[perm], y[perm]

    # split
    train_split = int(train_ratio * n)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test   = X[train_split:], y[train_split:]

    # tipuri consistente
    return (X_train.float(), y_train.float(),
            X_test.float(),  y_test.float())
