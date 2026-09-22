"""Twin state estimators.

All estimators map the last L minutes of observations to the twin state at minute t:
    behaviour probabilities (5), lying probability, core body temperature (degC).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

NB = len(C.BEHAVIOURS)


# ----------------------------------------------------------------------------- neural backbones
class GRUNet(nn.Module):
    def __init__(self, d_in, d=64, layers=2, p=0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.rnn = nn.GRU(d, d, layers, batch_first=True, dropout=p)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h, _ = self.rnn(F.gelu(self.inp(x)))
        return self.drop(h[:, -1])


class TransformerNet(nn.Module):
    def __init__(self, d_in, d=64, layers=2, heads=4, p=0.1, L=60):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):
        L = x.shape[1]
        h = self.enc(self.inp(x) + self.pos[:, -L:], mask=self.mask[:L, :L])
        return self.drop(self.norm(h[:, -1]))


class SelectiveSSMBlock(nn.Module):
    """Mamba-2-style selective state-space block (SSD form).

    Input-dependent step dt_t, input/output projections B_t, C_t and a scalar decay per channel:
        h_t = exp(dt_t * a) h_{t-1} + dt_t B_t u_t ,   y_t = C_t . h_t + D u_t
    evaluated in its equivalent masked-matrix form, which is parallel over time and numerically
    stable because every exponent is <= 0.
    """

    def __init__(self, d, d_state=16, expand=2, conv=4, p=0.1):
        super().__init__()
        di = expand * d
        self.di, self.ds = di, d_state
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, conv, groups=di, padding=conv - 1)
        self.x_proj = nn.Linear(di, 2 * d_state)
        self.dt_proj = nn.Linear(di, di)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, di)))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, L, _ = x.shape
        u, z = self.in_proj(self.norm(x)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        Bm, Cm = self.x_proj(u).chunk(2, -1)                               # (B, L, ds)
        dt = F.softplus(self.dt_proj(u) - 2.0)                             # (B, L, di)
        a = -torch.exp(self.A_log)
        if self.training:
            S = torch.cumsum(dt * a, 1)                                    # (B, L, di), decreasing
            seg = S.unsqueeze(2) - S.unsqueeze(1)                          # (B, t, s, di) = S_t - S_s
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            decay = torch.exp(seg.masked_fill(~causal[None, :, :, None], float("-inf")))
            G = torch.einsum("btn,bsn->bts", Cm, Bm)
            y = torch.einsum("btsd,bts,bsd->btd", decay, G, dt * u)
        else:
            # identical recurrence, evaluated as a scan: 4x faster for large inference batches
            dA, xu = torch.exp(dt * a), dt * u
            h = torch.zeros(B, self.di, self.ds, device=x.device)
            ys = []
            for t in range(L):
                h = dA[:, t, :, None] * h + xu[:, t, :, None] * Bm[:, t, None, :]
                ys.append((h * Cm[:, t, None, :]).sum(-1))
            y = torch.stack(ys, 1)
        y = y + u * self.D
        return x + self.drop(self.out_proj(y * F.silu(z)))


class SSMNet(nn.Module):
    def __init__(self, d_in, d=64, layers=2, p=0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(d, p=p) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.norm(h[:, -1])


BACKBONES = {"gru": GRUNet, "transformer": TransformerNet, "ssm": SSMNet}


class TwinNet(nn.Module):
    def __init__(self, arch, d_in, d=64, L=60):
        super().__init__()
        kw = {"L": L} if arch == "transformer" else {}
        self.body = BACKBONES[arch](d_in, d=d, **kw)
        self.head = nn.Linear(d, NB + 2)   # behaviour logits, lying logit, CBT (standardised)

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, :NB], o[:, NB], o[:, NB + 1]


# ----------------------------------------------------------------------------- training data
class Windows:
    """Index over (version, cow, minute) with lazily cut windows of length L."""

    def __init__(self, obs, tgt, L):
        self.obs, self.L = obs, L          # obs[v][c] : (N_MIN, F)
        self.tgt = tgt                     # tgt[c] = (label, lying, cbt_std)
        idx = []
        for v in range(len(obs)):
            for c in range(len(obs[v])):
                t = np.arange(L - 1, C.N_MIN)
                idx.append(np.column_stack([np.full_like(t, v), np.full_like(t, c), t]))
        self.idx = np.concatenate(idx)
        self.lab = np.stack([t[0] for t in tgt])
        self.ly = np.stack([t[1] for t in tgt])
        self.cb = np.stack([t[2] for t in tgt])
        lab = self.lab[self.idx[:, 1], self.idx[:, 2]]
        self.labeled = np.where(lab >= 0)[0]
        self.unlabeled = np.where(lab < 0)[0]

    def batch(self, ids):
        v, c, t = self.idx[ids].T
        x = np.stack([self.obs[vi][ci][ti - self.L + 1: ti + 1] for vi, ci, ti in zip(v, c, t)])
        return x, self.lab[c, t], self.ly[c, t], self.cb[c, t]


def device():
    import os
    if os.environ.get("DTQ1_DEVICE"):
        return torch.device(os.environ["DTQ1_DEVICE"])
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_nn(arch, obs, tgt, seed, L=30, epochs=12, steps=400, bs=256, lr=2e-3, class_w=None, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    W = Windows(obs, tgt, L)
    net = TwinNet(arch, obs[0][0].shape[1], L=L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps)
    cw = torch.tensor(class_w, dtype=torch.float32, device=dev) if class_w is not None else None
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _ in range(steps):
            ids = np.concatenate([rng.choice(W.labeled, bs // 2), rng.choice(W.unlabeled, bs // 2)])
            x, y, ly, cb = (torch.from_numpy(a).to(dev) for a in W.batch(ids))
            lo, ll, lc = net(x.float())
            m, ml, mc = y >= 0, ~torch.isnan(ly), ~torch.isnan(cb)
            loss = F.cross_entropy(lo[m], y[m], weight=cw) + \
                F.binary_cross_entropy_with_logits(ll[ml], ly[ml].float()) + F.mse_loss(lc[mc], cb[mc].float())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if log:
            log(f"{arch} seed{seed} ep{ep} loss {tot / steps:.3f}")
    return net.cpu().eval()


@torch.no_grad()
def predict_nn(net, obs, L=30, bs=2048, mc=0):
    """Returns (p_behaviour (N,5), p_lying (N,), cbt_std (N,)) for every minute (first L-1 padded)."""
    dev = device()
    net = net.to(dev)
    if mc:
        net.train()
    pad = np.concatenate([np.repeat(obs[:1], L - 1, 0), obs])
    win = np.lib.stride_tricks.sliding_window_view(pad, L, axis=0).transpose(0, 2, 1)
    outs = []
    for i in range(0, len(win), bs):
        x = torch.from_numpy(np.ascontiguousarray(win[i:i + bs])).to(dev)
        lo, ll, lc = net(x)
        outs.append(torch.cat([F.softmax(lo, -1), torch.sigmoid(ll)[:, None], lc[:, None]], 1).cpu().numpy())
    net.eval().cpu()
    o = np.concatenate(outs)
    return o[:, :NB], o[:, NB], o[:, NB + 1]
