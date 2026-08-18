"""
CUDA-Ising 求解器(GPU 平行退火 / 平行回火)—— 這就是你要接的求解器
==================================================================
問題是「任意稠密耦合圖」的 Ising(基因兩兩都可能有耦合),不是 2D 晶格,
所以用 batched simulated annealing + parallel tempering(PT):
一次在 GPU 上跑「上千條 replica」,正好吃滿 DGX-B200 的平行度。

- 有 PyTorch + CUDA -> 自動用 GPU(DGX-B200 就走這條)。
- 沒有 torch -> 退回 NumPy(演算法完全相同,方便離線/小規模驗證)。

能量慣例與 step03 一致(dimod 最小化):
    E(s) = h·s + sum_{i<j} J_ij s_i s_j = 0.5 s^T Jsym s + h·s,  Jsym=triu(J)+triu(J)^T
翻轉自旋 i 的能量變化: dE = -2 s_i (h_i + (Jsym s)_i)。

輸入:outputs/ising.npz(h, J, genes)—— step03 產出。
輸出:每列一個 ±1 解的狀態檔(預設 outputs/states.txt),直接餵給
      step05_postprocess.py --states outputs/states.txt。

用法(DGX 上):
    python solve_gpu.py --ising outputs/ising.npz --out outputs/states.txt \
        --n-temp 16 --chains 128 --sweeps 1000 --device cuda
"""
from __future__ import annotations
import argparse
import numpy as np


# ===================== 後端(torch GPU / numpy)=====================
class NumpyBackend:
    name = "numpy"

    def __init__(self, seed=0, device="cpu"):
        self.rng = np.random.default_rng(seed)

    def pm1(self, shape):                 # 隨機 ±1
        return self.rng.integers(0, 2, size=shape).astype(np.float64) * 2 - 1

    def rand(self, shape):
        return self.rng.random(shape)

    asarray = staticmethod(lambda a: np.asarray(a, dtype=np.float64))
    exp = staticmethod(np.exp)
    where = staticmethod(np.where)
    to_numpy = staticmethod(lambda a: np.asarray(a))

    def safe_exp(self, a):                # exp(min(a,0)) == min(1, exp(a)),避免溢位
        return np.exp(np.minimum(a, 0.0))


class TorchBackend:
    name = "torch"

    def __init__(self, seed=0, device="cuda"):
        import torch
        self.torch = torch
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(int(seed))

    def pm1(self, shape):
        t = self.torch.randint(0, 2, shape, generator=self.gen,
                               device=self.device, dtype=self.torch.float32)
        return t * 2 - 1

    def rand(self, shape):
        return self.torch.rand(shape, generator=self.gen, device=self.device)

    def asarray(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float32),
                                    device=self.device)

    def exp(self, a):
        return self.torch.exp(a)

    def safe_exp(self, a):                # exp(min(a,0)),避免溢位
        return self.torch.exp(self.torch.clamp(a, max=0.0))

    def where(self, c, a, b):
        return self.torch.where(c, a, b)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()


def make_backend(device, seed):
    if device != "cpu":
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                print("[solver] 找不到 CUDA,改用 CPU torch。")
                device = "cpu"
            return TorchBackend(seed, device)
        except Exception as e:
            print(f"[solver] 無 torch({type(e).__name__}),退回 NumPy。")
    return NumpyBackend(seed)


# ===================== PT 取樣器 =====================
def energies(B, S, Jsym, h):
    # 0.5 * sum(S @ Jsym * S) + S @ h
    return 0.5 * (S @ Jsym * S).sum(1) + S @ h    # .sum(1):numpy/torch 皆可


def sample(h, J, n_temp=16, chains=128, sweeps=1000, tmin=0.05, tmax=5.0,
           swap_interval=5, device="cuda", seed=0, verbose=True):
    B = make_backend(device, seed)
    n = len(h)
    R = n_temp * chains

    Jsym_np = np.triu(J, 1)
    Jsym_np = Jsym_np + Jsym_np.T
    Jsym = B.asarray(Jsym_np)
    hh = B.asarray(h)

    # 溫度階梯:block 佈局,replica r 屬於溫度 r//chains
    ladder = np.geomspace(tmax, tmin, n_temp)
    beta_block = 1.0 / ladder
    beta = B.asarray(np.repeat(beta_block, chains))       # (R,)

    S = B.pm1((R, n))
    bestS = S
    bestE = energies(B, S, Jsym, hh)

    order = np.arange(n)
    for sweep in range(sweeps):
        # --- 一個 sweep:逐 spin、跨 R 條 replica 平行 Metropolis ---
        np.random.default_rng(seed + sweep).shuffle(order)
        for i in order:
            local = S @ Jsym[:, i] + hh[i]                # (R,)
            dE = -2.0 * S[:, i] * local
            accept = (dE <= 0) | (B.rand((R,)) < B.safe_exp(-dE * beta))
            S[:, i] = B.where(accept, -S[:, i], S[:, i])

        # --- Parallel tempering:相鄰溫度交換 ---
        if swap_interval and (sweep % swap_interval == 0):
            E = energies(B, S, Jsym, hh)
            for t in range(n_temp - 1):
                a0, b0 = t * chains, (t + 1) * chains
                Ea, Eb = E[a0:a0 + chains], E[b0:b0 + chains]
                delta = (beta_block[t] - beta_block[t + 1]) * (Ea - Eb)
                sw = B.rand((chains,)) < B.safe_exp(delta)
                # 交換被接受的鏈的整組自旋
                sa = S[a0:a0 + chains].clone() if B.name == "torch" else S[a0:a0 + chains].copy()
                sb = S[b0:b0 + chains].clone() if B.name == "torch" else S[b0:b0 + chains].copy()
                mask = sw.reshape(-1, 1)
                S[a0:a0 + chains] = B.where(mask, sb, sa)
                S[b0:b0 + chains] = B.where(mask, sa, sb)
                Ea2 = B.where(sw, Eb, Ea)
                Eb2 = B.where(sw, Ea, Eb)
                E[a0:a0 + chains] = Ea2
                E[b0:b0 + chains] = Eb2

        # --- 記錄每條 replica 造訪過的最低能量組態 ---
        E = energies(B, S, Jsym, hh)
        improve = E < bestE
        m = improve.reshape(-1, 1)
        bestS = B.where(m, S, bestS)
        bestE = B.where(improve, E, bestE)

        if verbose and (sweep % max(1, sweeps // 10) == 0):
            print(f"  sweep {sweep:5d}/{sweeps}  best E={float(B.to_numpy(bestE).min()):.4f}")

    return B.to_numpy(bestS), B.to_numpy(bestE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ising", default="outputs/ising.npz")
    ap.add_argument("--out", default="outputs/states.txt")
    ap.add_argument("--n-temp", type=int, default=16)
    ap.add_argument("--chains", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=1000)
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=5.0)
    ap.add_argument("--swap-interval", type=int, default=5)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.ising, allow_pickle=True)
    h, J = d["h"].astype(float), d["J"].astype(float)
    print(f"[solver] n={len(h)} spins, R={args.n_temp*args.chains} replicas, "
          f"sweeps={args.sweeps}, device={args.device}")

    bestS, bestE = sample(h, J, args.n_temp, args.chains, args.sweeps,
                          args.tmin, args.tmax, args.swap_interval,
                          args.device, args.seed)

    # 只輸出「有選到基因」的解會較有用;這裡輸出全部 best 組態(±1),
    # 交給 step05 依能量排序、取低能量群統計頻率。
    order = np.argsort(bestE)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(f"# CUDA-Ising PT solver | best E={bestE.min():.6f} | "
                 f"{len(bestS)} states | spin +1 = 基因被選\n")
        for r in order:
            fh.write(" ".join("1" if v > 0 else "-1" for v in bestS[r]) + "\n")
    print(f"[solver] 最低能量 = {bestE.min():.4f}")
    print(f"[solver] 已寫 {len(bestS)} 個解 -> {args.out}")
    print(f"接著:python step05_postprocess.py <config> --states {args.out}")


if __name__ == "__main__":
    main()
