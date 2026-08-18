import time, csv, os
import numpy as np
import solve_gpu as sg

def gpu(h, J, nt, ch, sw, dev, sd):
    t = time.perf_counter()
    _, e = sg.sample(h, J, n_temp=nt, chains=ch, sweeps=sw, device=dev, seed=sd, verbose=False)
    return time.perf_counter() - t, float(e.min()), float(e.mean())

def neal(h, J, rd, sw, sd):
    import dimod
    try:
        from neal import SimulatedAnnealingSampler as S
    except Exception:
        from dwave.samplers import SimulatedAnnealingSampler as S
    n = len(h); Ju = np.triu(J, 1)
    lin = {i: float(h[i]) for i in range(n)}
    quad = {(i, j): float(Ju[i, j]) for i in range(n) for j in range(i+1, n) if Ju[i, j] != 0}
    bqm = dimod.BinaryQuadraticModel(lin, quad, 0.0, dimod.SPIN)
    t = time.perf_counter()
    ss = S().sample(bqm, num_reads=rd, num_sweeps=sw, seed=sd)
    E = np.asarray(ss.record.energy, float)
    return time.perf_counter() - t, float(E.min()), float(E.mean())

d = np.load("outputs/ising.npz", allow_pickle=True)
h, J = d["h"].astype(float), d["J"].astype(float)
NT, CH, SW, REPS = 16, 256, 2000, 5
reads = NT * CH
print("problem: n=%d spins | replicas=%d | sweeps=%d | reps=%d\n" % (len(h), reads, SW, REPS))
rows = []

def bench(name, fn, note):
    ts = []; bes = []; mes = []
    for r in range(REPS):
        try:
            dt, be, me = fn(r)
        except Exception as ex:
            print("  %-16s FAILED (%s: %s)" % (name, type(ex).__name__, ex)); return
        ts.append(dt); bes.append(be); mes.append(me)
    rows.append([name, float(np.mean(ts)), float(np.std(ts)), min(bes), float(np.mean(mes)), note])
    print("  %-16s time=%7.2f+/-%5.2fs  best_E=%12.2f  mean_E=%12.2f  %s" % (
        name, np.mean(ts), np.std(ts), min(bes), np.mean(mes), note))

bench("GPU-PT (B200)", lambda s: gpu(h, J, NT, CH, SW, "cuda", s), "%d replicas, %d sweeps" % (reads, SW))
bench("dwave-neal SA", lambda s: neal(h, J, reads, SW, s), "%d reads, %d sweeps" % (reads, SW))
bench("CPU-PT (NumPy)", lambda s: gpu(h, J, NT, 32, 500, "cpu", s), "%d replicas, 500 sweeps (reduced)" % (NT*32))

os.makedirs("outputs", exist_ok=True)
Ebest = min(r[3] for r in rows) if rows else 0.0
with open("outputs/benchmark_table2.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "time_s_mean", "time_s_std", "best_energy", "mean_best_energy", "energy_gap_vs_best", "settings"])
    for nm, tm, tsd, be, me, note in rows:
        gap = (be - Ebest) / (abs(Ebest) + 1e-9)
        w.writerow([nm, "%.3f" % tm, "%.3f" % tsd, "%.4f" % be, "%.4f" % me, "%.3e" % gap, note])
print("\n-> outputs/benchmark_table2.csv")
