"""Validate a hand-authored oracle against its family reference on Modal L40S.

    uv run modal run validate/check_oracle.py --fam fused_silu_fp8

Reads reference.py from the first subset_final/<fam>* task and the oracle from
common/oracles/<fam>.py, then runs correctness (vs fp32 GT) + speedup vs torch.compile.
"""
import os
import glob
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0", "numpy")
)
app = modal.App("check-oracle", image=image)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.function(gpu="L40S", timeout=1200)
def evaluate(ref_src: str, gen_src: str):
    import torch, importlib.util, sys, math  # noqa

    def load_module(src, name):
        path = f"/tmp/{name}.py"
        open(path, "w").write(src)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def bench(fn, warmup=25, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sorted(ts)[iters // 2]

    out = {"correct": False, "speedup_vs_compile": None, "speedup_vs_eager": None,
           "max_err": None, "error": None}
    try:
        refmod = load_module(ref_src, "kb_ref")
        Model, get_inputs = refmod.Model, refmod.get_inputs
        genmod = load_module(gen_src, "kb_gen")
        ModelNew = genmod.ModelNew

        ok = True; max_err = 0.0; cos_min = 1.0; rel_max = 0.0
        for _ in range(5):
            inputs = get_inputs()
            gt_inputs = [t.float() if t.is_floating_point() else t for t in inputs]
            gt = Model().cuda()(*gt_inputs)
            y = ModelNew().cuda()(*inputs)
            max_err = max(max_err, (y.float() - gt).abs().max().item())
            ok = ok and bool(torch.allclose(y.float(), gt, atol=2e-2, rtol=2e-2))
            yf, gf = y.float().flatten(), gt.flatten()
            cos_min = min(cos_min, torch.nn.functional.cosine_similarity(yf, gf, dim=0).item())
            rel_max = max(rel_max, (yf - gf).abs().sum().item() / (gf.abs().sum().item() + 1e-8))
        out["correct"], out["max_err"] = ok, max_err
        out["cosine_min"], out["rel_l1_max"] = cos_min, rel_max
        out["cosine_pass"] = bool(cos_min >= 0.99 and rel_max <= 0.05)

        ref = Model().cuda(); mnew = ModelNew().cuda(); inputs = get_inputs()
        t_eager = bench(lambda: ref(*inputs))
        try:
            compiled = torch.compile(lambda *a: ref(*a), mode="max-autotune")
            compiled(*inputs)
            t_compile = bench(lambda: compiled(*inputs))
        except Exception:
            t_compile = t_eager
        t_new = bench(lambda: mnew(*inputs))
        out["speedup_vs_compile"] = t_compile / t_new
        out["speedup_vs_eager"] = t_eager / t_new
    except Exception as exc:
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1800:]}"
    return out


@app.local_entrypoint()
def main(fam: str, subdir: str = "subset_final"):
    task = sorted(glob.glob(os.path.join(REPO, subdir, fam + "*")))
    assert task, f"no {subdir} task for {fam}"
    ref_src = open(os.path.join(task[0], "environment", "reference.py")).read()
    orc = sorted(glob.glob(os.path.join(REPO, "common", "oracles", fam.split("__")[0] + "*.py")))
    gen_src = open(orc[0]).read()
    r = evaluate.remote(ref_src, gen_src)
    print("\n==== ORACLE CHECK:", fam, "====")
    for k, v in r.items():
        print(f"  {k}: {v}")
    ok = r.get("correct") and (r.get("speedup_vs_compile") or 0) >= 1.0
    print("  => PASS" if ok else "  => FAIL")
