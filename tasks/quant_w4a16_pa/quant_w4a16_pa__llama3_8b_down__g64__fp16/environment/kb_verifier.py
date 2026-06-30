"""
Shared verifier for KernelBench-Harbor tasks. Runs INSIDE the task container.

Usage (called by tests/test.sh):
    python kb_verifier.py <reference.py> <solution.py> <config.json> <reward_out.json>

Grades a model-generated ModelNew against a naive PyTorch reference `Model`:
  - anti-reward-hack static check (real @triton.jit; no sdpa/flash_attn/cublas offload)
  - correctness vs fp32 ground truth over several random input draws (dtype-aware tolerance)
  - performance: median runtime vs torch.compile(reference, max-autotune) baseline (eager fallback)
  - reward = 1 iff correct AND speedup_vs_compile >= threshold AND static check passes

Writes reward.json: {"reward":0/1, "correct":bool, "speedup_vs_compile":float, ...}
"""
import sys
import os
import glob
import json
import importlib.util


def resolve_solution(sol_path):
    """Be robust to agents that wrote the solution to a different path/name:
    if the expected file is missing, search common dirs for a .py defining ModelNew."""
    if os.path.exists(sol_path):
        return sol_path
    seen = set()
    for d in ["/app", "/workspace", os.path.dirname(sol_path) or "."]:
        for c in sorted(glob.glob(os.path.join(d, "*.py")) + glob.glob(os.path.join(d, "**", "*.py"), recursive=True)):
            if c in seen:
                continue
            seen.add(c)
            try:
                if "class ModelNew" in open(c).read():
                    return c
            except Exception:
                pass
    return sol_path  # original (will error clearly if truly absent)


import ast
import io
import tokenize

# Attribute tails that mean "core compute was offloaded to a library kernel" (cuBLAS /
# cuDNN / fused attention). A real Triton kernel uses tl.dot etc., never these.
_BANNED_ATTRS = {
    "matmul", "mm", "bmm", "addmm", "einsum", "softmax", "linear",
    "scaled_dot_product_attention", "sdpa", "conv1d", "conv2d", "conv3d",
}
# Bare imported names called directly, e.g. `from torch.nn.functional import softmax`.
_BANNED_NAMES = {"scaled_dot_product_attention", "sdpa"}
# Vendor fused-attention / BLAS libraries — banned at import.
_BANNED_IMPORTS = {"flash_attn", "xformers"}


def _strip_code(src: str) -> str:
    """Source with comments and string literals removed — used only as a fallback when
    the AST won't parse, so a banned token mentioned in a comment/string can't trip us."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
        return " ".join(out)
    except Exception:
        return src


def static_check(src: str):
    """Reject submissions that don't do the core compute in Triton. Allows torch for
    allocation / launch / shape only. (The 'must beat torch.compile' gate is the real
    anti-hack backstop — e.g. x @ w just calls cuBLAS and ties the baseline.)

    AST-based: a banned op only fails if it's a real *use* (call / attribute access /
    import), not a mention in a comment or string. Some models echo the instruction's
    banned-op list into a comment in their solution; a naive substring grep would falsely
    reject those (observed: it hit Gemini almost exclusively)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # Can't parse (will fail at import during correctness anyway) — fall back to a
        # comment/string-stripped substring scan so we still catch blatant offloads.
        code = _strip_code(src)
        if "triton.jit" not in code:
            return False, "no @triton.jit kernel"
        for b in ["scaled_dot_product_attention", "flash_attn", "xformers",
                  "torch.matmul", "torch.mm", "torch.bmm", "torch.addmm", "torch.einsum",
                  "torch.softmax", "F.linear", "F.softmax", "torch.conv", "F.conv",
                  "torch.nn.functional.linear", "torch.nn.functional.softmax"]:
            if b in code:
                return False, f"banned op (offloads core compute): {b}"
        return True, "ok (fallback scan)"

    has_jit = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                d = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(d, ast.Attribute) and d.attr == "jit":
                    has_jit = True
                elif isinstance(d, ast.Name) and d.id == "jit":
                    has_jit = True
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _BANNED_IMPORTS:
                    return False, f"banned import (vendor fused op): {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _BANNED_IMPORTS:
                return False, f"banned import (vendor fused op): {node.module}"
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return False, f"banned op (offloads core compute): .{node.attr}"
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                return False, f"banned op (offloads core compute): {node.id}"
    if not has_jit:
        return False, "no @triton.jit kernel"
    return True, "ok"


def load_module(path, name):
    # Import from a real file so Triton's @triton.jit can read source (inspect.getsource).
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench(fn, warmup, iters):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def grade(ref_path, sol_path, cfg):
    import torch
    atol = cfg.get("atol", 2e-2)
    rtol = cfg.get("rtol", 2e-2)
    threshold = cfg.get("speedup_threshold", 1.0)
    n_correct = cfg.get("num_correctness_trials", 5)
    warmup = cfg.get("warmup", 25)
    iters = cfg.get("perf_trials", 100)

    metric = cfg.get("metric", "allclose")          # "allclose" or "cosine" (per family)
    cos_min = cfg.get("cosine_min", 0.99)
    rel_max = cfg.get("rel_l1_max", 0.05)
    suspicious_x = cfg.get("suspicious_speedup", 3.0)

    out = {"reward": 0, "correct": False, "speedup_vs_compile": None,
           "speedup_vs_eager": None, "max_err": None, "static_ok": False,
           "compile_ok": None, "suspicious": False, "error": None}

    sol_src = open(sol_path).read()
    ok, reason = static_check(sol_src)
    out["static_ok"] = ok
    if not ok:
        out["error"] = f"static check failed: {reason}"
        return out

    def is_correct(y, gt):
        if metric == "cosine":
            yf, gf = y.float().flatten(), gt.flatten()
            cos = torch.nn.functional.cosine_similarity(yf, gf, dim=0).item()
            rel = (yf - gf).abs().sum().item() / (gf.abs().sum().item() + 1e-8)
            return cos >= cos_min and rel <= rel_max
        return torch.allclose(y.float(), gt, atol=atol, rtol=rtol)

    try:
        refmod = load_module(ref_path, "kb_ref")
        Model, get_inputs = refmod.Model, refmod.get_inputs
        genmod = load_module(sol_path, "kb_sol")
        ModelNew = genmod.ModelNew

        ref = Model().cuda()
        try:
            mnew = ModelNew().cuda()
        except TypeError as e:
            out["error"] = f"ModelNew() must take no constructor args: {e}"
            return out

        max_err, all_correct = 0.0, True
        for _ in range(n_correct):
            inputs = get_inputs()                       # fresh draw each trial (values + jittered shape)
            gt_inputs = [t.float() if t.is_floating_point() else t for t in inputs]
            gt = Model().cuda()(*gt_inputs)
            y = mnew(*inputs)
            if y.shape != gt.shape:
                all_correct = False
                out["error"] = f"shape mismatch {tuple(y.shape)} vs {tuple(gt.shape)}"
                break
            max_err = max(max_err, (y.float() - gt).abs().max().item())
            if not is_correct(y, gt):
                all_correct = False
        out["max_err"], out["correct"] = max_err, all_correct

        inputs = get_inputs()
        t_eager = bench(lambda: ref(*inputs), warmup, iters)
        try:
            compiled = torch.compile(lambda *a: ref(*a), mode="max-autotune")
            compiled(*inputs)
            t_compile = bench(lambda: compiled(*inputs), warmup, iters)
            out["compile_ok"] = True
        except Exception:
            t_compile = t_eager
            out["compile_ok"] = False
        t_new = bench(lambda: mnew(*inputs), warmup, iters)
        out["speedup_vs_compile"] = t_compile / t_new
        out["speedup_vs_eager"] = t_eager / t_new
        # roofline sanity: implausibly large speedups are flagged for manual review
        out["suspicious"] = out["speedup_vs_compile"] > suspicious_x
        out["reward"] = int(all_correct and out["speedup_vs_compile"] >= threshold)
    except Exception as exc:
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}"
    return out


if __name__ == "__main__":
    ref_path, sol_path, cfg_path, reward_out = sys.argv[1:5]
    sol_path = resolve_solution(sol_path)
    result = grade(ref_path, sol_path, json.load(open(cfg_path)))
    with open(reward_out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["reward"] == 1 else 1)
