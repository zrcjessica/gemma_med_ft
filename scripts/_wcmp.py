import subprocess, sys, torch, transformers

base = subprocess.check_output(
    ["bash", "-c", "source scripts/_resolve_model.sh; resolve_gemma 4b it"]
).decode().strip()
ck = sys.argv[1] if len(sys.argv) > 1 else "outputs/4b/smoke_hfsave_lr1e-5_25713728/checkpoint-1"

b = transformers.Gemma3ForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
c = transformers.Gemma3ForCausalLM.from_pretrained(ck, dtype=torch.bfloat16)
bs, cs = dict(b.named_parameters()), dict(c.named_parameters())
print("RESULT param_count base=%d ckpt=%d same_keys=%s" % (len(bs), len(cs), set(bs) == set(cs)))

tot_n = tot_d = 0.0
worst = []
for k in bs:
    if k not in cs:
        continue
    d = (cs[k].float() - bs[k].float()).norm().item()
    n = bs[k].float().norm().item()
    tot_n += d * d
    tot_d += n * n
    if n > 0:
        worst.append((d / n, k))
worst.sort(reverse=True)
print("RESULT global_rel_frobenius_diff = %.4e" % ((tot_n ** 0.5) / (tot_d ** 0.5)))
print("RESULT largest per-tensor relative diffs:")
for r, k in worst[:5]:
    print("RESULT   %.4e  %s" % (r, k))
