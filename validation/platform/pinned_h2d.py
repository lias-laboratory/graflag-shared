"""Do host-to-GPU copies from pinned memory arrive intact?

Batches of 128 float32 tensors, 20..170 square (the sizes of GeneralDyG's
batches), each pinned and copied with a plain synchronous .to("cuda"), then
copied back and compared with the original. Repeated with pageable memory as
the control. No method code is involved.
"""
import sys
import torch

torch.manual_seed(0)
dev = torch.device("cuda:0")
batches = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def trial(pin):
    bad_batches = bad_tensors = 0
    for _ in range(batches):
        host = [torch.rand(n, n) for n in torch.randint(20, 171, (128,)).tolist()]
        src = [t.pin_memory() for t in host] if pin else host
        moved = [t.to(dev) for t in src]
        wrong = sum(not torch.equal(m.cpu(), h) for m, h in zip(moved, host))
        bad_tensors += wrong
        bad_batches += bool(wrong)
    return bad_batches, bad_tensors


print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)
for pin in (True, False):
    b, t = trial(pin)
    print(f"  {'pinned  ' if pin else 'pageable'}: {b} of {batches} batches corrupted, {t} of {batches * 128} tensors", flush=True)
