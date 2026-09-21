"""
LDPC encoder/decoder using a pyldpc-derived parity-check matrix (PCM) together
with NVIDIA Sionna's LDPC belief-propagation decoder.

This implementation generates a custom LDPC code (H and G matrices) using
algorithms derived from pyldpc, allowing arbitrary code lengths and rates.
Decoding is performed by Sionna's generic LDPCBPDecoder, which supports
iterative soft-decision belief-propagation decoding on user-supplied parity-
check matrices and can run efficiently on GPU via TensorFlow.

Flow:
    message bits → LDPC encode → random bit flips (BSC)
    → LLR generation → Sionna BP decode → error statistics

References
----------
pyldpc:
    https://github.com/hichamjanati/pyldpc

Derived components:
    make_random_ldpc_H  <- parity_check_matrix
    _gf2_generator      <- coding_matrix_systematic
    _gaussjordan        <- gaussjordan
    _binaryproduct      <- binaryproduct

Sionna:
    https://github.com/NVlabs/sionna
"""
import time
import numpy as np
import torch
from sionna.phy.fec.ldpc import LDPCBPDecoder   # CHANGED: generic BP decoder (was LDPC5GDecoder)
                                                  # REMOVED: LDPC5GEncoder (replaced by GF2 encoder)


# ── GF(2) helper: derive systematic G from any H ─────────────────────────────

def _gf2_generator(H):
    """
    Reduce H over GF(2) to find a systematic generator matrix G (k × n).

    Returns
    -------
    G         : np.ndarray (k, n) uint8  — generator matrix
    free_cols : list[int]               — systematic (info-bit) column positions;
                                          c[free_cols] == v after encoding
    """
    H_r = H.copy().astype(np.uint8)
    m, n = H_r.shape
    pivot_cols, pr = [], 0

    for col in range(n):
        if pr >= m:
            break
        found = next((r for r in range(pr, m) if H_r[r, col]), -1)
        if found == -1:
            continue
        H_r[[pr, found]] = H_r[[found, pr]]
        for r in range(m):
            if r != pr and H_r[r, col]:
                H_r[r] = (H_r[r] + H_r[pr]) % 2
        pivot_cols.append(col)
        pr += 1

    free_cols = sorted(set(range(n)) - set(pivot_cols))
    k = len(free_cols)
    G = np.zeros((k, n), dtype=np.uint8)
    for i, fc in enumerate(free_cols):
        G[i, fc] = 1
        for j, pc in enumerate(pivot_cols):
            G[i, pc] = H_r[j, fc]
    return G, free_cols


# ── Convenience: build a random Gallager-style sparse H ──────────────────────

def make_random_ldpc_H(n, k_target, col_weight=3, seed=0):
    """
    Build a sparse (n − k_target) × n parity-check matrix with `col_weight`
    ones per column.  The actual k = n − rank(H) may differ slightly from
    k_target when H is rank-deficient; run_ldpc_custom_pcm reports the true k.
    """
    rng = np.random.default_rng(seed)
    m = n - k_target
    H = np.zeros((m, n), dtype=np.uint8)
    for col in range(n):
        rows = rng.choice(m, size=min(col_weight, m), replace=False)
        H[rows, col] = 1
    return H[H.any(axis=1)]          # drop degenerate all-zero rows


# ── Main function ─────────────────────────────────────────────────────────────

def run_ldpc_custom_pcm(
    H,                               # CHANGED: custom PCM (m × n, uint8) replaces (message_bits, code_bits)
    flip_ratio=0.05,
    num_iter=50,                     # CHANGED: default raised; low-rate codes need more BP iterations
    printer=False,
):
    """
    Encode a random message with a custom LDPC parity-check matrix H,
    corrupt the codeword over a BSC, decode with Sionna's belief-propagation
    decoder, and return the number of message-bit errors.

    Parameters
    ----------
    H          : np.ndarray (m, n) uint8 — parity-check matrix
    flip_ratio : float  — fraction of codeword bits flipped (BSC crossover prob)
    num_iter   : int    — number of BP decoding iterations
    printer    : bool   — print diagnostics if True

    Returns
    -------
    error : int — message bits decoded incorrectly
    """
    # CHANGED: derive encoder from H via GF(2) reduction (replaces LDPC5GEncoder)
    st_time = time.time()
    G, free_cols = _gf2_generator(H)
    k = len(free_cols)
    n = H.shape[1]

    if printer:
        print(f"N={n}  K={k}  R={k/n:.4f}")
        print(f"G generation time: {time.time()- st_time}")

    # CHANGED: encode via G over GF(2) (replaces encoder(torch.tensor(v)))
    v = np.random.randint(2, size=k).astype(np.uint8)
    c = (v @ G) % 2                              # shape [n], uint8

    # flip bits  ← identical to original
    c_torch = torch.tensor(c, dtype=torch.float32)
    flip_mask = torch.rand(n) < flip_ratio
    c_torch[flip_mask] = 1.0 - c_torch[flip_mask]

    if printer:
        print(f"Bits flipped: {flip_mask.sum()} / {n}  ({flip_mask.float().mean()*100:.1f}%)")

    # bits → LLR  ← identical to original
    llr = -(1.0 - 2.0 * c_torch) * 10.0         # shape [n]

    # CHANGED: LDPCBPDecoder with custom H (replaces LDPC5GDecoder)
    # Returns full codeword of shape [n]; we extract info bits via free_cols.

    decoder = LDPCBPDecoder(H, hard_out=True, num_iter=num_iter)
    c_hat = decoder(llr.unsqueeze(0)).squeeze(0) # shape [n]

    # CHANGED: extract info bits from systematic positions (replaces return_infobits=True)
    u_hat = c_hat[free_cols]                     # shape [k]

    error = int((v != u_hat.cpu().numpy().astype(np.uint8)).sum())
    if printer:
        print(f"Bit errors after decoding: {error} / {k}")
    return error


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _assert(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  PASS: {msg}")


def test_orthogonality():
    """G · H^T == 0 over GF(2)."""
    H = make_random_ldpc_H(n=300, k_target=30, col_weight=3, seed=1)
    G, _ = _gf2_generator(H)
    _assert(np.all((G @ H.T) % 2 == 0), f"GH^T = 0  (shape G={G.shape}, H={H.shape})")


def test_systematic_positions():
    """c[free_cols] == v (encoding is systematic)."""
    H = make_random_ldpc_H(n=300, k_target=30, col_weight=3, seed=2)
    G, free_cols = _gf2_generator(H)
    v = np.random.randint(0, 2, size=len(free_cols), dtype=np.uint8)
    c = (v @ G) % 2
    _assert(np.all(c[free_cols] == v), "c[free_cols] == v")


def test_valid_codeword():
    """H · c^T == 0 over GF(2) for encoded codeword."""
    H = make_random_ldpc_H(n=300, k_target=30, col_weight=3, seed=3)
    G, _ = _gf2_generator(H)
    v = np.random.randint(0, 2, size=G.shape[0], dtype=np.uint8)
    c = (v @ G) % 2
    _assert(np.all((H @ c) % 2 == 0), "H · c^T = 0")


def test_zero_errors_no_noise():
    """flip_ratio=0 must yield 0 errors."""
    H = make_random_ldpc_H(n=300, k_target=30, col_weight=3, seed=4)
    errors = run_ldpc_custom_pcm(H, flip_ratio=0.0, num_iter=20)
    _assert(errors == 0, "0 errors with no noise")


def test_rate_approx_one_tenth():
    """Effective rate should be near 1/10 when k_target = n/10."""
    H = make_random_ldpc_H(n=500, k_target=50, col_weight=3, seed=5)
    _, free_cols = _gf2_generator(H)
    rate = len(free_cols) / H.shape[1]
    _assert(0.08 <= rate <= 0.15, f"Rate ≈ 1/10  (got {rate:.4f})")


def test_low_flip_decodes_correctly():
    """10 trials at 1% flip rate → 0 total message-bit errors."""
    H = make_random_ldpc_H(n=500, k_target=50, col_weight=3, seed=6)
    total = sum(run_ldpc_custom_pcm(H, flip_ratio=0.01, num_iter=100) for _ in range(10))
    _assert(total == 0, f"0 errors over 10 trials at flip=1%  (got {total})")

def test_medium_flip_decodes_correctly():
    """10 trials at 10% flip rate → 0 total message-bit errors."""
    H = make_random_ldpc_H(n=500, k_target=50, col_weight=3, seed=6)
    total = sum(run_ldpc_custom_pcm(H, flip_ratio=0.1, num_iter=100) for _ in range(10))
    _assert(total == 0, f"0 errors over 10 trials at flip=10%  (got {total})")

def test_high_flip_has_errors():
    """40% flip rate must produce some errors (sanity / non-trivial decoder)."""
    np.random.seed(77)
    H = make_random_ldpc_H(n=300, k_target=30, col_weight=3, seed=7)
    results = [run_ldpc_custom_pcm(H, flip_ratio=0.40, num_iter=20) for _ in range(5)]
    _assert(any(e > 0 for e in results), f"Some errors expected at flip=40%  (got {results})")


def run_all_tests():
    tests = [
        test_orthogonality,
        test_systematic_positions,
        test_valid_codeword,
        test_zero_errors_no_noise,
        test_rate_approx_one_tenth,
        test_low_flip_decodes_correctly,
        test_medium_flip_decodes_correctly,
        test_high_flip_has_errors,
    ]
    print("=" * 60)
    print("Tests for run_ldpc_custom_pcm  (Sionna LDPCBPDecoder)")
    print("=" * 60)
    passed = failed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t(); passed += 1
        except AssertionError as e:
            print(f"  {e}"); failed += 1
        except Exception as e:
            print(f"  ERROR: {e}"); failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

    fratio = 0.25
    n = 50000
    k = 1000

    print(f"\n── Demo: rate≈1/10, flip={fratio*100:.1f}%, printer=True ──")
    H_demo = make_random_ldpc_H(n=n, k_target=k, col_weight=3, seed=99)
    run_ldpc_custom_pcm(H_demo, flip_ratio=fratio, num_iter=200, printer=True)
