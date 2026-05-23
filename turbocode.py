"""
turbo_code.py
=============
Turbo Code encoder + decoder with pluggable low-rate extensions.

Sections
--------
  1.  RSC Encoder          — the building block of every turbo code
  2.  BCJR Decoder         — optimal soft-input soft-output decoder
  3.  Turbo Encoder        — two RSC encoders in parallel
  4.  Turbo Decoder        — iterative BCJR exchange
  5.  Repetition Layer     — repeat each coded bit R times  (lowers rate to 1/(3R))
  6.  Puncturing Layer     — delete selected parity bits    (raises rate toward 1/2)
  7.  Channel Simulation   — BPSK + AWGN → LLRs
  8.  Utilities            — polynomial helpers, log-sum-exp
  9.  Demos                — run_demo_standard()  and  run_demo_lowrate()

Rate cheat-sheet
----------------
  Standard turbo (no repetition, no puncturing) : rate = 1/3  ≈ 33 %
  Repetition factor R=2                          : rate = 1/6  ≈ 17 %
  Repetition factor R=3                          : rate = 1/9  ≈ 11 %
  Repetition factor R=5                          : rate = 1/15 ≈  7 %
  Repetition factor R=10                         : rate = 1/30 ≈  3 %
  Puncture half the parities                     : rate = 1/2  ≈ 50 %

Quick-start
-----------
  python turbo_code.py            # runs both demos
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1.  RSC ENCODER
#     Recursive Systematic Convolutional encoder.
#     "Recursive"   = feedback path: the output is fed back into the shift reg.
#     "Systematic"  = the input bit always appears verbatim in the output.
#     "Convolutional" = each output depends on current + past K-1 input bits.
# ══════════════════════════════════════════════════════════════════════════════

class RSCEncoder:
    """
    Rate-1/2 RSC encoder.

    Internal state
    --------------
    state : (m,) int array  — the shift register (m = memory depth = K-1)
                              state[0] = most recently entered bit (newest)
                              state[m-1] = oldest stored bit

    Generator polynomials (octal notation)
    ---------------------------------------
    g_feedback  selects which register taps are XOR-ed back with the input.
    g_forward   selects which register taps contribute to the parity output.

    Both are written in octal where each binary digit = one tap:
      0o13 = binary 1011 → taps at positions 0, 1, 3  (position 0 = current)
      0o15 = binary 1101 → taps at positions 0, 2, 3

    Preset options by constraint length K (memory m = K-1, states = 2^m)
    -----------------------------------------------------------------------
      K=3 (m=2, 4  states) : RSCEncoder(0o7,  0o5)   — fast, weaker
      K=4 (m=3, 8  states) : RSCEncoder(0o13, 0o15)  — LTE standard  ← default
      K=5 (m=4, 16 states) : RSCEncoder(0o23, 0o35)  — stronger floor
      K=6 (m=5, 32 states) : RSCEncoder(0o53, 0o75)  — deep-space quality
    Larger K → higher minimum distance → lower error floor, but 2× more states
    per extra bit of memory (decoder complexity doubles per K increase).
    """

    def __init__(self, g_feedback=0o13, g_forward=0o15):
        # FIX 1: Store the original polynomial values so callers and demos
        # can inspect/print them without hardcoding magic literals.
        self.g_feedback = g_feedback
        self.g_forward  = g_forward

        # Convert octal polynomial → array of binary tap coefficients (MSB first)
        # e.g. 0o13 = 0b1011 → [1, 0, 1, 1]
        self.fb_taps = _poly2taps(g_feedback)   # feedback taps  (length K)
        self.ff_taps = _poly2taps(g_forward)    # feedforward taps (length K)

        # m = number of flip-flops (delay elements) in the shift register
        # = K - 1  because the first tap (index 0) is the *current* input
        self.m = len(self.fb_taps) - 1

        # Total number of distinct encoder states = 2^m
        # The BCJR trellis has one node per state per time step
        self.n_states = 2 ** self.m

    # ──────────────────────────────────────────────────────────────────────────
    def encode(self, bits):
        """
        Encode a bit array through the RSC.

        At each time step k:
          1. Compute feedback:  fb = input XOR (feedback-tapped register bits)
          2. Compute parity  :  par= feedforward-tapped [fb || register] bits XOR'd
          3. Shift register  :  push fb into position 0, old bits shift right

        Variables
        ---------
        u   : current input bit (scalar int, 0 or 1)
        fb  : feedback value pushed into the shift register (scalar int)
        reg : [fb, state[0], state[1], ...] — full register view for parity calc

        Returns
        -------
        systematic : (N,) — input bits verbatim (RSC is systematic by definition)
        parity     : (N,) — parity bits
        """
        N      = len(bits)
        state  = np.zeros(self.m, dtype=np.int32)  # shift register, all-zero start
        parity = np.zeros(N,      dtype=np.int32)

        for k in range(N):
            u  = int(bits[k])

            # Feedback: XOR input with selected existing register taps
            # fb_taps[1:] skips tap[0] which corresponds to the current input
            # itself (it's handled separately to close the loop)
            fb = u ^ int(np.dot(self.fb_taps[1:], state) % 2)

            # Parity: feedforward polynomial over [fb, current register contents]
            # reg = full shift register *after* the feedback bit enters
            reg       = np.concatenate(([fb], state))
            parity[k] = int(np.dot(self.ff_taps, reg) % 2)

            # Advance shift register: roll right (oldest bit falls off), insert fb
            state    = np.roll(state, 1)
            state[0] = fb

        return bits.copy(), parity

    # ──────────────────────────────────────────────────────────────────────────
    def build_trellis(self):
        """
        Pre-compute the full state-transition table (trellis) for BCJR.

        The trellis maps every (state, input_bit) pair to:
          - the next state
          - the systematic output (= input bit, trivially)
          - the parity output

        Pre-computing this once avoids repeating the polynomial arithmetic
        inside the inner loop of the BCJR algorithm (called N × S × 2 times).

        Variables
        ---------
        s  : current state integer (0 … S-1)
        sb : s expressed as a bit array of length m (MSB first)
        u  : input bit (0 or 1)
        fb : feedback bit computed from (u, sb)
        nb : next state as a bit array (= roll sb right, insert fb at [0])

        Returns
        -------
        next_state : (S, 2) int  — next_state[s, u] = state after input u from s
        out_sys    : (S, 2) int  — systematic output (always = u)
        out_par    : (S, 2) int  — parity output
        """
        S          = self.n_states
        next_state = np.zeros((S, 2), dtype=np.int32)
        out_sys    = np.zeros((S, 2), dtype=np.int32)
        out_par    = np.zeros((S, 2), dtype=np.int32)

        for s in range(S):
            sb = _int2bits(s, self.m)          # current state as bits
            for u in range(2):
                fb  = u ^ int(np.dot(self.fb_taps[1:], sb) % 2)
                reg = np.concatenate(([fb], sb))
                par = int(np.dot(self.ff_taps, reg) % 2)
                nb  = np.roll(sb, 1);  nb[0] = fb  # next state bits

                next_state[s, u] = _bits2int(nb)
                out_sys[s, u]    = u    # systematic: output always = input
                out_par[s, u]    = par

        return next_state, out_sys, out_par


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BCJR (MAP) DECODER  — log domain, unterminated trellis
#
#     BCJR computes the exact posterior LLR for each bit given all received
#     samples.  It runs two passes over the trellis:
#       α (forward)  — probability of being in state s after processing bits 0…k
#       β (backward) — probability of the remaining bits k+1…N given state s at k
#     Combined with the branch metric γ they give the full posterior.
#
#     We work in log domain throughout to avoid numerical underflow.
# ══════════════════════════════════════════════════════════════════════════════

def bcjr_decode(llr_sys, llr_par, next_state, out_sys, out_par,
                llr_apriori=None):
    """
    Log-domain BCJR decoder (unterminated trellis).

    Unterminated trellis
    --------------------
    The shift register is NOT flushed to zero at the end of the block.
    → We don't know the final state.
    → β is initialised uniformly: log β(N, s) = 0 for all s.
    (If the trellis were terminated, only log β(N, 0) = 0, rest = -∞.)

    LLR sign convention
    -------------------
      LLR(k) > 0  →  bit k is more likely 0
      LLR(k) < 0  →  bit k is more likely 1

    BPSK mapping
    ------------
      bit 0 → transmitted as +1
      bit 1 → transmitted as -1
      received sample y = (1 − 2·bit) + noise
      channel LLR = 2·y / σ²

    Branch metric γ
    ---------------
      γ(k, s, u) = ½·La(u)·(1−2u)  +  ½·Lc_sys·(1−2u)  +  ½·Lc_par·(1−2·par)
      The ½ factor prevents double-counting because α and β each accumulate
      one half of the evidence independently.

    Parameters
    ----------
    llr_sys     : (N,) channel LLR for the systematic stream
    llr_par     : (N,) channel LLR for the parity stream
    next_state  : (S,2) trellis next-state table from build_trellis()
    out_sys     : (S,2) systematic output table  (not used numerically here)
    out_par     : (S,2) parity output table
    llr_apriori : (N,) a-priori LLR from the partner decoder.
                  Pass None or zeros on the first half-iteration.

    Variables
    ---------
    N           : block length (number of info bits)
    S           : number of trellis states
    log_alpha   : (N+1, S) forward log-probabilities  α[k, s]
    log_beta    : (N+1, S) backward log-probabilities β[k, s]
    prev_state  : (S, 2)   inverse trellis — which state led to state s?
    prev_input  : (S, 2)   which input bit caused that transition?
    p0, p1      : lists of path log-probabilities for u=0 and u=1 at time k

    Returns
    -------
    llr_extrinsic : (N,) extrinsic LLR  =  total − channel_sys − a_priori
                    Pass this to the partner decoder as its a-priori input.
    llr_total     : (N,) posterior LLR.
                    Use (llr_total < 0).astype(int) for hard decisions.
    """
    N = len(llr_sys)
    S = next_state.shape[0]

    if llr_apriori is None:
        llr_apriori = np.zeros(N)

    NEG_INF = -1e30   # stand-in for log(0); large enough to never dominate

    # ── Build inverse trellis ─────────────────────────────────────────────────
    # For each destination state ns_, record which (source state, input) caused it.
    # Each state has exactly 2 incoming transitions (binary trellis).
    prev_state = -np.ones((S, 2), dtype=np.int32)   # -1 = slot not yet filled
    prev_input  = -np.ones((S, 2), dtype=np.int32)
    for s in range(S):
        for u in range(2):
            ns_ = next_state[s, u]
            filled = False
            for slot in range(2):
                if prev_state[ns_, slot] == -1:
                    prev_state[ns_, slot] = s
                    prev_input[ns_, slot] = u
                    filled = True
                    break
            # FIX 3: Detect degenerate trellis (> 2 predecessors for one state)
            # instead of silently corrupting results via numpy's negative indexing.
            if not filled:
                raise ValueError(
                    f"State {ns_} has more than 2 predecessors — "
                    f"degenerate trellis produced by the RSC polynomials. "
                    f"Check g_feedback / g_forward."
                )

    # ── Branch metric γ ───────────────────────────────────────────────────────
    def branch_metric(k, s, u):
        sign_u   = 1 - 2 * u              # bit 0 → +1, bit 1 → -1  (BPSK sign)
        sign_par = 1 - 2 * out_par[s, u]  # parity bit sign
        return (0.5 * llr_apriori[k] * sign_u    # a-priori contribution
                + 0.5 * llr_sys[k]   * sign_u    # channel systematic
                + 0.5 * llr_par[k]   * sign_par) # channel parity

    # ── Forward sweep α ───────────────────────────────────────────────────────
    # log_alpha[k, s] = log P(received y_0…y_{k-1}, in state s at time k)
    log_alpha = np.full((N + 1, S), NEG_INF)
    log_alpha[0, 0] = 0.0    # encoder starts in state 0 with certainty

    for k in range(N):
        for ns_ in range(S):
            # Accumulate contributions from all predecessor states
            vals = []
            for slot in range(2):
                ps = prev_state[ns_, slot]
                if ps < 0:
                    continue    # unused slot
                u = prev_input[ns_, slot]
                vals.append(log_alpha[k, ps] + branch_metric(k, ps, u))
            if vals:
                log_alpha[k + 1, ns_] = _log_sum_exp_list(vals)

    # ── Backward sweep β ─────────────────────────────────────────────────────
    # log_beta[k, s] = log P(received y_k…y_{N-1} | in state s at time k)
    # Unterminated: all final states equally likely → log β(N, s) = log(1) = 0
    log_beta = np.zeros((N + 1, S))    # 0 = log(1) = uniform over final states

    for k in range(N - 1, -1, -1):
        for s in range(S):
            vals = [log_beta[k+1, next_state[s,u]] + branch_metric(k, s, u)
                    for u in range(2)]
            log_beta[k, s] = _log_sum_exp_list(vals)

    # ── Posterior LLR  =  log P(u=0|y) / P(u=1|y) ───────────────────────────
    llr_total = np.zeros(N)
    for k in range(N):
        p0, p1 = [], []    # log-probabilities for paths with u=0 and u=1
        for s in range(S):
            for u in range(2):
                # Full path probability: α × γ × β
                v = (log_alpha[k, s]
                     + branch_metric(k, s, u)
                     + log_beta[k + 1, next_state[s, u]])
                (p0 if u == 0 else p1).append(v)
        # LLR = log(sum of u=0 paths) - log(sum of u=1 paths)
        llr_total[k] = _log_sum_exp_list(p0) - _log_sum_exp_list(p1)

    # ── Extrinsic LLR ────────────────────────────────────────────────────────
    # Strip what the partner already knew (channel sys + its own a-priori).
    # This prevents circular reinforcement in the iterative exchange.
    llr_extrinsic = llr_total - llr_sys - llr_apriori

    return llr_extrinsic, llr_total


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TURBO ENCODER
#     Two RSC encoders in parallel, separated by an interleaver.
#     Rate = 1/3  (1 systematic bit + 2 parity bits per input bit).
# ══════════════════════════════════════════════════════════════════════════════

class TurboEncoder:
    """
    Parallel concatenated turbo encoder (rate 1/3).

    Variables
    ---------
    pi  : interleaver — a permutation array of length N.
          Encoder 2 sees the *same* information bits but in a shuffled order.
          Good interleavers spread burst errors; random permutations work well
          for N ≥ 256. For N < 256 use a structured interleaver (e.g. LTE QPP).
    """

    def __init__(self, encoder: RSCEncoder, interleaver: np.ndarray):
        self.enc = encoder
        self.pi  = interleaver   # permutation indices, shape (N,)

    def encode(self, bits):
        """
        Returns
        -------
        sys  : (N,) systematic bits  (= input, transmitted once)
        par1 : (N,) parity from encoder 1  (encoding original order)
        par2 : (N,) parity from encoder 2  (encoding interleaved order)
        """
        sys,  par1 = self.enc.encode(bits)
        _,    par2 = self.enc.encode(bits[self.pi])   # interleaved input
        return sys, par1, par2


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TURBO DECODER
#     Two BCJR sub-decoders exchanging extrinsic information iteratively.
#
#     Iteration structure:
#       ┌─ Dec1: uses ch_sys, ch_par1, La1 (extrinsic from Dec2 last round) ─┐
#       │  → ext1 (extrinsic)                                                 │
#       │  interleave ext1 → La2 for Dec2                                     │
#       │  Dec2: uses interleaved ch_sys, ch_par2, La2                        │
#       │  → ext2 (extrinsic in interleaved domain)                           │
#       └─ de-interleave ext2 → La1 for next round ──────────────────────────┘
#     After last round: de-interleave Dec2's posterior → hard decisions.
# ══════════════════════════════════════════════════════════════════════════════

class TurboDecoder:
    """
    Iterative turbo decoder.

    Variables
    ---------
    pi      : interleaver (same permutation used at the encoder)
    pi_inv  : de-interleaver = argsort(pi). Undoes the permutation.
    n_iter  : number of full decoder iterations.
              More iterations → better BER, but linear time cost.
              Typical values: 6 (fast), 8 (standard), 12 (near-optimal).
    La1     : a-priori LLR fed into decoder 1 at each iteration.
              Starts at zero (no prior knowledge); updated each round.
    llr_sys_int : channel systematic LLR in interleaved order.
                  Pre-computed once — decoder 2 always needs this.
    """

    def __init__(self, encoder: RSCEncoder, interleaver: np.ndarray, n_iter: int = 8):
        self.trellis = encoder.build_trellis()   # (next_state, out_sys, out_par)
        self.pi      = interleaver
        self.pi_inv  = np.argsort(interleaver)   # inverse permutation
        self.n_iter  = n_iter

    def decode(self, llr_sys, llr_par1, llr_par2, La1=None):
        """
        Parameters
        ----------
        llr_sys  : (N,) channel LLR for systematic stream
        llr_par1 : (N,) channel LLR for parity 1
        llr_par2 : (N,) channel LLR for parity 2

        Returns
        -------
        bits_hat : (N,) hard-decision decoded bits  (0 or 1)
        """
        ns, os_, op = self.trellis
        N            = len(llr_sys)
        llr_sys_int  = llr_sys[self.pi]     # interleaved sys for decoder 2

        if La1 is None:
            La1          = np.zeros(N)           # zero a-priori on first iteration

        for _ in range(self.n_iter):

            # Decoder 1: standard order
            ext1, _ = bcjr_decode(llr_sys, llr_par1, ns, os_, op,
                                   llr_apriori=La1)

            # Pass extrinsic to decoder 2 in interleaved domain
            La2 = ext1[self.pi]

            # Decoder 2: interleaved order
            ext2, llr_post_int = bcjr_decode(llr_sys_int, llr_par2, ns, os_, op,
                                             llr_apriori=La2)

            # De-interleave extrinsic from decoder 2 → a-priori for decoder 1
            La1 = ext2[self.pi_inv]

        # Final posterior in original (non-interleaved) bit order
        llr_final = llr_post_int[self.pi_inv]
        return (llr_final < 0).astype(np.int32)   # 0 if LLR > 0, else 1


# ══════════════════════════════════════════════════════════════════════════════
# 5.  REPETITION LAYER
#
#     Repeat each coded bit R times before transmission.
#     On reception, sum the R received LLRs for each bit (optimal combining).
#
#     Why summing LLRs is optimal
#     ---------------------------
#     For AWGN + BPSK, each received LLR_i = 2·y_i / σ².
#     Summing R independent copies of y is equivalent to MRC (Maximum Ratio
#     Combining).  The combined SNR is R × the single-copy SNR, giving an
#     effective gain of 10·log10(R) dB.
#
#     Effective rate with repetition
#     --------------------------------
#     Standard turbo rate = 1/3.
#     After repeating every bit R times: rate = 1 / (3·R).
#
#       R=1  → 1/3  ≈ 33%
#       R=2  → 1/6  ≈ 17%
#       R=3  → 1/9  ≈ 11%   ← ~10% target
#       R=5  → 1/15 ≈  7%   ← ~5% target
#       R=10 → 1/30 ≈  3%
# ══════════════════════════════════════════════════════════════════════════════

def repeat_encode(bits, R):
    """
    Repeat each bit R times.

    Example: bits=[0,1,0], R=3 → [0,0,0, 1,1,1, 0,0,0]

    Parameters
    ----------
    bits : (N,) binary array  — one of {sys, par1, par2}
    R    : int  — repetition factor

    Returns
    -------
    repeated : (N*R,) binary array
    """
    return np.repeat(bits, R)    # numpy broadcasts cleanly


def repeat_decode(llr_rx, R):
    """
    Combine R received LLR copies per original bit by summing.

    Parameters
    ----------
    llr_rx : (N*R,) LLR array from the channel
    R      : int — must match the repetition factor used at the encoder

    Variables
    ---------
    N           : number of original bits = len(llr_rx) // R
    reshaped    : (N, R) view — row k holds the R copies for original bit k

    Returns
    -------
    llr_combined : (N,) one combined LLR per original bit
                   Combined LLR has R× better effective SNR than a single copy.
    """
    N = len(llr_rx) // R
    return llr_rx.reshape(N, R).sum(axis=1)   # sum across the R copies (axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PUNCTURING LAYER
#
#     Puncturing *removes* selected parity bits before transmission, raising
#     the effective code rate above 1/3.  The decoder fills removed positions
#     with LLR = 0 (complete uncertainty = erasure).
#
#     This is how LTE achieves variable rates from a single 1/3 mother code.
#
#     Built-in patterns
#     -----------------
#     'rate_half'  : alternate par1/par2 each symbol → rate 1/2
#                    Even k: transmit par1[k], erase par2[k]
#                    Odd  k: erase par1[k], transmit par2[k]
#     'rate_2_5'   : keep 3 out of every 4 parity bits → rate 2/5
#     'none'       : no puncturing → rate 1/3  (passthrough)
#
#     Custom patterns
#     ---------------
#     Pass mask1, mask2 to puncture() directly:
#       mask1[k] = 1 → keep par1[k],  0 → drop it
#       mask2[k] = 1 → keep par2[k],  0 → drop it
# ══════════════════════════════════════════════════════════════════════════════

def puncture(par1, par2, pattern='rate_half'):
    """
    Remove parity bits according to a puncturing pattern.

    Parameters
    ----------
    par1, par2 : (N,) encoded parity arrays
    pattern    : str — 'rate_half', 'rate_2_5', or 'none'

    Variables
    ---------
    mask1 : (N,) bool/int — which par1 bits to *keep* (1=keep, 0=drop)
    mask2 : (N,) bool/int — which par2 bits to *keep*

    Returns
    -------
    par1_tx : kept par1 bits (shorter array)
    par2_tx : kept par2 bits (shorter array)
    mask1   : keep-mask for par1  (needed by depuncture)
    mask2   : keep-mask for par2
    """
    N = len(par1)

    if pattern == 'none':
        # No puncturing; transmit everything
        mask1 = np.ones(N, dtype=np.int32)
        mask2 = np.ones(N, dtype=np.int32)

    elif pattern == 'rate_half':
        # Alternate: par1 on even positions, par2 on odd positions
        # Transmits N parity bits instead of 2N → rate goes from 1/3 to 1/2
        mask1 = np.array([1, 0] * (N // 2), dtype=np.int32)  # keep even
        mask2 = np.array([0, 1] * (N // 2), dtype=np.int32)  # keep odd
        if N % 2:   # handle odd N
            mask1 = np.append(mask1, 1)
            mask2 = np.append(mask2, 0)

    elif pattern == 'rate_2_5':
        # Keep 3 out of every 4 parity bit positions
        # Effectively transmits 3/4 × 2N = 1.5N parity bits → rate = 2/5
        mask1 = np.tile([1, 1, 0, 1], int(np.ceil(N / 4)))[:N]
        mask2 = np.tile([1, 0, 1, 1], int(np.ceil(N / 4)))[:N]

    else:
        raise ValueError(f"Unknown puncturing pattern: '{pattern}'. "
                         f"Choose from 'none', 'rate_half', 'rate_2_5'.")

    par1_tx = par1[mask1 == 1]   # only the kept bits are transmitted
    par2_tx = par2[mask2 == 1]

    return par1_tx, par2_tx, mask1, mask2


def depuncture(par1_tx_llr, par2_tx_llr, mask1, mask2, N):
    """
    Re-insert erased positions as LLR = 0 before passing to the turbo decoder.

    LLR = 0 means "completely uncertain" — equivalent to a received erasure.
    The BCJR decoder handles this correctly; it simply gets no information
    from those positions, which is mathematically identical to not transmitting.

    Parameters
    ----------
    par1_tx_llr : channel LLRs for the *transmitted* par1 bits (shorter array)
    par2_tx_llr : channel LLRs for the *transmitted* par2 bits (shorter array)
    mask1       : (N,) keep-mask returned by puncture() for par1
    mask2       : (N,) keep-mask returned by puncture() for par2
    N           : original block length

    Variables
    ---------
    llr_par1 : (N,) full LLR array for par1 — erased positions set to 0
    llr_par2 : (N,) full LLR array for par2 — erased positions set to 0

    Returns
    -------
    llr_par1 : (N,) restored LLR array for par1
    llr_par2 : (N,) restored LLR array for par2
    """
    llr_par1 = np.zeros(N)   # 0 = LLR for erased (unknown) positions
    llr_par2 = np.zeros(N)

    llr_par1[mask1 == 1] = par1_tx_llr   # fill back the received values
    llr_par2[mask2 == 1] = par2_tx_llr

    return llr_par1, llr_par2


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CHANNEL SIMULATION  —  BPSK + AWGN → LLRs
# ══════════════════════════════════════════════════════════════════════════════

def awgn_llr(bits, snr_db, code_rate=1/3):
    """
    Simulate BPSK modulation over AWGN and return soft channel LLRs.

    Mapping   : bit 0 → +1,  bit 1 → -1
    Received  : y = (1 − 2·bit) + n,   n ~ N(0, σ²)
    LLR       : L = 2·y / σ²

    Parameters
    ----------
    bits      : binary array (any length)
    snr_db    : Eb/N0 in dB.  Eb = energy per *information* bit.
    code_rate : R = k/n.  Used to convert Eb/N0 → Es/N0.
                Set this to the *actual* transmitted rate so the SNR axis
                is fair across different rates.

    Variables
    ---------
    snr_lin   : Eb/N0 as a linear ratio (not dB)
    noise_var : σ² = N0/2 = 1 / (2·R·Eb/N0)  — AWGN noise power per dimension
    bpsk      : BPSK-modulated signal (+1 or -1)
    received  : noisy observation y

    Returns
    -------
    llr : (N,) channel LLRs.  LLR > 0 → likely bit 0, LLR < 0 → likely bit 1.
    """
    snr_lin   = 10 ** (snr_db / 10)
    noise_var = 1.0 / (2.0 * code_rate * snr_lin)   # σ²
    bpsk      = 1.0 - 2.0 * bits.astype(float)       # 0→+1, 1→-1
    received  = bpsk + np.random.randn(len(bits)) * np.sqrt(noise_var)
    return 2.0 * received / noise_var                 # soft channel LLR


def noiseless_llr(bits, L=50.0):
    """
    Convert hard bits → strong LLRs (noiseless channel)

    L: magnitude of confidence (avoid too large to prevent overflow)
    """
    bits = bits.astype(np.float64)
    return (1.0 - 2.0 * bits) * L


# ══════════════════════════════════════════════════════════════════════════════
# 8.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _poly2taps(poly_octal):
    """
    Convert an octal generator polynomial integer to a binary tap array.

    Example: 0o13 = 0b1011 → [1, 0, 1, 1]   (MSB first)
    The array length = number of binary digits = constraint length K.
    """
    return np.array([int(b) for b in bin(poly_octal)[2:]], dtype=np.int32)


def _int2bits(n, length):
    """
    Integer → fixed-length binary array, MSB first.

    Example: n=5, length=4 → [0, 1, 0, 1]
    Used to convert a trellis state integer into its register bit representation.
    """
    bits = np.zeros(length, dtype=np.int32)
    for i in range(length - 1, -1, -1):
        bits[i] = n & 1
        n >>= 1
    return bits


def _bits2int(bits):
    """
    Binary array (MSB first) → integer.

    Example: [0, 1, 0, 1] → 5
    Inverse of _int2bits.
    """
    return int(np.dot(bits, 1 << np.arange(len(bits) - 1, -1, -1)))


def _log_sum_exp_list(vals):
    """
    Numerically stable  log( Σ exp(v_i) )  over a Python list.

    Naive computation overflows for large values or underflows for large
    negatives.  The identity:
      log Σ exp(v_i) = a + log Σ exp(v_i − a),   a = max(v_i)
    keeps all exponent arguments ≤ 0, preventing overflow while keeping
    precision near 1.0.

    Variables
    ---------
    a : the maximum value in vals — subtracted to normalise the exponents
    """
    if not vals:
        return -1e30    # log(0) — returned when there are no valid paths
    a = max(vals)
    return a + np.log(sum(np.exp(v - a) for v in vals))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DEMOS
# ══════════════════════════════════════════════════════════════════════════════

def _make_codec(N, rsc_fb=0o13, rsc_ff=0o15, n_iter=8, seed=42):
    """
    Helper: build a matched (encoder, decoder) pair.

    Parameters
    ----------
    N       : block length (number of information bits)
    rsc_fb  : RSC feedback polynomial (octal)
    rsc_ff  : RSC feedforward polynomial (octal)
    n_iter  : turbo decoder iterations
    seed    : RNG seed for interleaver generation

    Returns
    -------
    rsc, encoder, decoder, interleaver
    """
    np.random.seed(seed)
    rsc         = RSCEncoder(g_feedback=rsc_fb, g_forward=rsc_ff)
    interleaver = np.random.permutation(N)
    encoder     = TurboEncoder(rsc, interleaver)
    decoder     = TurboDecoder(rsc, interleaver, n_iter=n_iter)
    return rsc, encoder, decoder, interleaver


# ──────────────────────────────────────────────────────────────────────────────

def run_demo_standard():
    """
    Demo 1 — Standard rate-1/3 turbo code.
    Sweeps Eb/N0 and prints BER.  Shows the turbo cliff.
    """
    print("=" * 60)
    print("Demo 1 — Standard Turbo Code  (rate 1/3, N=256, K=4)")
    print("=" * 60)

    N = 256
    rsc, encoder, decoder, _ = _make_codec(N, n_iter=8, seed=42)

    # FIX 2: Use rsc.g_feedback / rsc.g_forward instead of hardcoded literals,
    # so this print is correct for any polynomial configuration passed to _make_codec.
    print(f"RSC: K={rsc.m+1}, states={rsc.n_states}, "
          f"fb={oct(rsc.g_feedback)}, ff={oct(rsc.g_forward)}\n")

    np.random.seed(1)
    bits            = np.random.randint(0, 2, N)
    sys, par1, par2 = encoder.encode(bits)

    # Noiseless sanity check
    HUGE = 1e6
    s = lambda b: (1 - 2*b.astype(float)) * HUGE
    assert np.all(bits == decoder.decode(s(sys), s(par1), s(par2))), \
        "BUG: noiseless decode failed!"
    print("✓ Noiseless correctness check passed\n")

    print(f"{'Eb/N0 (dB)':<14} {'Errors':<10} {'BER':<12} {'Rate'}")
    print("-" * 48)
    rate = 1/3
    for snr in [-2, 0, 1, 2, 3, 4]:
        np.random.seed(abs(snr) * 7 + 1)
        bits_hat = decoder.decode(
            awgn_llr(sys,  snr, rate),
            awgn_llr(par1, snr, rate),
            awgn_llr(par2, snr, rate)
        )
        errs = int(np.sum(bits != bits_hat))
        print(f"{snr:<14} {errs:<10} {errs/N:<12.4f} {rate:.4f}")

    print("\n↑ Turbo cliff visible around 0–1 dB\n")


# ──────────────────────────────────────────────────────────────────────────────

def run_demo_repetition():
    """
    Demo 2 — Low-rate turbo via repetition (rate 1/(3·R)).

    Shows how increasing R shifts the turbo cliff to lower Eb/N0,
    at the cost of more bandwidth (more transmitted bits per info bit).

    Variables
    ---------
    R           : repetition factor — each of the 3 streams is sent R times
    actual_rate : 1 / (3*R) — the true transmitted-bits-per-info-bit rate
    llr_rx_*    : (N*R,) raw received LLRs for the repeated stream
    llr_*       : (N,)   combined LLRs after summing R copies
    """
    print("=" * 60)
    print("Demo 2 — Low-Rate Turbo via Repetition  (N=256, K=4)")
    print("=" * 60)
    print(f"{'R':<6} {'Rate':<10} {'Eb/N0 (dB)':<14} {'Errors':<10} {'BER'}")
    print("-" * 56)

    N = 256
    _, encoder, decoder, _ = _make_codec(N, n_iter=8, seed=42)

    np.random.seed(1)
    bits            = np.random.randint(0, 2, N)
    sys, par1, par2 = encoder.encode(bits)

    for R in [1, 2, 3, 5]:
        actual_rate = 1.0 / (3 * R)

        # Repeat each of the 3 coded streams
        sys_rep  = repeat_encode(sys,  R)   # (N*R,) transmitted bits
        par1_rep = repeat_encode(par1, R)
        par2_rep = repeat_encode(par2, R)

        # Test at Eb/N0 = 0 dB (a challenging but workable SNR for low rates)
        snr_db = 0.0
        np.random.seed(R * 13 + 1)

        # Channel: each repeated bit travels independently through AWGN
        llr_rx_sys  = awgn_llr(sys_rep,  snr_db, actual_rate)  # (N*R,)
        llr_rx_par1 = awgn_llr(par1_rep, snr_db, actual_rate)
        llr_rx_par2 = awgn_llr(par2_rep, snr_db, actual_rate)

        # Combine: sum R LLRs per original bit → (N,)
        llr_sys_c  = repeat_decode(llr_rx_sys,  R)
        llr_par1_c = repeat_decode(llr_rx_par1, R)
        llr_par2_c = repeat_decode(llr_rx_par2, R)

        bits_hat = decoder.decode(llr_sys_c, llr_par1_c, llr_par2_c)
        errs     = int(np.sum(bits != bits_hat))

        print(f"{R:<6} {actual_rate:<10.4f} {snr_db:<14} {errs:<10} {errs/N:.4f}")

    print("\n↑ Higher R → lower rate → fewer errors at same Eb/N0\n")


# ──────────────────────────────────────────────────────────────────────────────

def run_demo_puncturing():
    """
    Demo 3 — Rate adjustment via puncturing (rate 1/3 → 1/2).

    Shows how removing parity bits raises the rate at the cost of
    error correction capacity.  The decoder uses LLR=0 for erased positions.

    Variables
    ---------
    pattern     : puncturing pattern string — controls which bits are dropped
    par1_tx     : transmitted (kept) par1 bits — shorter than N
    mask1/mask2 : which positions were kept — passed to depuncture
    llr_par1_full : restored (N,) LLR array with 0s at erased positions
    """
    print("=" * 60)
    print("Demo 3 — Puncturing  (rate 1/3 up to 1/2, N=256, K=4)")
    print("=" * 60)
    print(f"{'Pattern':<14} {'Rate':<10} {'Eb/N0 (dB)':<14} {'Errors':<10} {'BER'}")
    print("-" * 60)

    N = 256
    _, encoder, decoder, _ = _make_codec(N, n_iter=8, seed=42)

    np.random.seed(1)
    bits            = np.random.randint(0, 2, N)
    sys, par1, par2 = encoder.encode(bits)
    snr_db          = 2.0   # fixed SNR for fair comparison

    patterns = {
        'none'     : 1/3,
        'rate_2_5' : 2/5,
        'rate_half': 1/2,
    }

    for pattern, rate in patterns.items():
        np.random.seed(99)

        # Encoder side: drop selected parity bits
        par1_tx, par2_tx, mask1, mask2 = puncture(par1, par2, pattern=pattern)

        # Channel: only transmitted bits go through AWGN
        llr_sys_rx  = awgn_llr(sys,     snr_db, rate)
        llr_par1_rx = awgn_llr(par1_tx, snr_db, rate)
        llr_par2_rx = awgn_llr(par2_tx, snr_db, rate)

        # Decoder side: restore full-length LLR arrays (0 at erased positions)
        llr_par1_full, llr_par2_full = depuncture(llr_par1_rx, llr_par2_rx,
                                                  mask1, mask2, N)

        bits_hat = decoder.decode(llr_sys_rx, llr_par1_full, llr_par2_full)
        errs     = int(np.sum(bits != bits_hat))

        print(f"{pattern:<14} {rate:<10.4f} {snr_db:<14} {errs:<10} {errs/N:.4f}")

    print("\n↑ Higher rate → fewer transmitted bits → more errors at same Eb/N0\n")


# ──────────────────────────────────────────────────────────────────────────────

def run_demo_strong_rsc():
    """
    Demo 4 — Compare RSC polynomial choices at fixed rate 1/3.

    Larger K → higher minimum free distance → lower error floor.
    Complexity doubles per extra K (trellis states = 2^(K-1)).
    """
    print("=" * 60)
    print("Demo 4 — RSC Polynomial Comparison  (rate 1/3, N=256)")
    print("=" * 60)
    print(f"{'K':<6} {'States':<10} {'Polys (oct)':<18} {'Errors @ 2dB':<16} {'BER'}")
    print("-" * 60)

    N      = 256
    snr_db = 2.0
    configs = [
        (3, 0o7,  0o5,  "0o7/0o5"),
        (4, 0o13, 0o15, "0o13/0o15"),   # LTE default
        (5, 0o23, 0o35, "0o23/0o35"),
    ]

    np.random.seed(1)
    bits = np.random.randint(0, 2, N)

    for K, fb, ff, label in configs:
        np.random.seed(1)
        interleaver = np.random.permutation(N)
        rsc         = RSCEncoder(g_feedback=fb, g_forward=ff)
        encoder     = TurboEncoder(rsc, interleaver)
        decoder     = TurboDecoder(rsc, interleaver, n_iter=8)

        sys, par1, par2 = encoder.encode(bits)
        np.random.seed(55)
        bits_hat = decoder.decode(
            awgn_llr(sys,  snr_db, 1/3),
            awgn_llr(par1, snr_db, 1/3),
            awgn_llr(par2, snr_db, 1/3)
        )
        errs = int(np.sum(bits != bits_hat))
        print(f"{K:<6} {rsc.n_states:<10} {label:<18} {errs:<16} {errs/N:.4f}")

    print("\n↑ Larger K generally gives fewer errors (lower error floor)\n")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demo_standard()
    run_demo_repetition()
    run_demo_puncturing()
    run_demo_strong_rsc()