"""tci_physics.py — TCI-Konstanten und GPU-Operatoren (Substrat für Eval O).

Was dieses Modul ist
--------------------
Die "fünf Prinzipien" der holografischen Projektnotizen, übersetzt in exakt
berechenbare Tensor-Operatoren, damit sie auf der GPU *gemessen* und nicht nur
behauptet werden können:

  P1  Irreversibler Zeitpfeil / Ouroboros-Stroboskop  -> alpha_TCI als Duty-Cycle
  P2  Holografische Informationserhaltung             -> l2-Renormierung pro Schritt
  P3  Algorithmische Subjektivität (Wide/Narrow)      -> H_norm gegen Phi^-1
  P4  Wick-Rotation in imaginäre Zeit                 -> exp((-i*alpha - dtau) H)
  P5  Ontologische Resonanz / Causal Integrity        -> S4-Block-H, dS = 0

Epistemischer Vertrag (DevMindEvidence, Evidenzgrade A-E)
---------------------------------------------------------
A  = Arithmetische Identität (nachrechenbar, wahr per Definition)
B  = Numerische Koinzidenz (trifft zu, aber ohne Mechanismus / billig zu treffen)
C  = Hypothese (prüfbar formuliert, Ausgang offen)
D  = Widerlegt (Messung spricht dagegen)
E  = Nicht messbar in dieser Umgebung

Jede Konstante trägt ihr Label. Die Labels werden durch Eval O gesetzt, nicht
vorab; was hier steht, ist die *Behauptung*, nicht das Urteil.

Herkunft der Zahlen: Nutzer-Spezifikation 2026-11-15 (Formelsatz "TCI",
Wyler 1969, Ouroboros-Stroboskop, Petrova-Korrektur). Die Formeln werden
unverändert implementiert; nur die Arithmetik wird geprüft.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Referenzwerte (CODATA 2018/2022, Planck 2018) ────────────────────────────
CODATA_ALPHA_INV = 137.035999177
CODATA_ALPHA_INV_SIGMA = 2.1e-8
C_LIGHT = 299_792_458.0
MPC_M = 3.085677581491367e22
PROTON_RADIUS_M = 0.8414e-15        # CODATA 2018 charge radius
H0_KM_S_MPC = {"Planck18": 67.4, "SH0ES": 73.0}


# ── Konstanten (P1-P4) ──────────────────────────────────────────────────────
def phi() -> float:
    return (1.0 + math.sqrt(5.0)) / 2.0


def alpha_inv_tci() -> float:
    """TCI-Feinstruktur: 4*pi^3 + pi^2 + pi."""
    return 4.0 * math.pi ** 3 + math.pi ** 2 + math.pi


def alpha_tci() -> float:
    """Ouroboros-Duty-Cycle (Behauptung: stroboskopische Refresh-Rate)."""
    return 1.0 / alpha_inv_tci()


def alpha_inv_wyler() -> float:
    """Wyler (1969): (9/(8 pi^4)) * (pi^5/(2^4 * 5!))^(1/4), invertiert.

    Achtung: Die Nutzer-Spezifikation schreibt "24 * 5!". Das ist ein
    Transkriptionsfehler — 2^4 * 5! = 1920 ergibt 137.03608245, 24 * 5! = 2880
    ergibt 151.65535479. Beide Varianten werden in `audit_constants()` geprüft.
    """
    w = (9.0 / (8.0 * math.pi ** 4)) * (math.pi ** 5 / (2 ** 4 * math.factorial(5))) ** 0.25
    return 1.0 / w


def alpha_inv_wyler_literal() -> float:
    """Die Formel *wie gedruckt* (24 * 5! = 2880) — zum Vergleich."""
    w = (9.0 / (8.0 * math.pi ** 4)) * (math.pi ** 5 / (24 * math.factorial(5))) ** 0.25
    return 1.0 / w


def phi_pow_201() -> float:
    return phi() ** 201


def ln72() -> float:
    return math.log(72.0)


def hubble_time_s(h0_km_s_mpc: float) -> float:
    return 1.0 / (h0_km_s_mpc * 1000.0 / MPC_M)


def proton_time_s() -> float:
    """Lichtlaufzeit über den Ladungsradius (die einzige natürliche
    Proton-Zeitskala; 't_proton' ist in der Vorlage nicht definiert)."""
    return PROTON_RADIUS_M / C_LIGHT


# ── O1: Konstanten-Audit ────────────────────────────────────────────────────
def _rel_dev(x: float, ref: float) -> float:
    return (x - ref) / ref


def look_elsewhere_alpha(eps_rel: float = 2.3e-6) -> Dict[str, object]:
    """Basisraten-Test: wie billig ist ein Treffer auf 1/alpha?

    Durchsucht zwei *vorab* fixierte Familien einfacher pi-Polynome und zählt,
    wie viele 1/alpha auf |rel. Abweichung| <= eps_rel treffen:

      F3: a*pi^3 + b*pi^2 + c*pi,  a in +-1..8, b,c in -8..8
      F4: a*pi^4 + b*pi^3 + c*pi^2 + d*pi, a in +-1..4, b,c,d in -4..4

    Zusätzlich die erwartete Trefferzahl unter einer Gleichverteilungs-Null:
    N * 2*eps*137 / mittlerer Abstand benachbarter Ausdrücke.
    """
    target = CODATA_ALPHA_INV
    hits: List[Dict[str, object]] = []
    n_total = 0
    spacings = []

    def consider(coeffs: Tuple[int, ...], powers: Tuple[int, ...]):
        nonlocal n_total
        n_total += 1
        v = sum(c * math.pi ** p for c, p in zip(coeffs, powers))
        if v <= 0:
            return
        rel = (v - target) / target
        if abs(rel) <= eps_rel:
            hits.append({"expr": " + ".join(f"{c}*pi^{p}" for c, p in zip(coeffs, powers)),
                         "value": v, "rel_dev": rel})

    for a in list(range(-8, 0)) + list(range(1, 9)):
        for b in range(-8, 9):
            for c in range(-8, 9):
                consider((a, b, c), (3, 2, 1))
                spacings.append(abs(4 * math.pi ** 3 + b * math.pi ** 2 + c * math.pi - target))
    for a in list(range(-4, 0)) + list(range(1, 5)):
        for b in range(-4, 5):
            for c in range(-4, 5):
                for d in range(-4, 5):
                    consider((a, b, c, d), (4, 3, 2, 1))
                    spacings.append(abs(a * math.pi ** 4 - target))

    mean_spacing = float(np.mean([s for s in spacings if s > 1e-9])) if spacings else 0.0
    expected = n_total * (2 * eps_rel * target) / mean_spacing if mean_spacing else 0.0
    hits_sorted = sorted(hits, key=lambda h: abs(float(h["rel_dev"])))
    return {
        "family": "pi-polynomials F3 (3 terms, |coef|<=8) + F4 (4 terms, |coef|<=4)",
        "n_expressions": n_total,
        "eps_rel": eps_rel,
        "n_hits": len(hits),
        "mean_neighbour_spacing": round(mean_spacing, 4),
        "expected_hits_uniform_null": round(expected, 2),
        "hits_best": hits_sorted[:8],
        "tci_expression_rank": next((i + 1 for i, h in enumerate(hits_sorted)
                                     if h["expr"] == "4*pi^3 + 1*pi^2 + 1*pi"), None),
    }


def audit_constants() -> Dict[str, object]:
    """O1: prüft die behaupteten Zahlen gegen ihre eigene Arithmetik und CODATA."""
    rows = []

    def row(name, claimed, computed, ref, note):
        rel = _rel_dev(float(computed), float(ref)) if ref else None
        rows.append({
            "parameter": name,
            "claimed": claimed,
            "computed": computed,
            "reference": ref,
            "rel_dev": None if rel is None else float(f"{rel:.3e}"),
            "abs_dev": None if ref is None else abs(float(computed) - float(ref)),
            "note": note,
        })

    a_inv = alpha_inv_tci()
    a = alpha_tci()
    w_inv = alpha_inv_wyler()
    w_lit = alpha_inv_wyler_literal()
    p201 = phi_pow_201()
    th_planck = hubble_time_s(H0_KM_S_MPC["Planck18"])
    th_shoes = hubble_time_s(H0_KM_S_MPC["SH0ES"])
    tp = proton_time_s()
    r_planck = th_planck / tp
    r_shoes = th_shoes / tp
    needed_exp_planck = math.log(r_planck) / math.log(phi())
    needed_exp_shoes = math.log(r_shoes) / math.log(phi())

    row("alpha_TCI^-1", "137.03630378", a_inv, CODATA_ALPHA_INV, "4*pi^3+pi^2+pi")
    row("alpha_TCI", "0.0072973363", a, 1.0 / CODATA_ALPHA_INV, "Duty-Cycle")
    row("alpha_Wyler^-1 (2^4*5!)", "137.03608245", w_inv, CODATA_ALPHA_INV,
        "Wyler 1969, korrekt gelesen")
    row("alpha_Wyler^-1 (24*5!, wie gedruckt)", "137.03608245", w_lit, CODATA_ALPHA_INV,
        "Transkription 24 statt 2^4 -> 151.6554, nicht 137.036")
    row("Phi^201", "1.015116e42", p201, None, "reine Arithmetik")
    row("ln(72)", "4.27666612 nats", ln72(), None, "Definition der Maximalentropie")
    row("Phi^-1", "0.61803399", 1.0 / phi(), None, "Wide/Narrow-Schwelle")
    row("T_Hubble/t_proton (Planck18)", "Phi^201", r_planck, p201,
        "H0=67.4; t_proton := r_p/c")
    row("T_Hubble/t_proton (SH0ES)", "Phi^201", r_shoes, p201, "H0=73.0")

    return {
        "rows": rows,
        "phi^201": p201,
        "T_Hubble_over_t_proton": {"Planck18": r_planck, "SH0ES": r_shoes},
        "exponent_needed_for_exact_match": {"Planck18": needed_exp_planck,
                                            "SH0ES": needed_exp_shoes},
        "claimed_exponent_decomposition": {"90": "Gravitations-Hierarchie",
                                           "111": "Dunkle Energie", "sum": 201},
        "look_elsewhere": look_elsewhere_alpha(),
    }


# ── O2: S4-Block-Hamiltonian und Wick-Propagator ────────────────────────────
def _s4_permutation_matrices(dtype=torch.float64) -> List[torch.Tensor]:
    perms = list(itertools.permutations(range(4)))
    idx = {p: i for i, p in enumerate(perms)}
    mats = []
    for g in perms:
        M = torch.zeros(24, 24, dtype=dtype)
        for h in perms:
            gh = tuple(g[h[i]] for i in range(4))     # (g o h)(i) = g[h[i]]
            M[idx[gh], idx[h]] = 1.0
        mats.append(M)
    return mats


def _cycle_type(p: Tuple[int, ...]) -> Tuple[int, ...]:
    seen = [False] * len(p)
    lens = []
    for i in range(len(p)):
        if seen[i]:
            continue
        j, n = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            n += 1
        lens.append(n)
    return tuple(sorted(lens, reverse=True))


# Vorab fixierte Klassen-Gewichte (S4-Klassen -> "Balancing" der Block-Symmetrie)
S4_CLASS_WEIGHTS = {
    (1, 1, 1, 1): +3.0,   # Identität
    (2, 1, 1): -1.0,      # Transpositionen (6)
    (2, 2): +0.5,         # Doppeltranspositionen (3)
    (3, 1): -0.5,         # 3-Zyklen (8)
    (4,): +0.25,          # 4-Zyklen (6)
}


def s4_hamiltonian_24(dtype=torch.float64) -> torch.Tensor:
    """Klassen-Summen-Hamiltonian der regulären S4-Darstellung (24x24).

    Kommutiert per Konstruktion mit der Linkswirkung von S4 — genau das, was
    "S4-Block-Symmetrie" bedeuten soll.
    """
    perms = list(itertools.permutations(range(4)))
    mats = _s4_permutation_matrices(dtype=dtype)
    H = torch.zeros(24, 24, dtype=dtype)
    classes: Dict[Tuple[int, ...], List[int]] = {}
    for i, p in enumerate(perms):
        classes.setdefault(_cycle_type(p), []).append(i)
    for ct, members in classes.items():
        w = S4_CLASS_WEIGHTS.get(ct, 0.0)
        if w == 0.0:
            continue
        proj = sum(mats[i] for i in members) / len(members)
        H = H + w * proj
    return H


def build_hamiltonian(kind: str = "s4_funnel", funnel: float = 3.0,
                      seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """72x72 Hamiltonian (3 x 24 = SH-Block) in drei Varianten.

    kind = "s4_funnel"  : S4-Klassen-Summe (24) x 3 Kanäle + semantischer
                          Massetrichter (diagonal)  -> die TCI-Behauptung
    kind = "funnel"     : nur der Trichter (Kontrolle: braucht es S4?)
    kind = "gue"        : Gaussisches unitäres Ensemble, Spektralradius
                          passend zu "s4_funnel" (Kontrolle: generisch?)
    """
    if kind in ("s4_funnel", "funnel", "s4_nofunnel", "s4_pure"):
        H24 = s4_hamiltonian_24(dtype=dtype)
        I3 = torch.eye(3, dtype=dtype)
        H = torch.kron(H24, I3) + 0.3 * torch.kron(torch.eye(24, dtype=dtype),
                                                   torch.diag(torch.tensor([0.0, 0.5, 1.0], dtype=dtype)))
        if kind == "s4_pure":
            H = torch.kron(H24, I3)
        if kind == "funnel":
            H = H * 0.0
        if kind in ("s4_funnel", "funnel"):
            x = torch.arange(72, dtype=dtype) / 71.0
            H = H + torch.diag(funnel * x ** 2)
    elif kind == "gue":
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(72, 72, generator=g, dtype=dtype)
        H = (A + A.T) / math.sqrt(2.0 * 72.0)
        lam = torch.linalg.eigvalsh(H)
        H = H / (lam.max() - lam.min())
    else:
        raise ValueError(kind)

    lam = torch.linalg.eigvalsh(H)
    return (H - lam.min() * torch.eye(72, dtype=dtype)) / (lam.max() - lam.min())


def _shannon_norm(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-30)
    return float(-(p * p.log()).sum() / math.log(72.0))


def run_trajectory(H: torch.Tensor, alpha: float, dtau: float, steps: int,
                   psi0: torch.Tensor, renormalize: bool = True,
                   strobe: Optional[torch.Tensor] = None,
                   eig: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                   ) -> Dict[str, object]:
    """Wick-Propagator exp((-i*alpha - dtau) H) auf einem 72-dim Zustand.

    Misst pro Schritt die normierte Shannon-Entropie von |psi|^2 (H_norm in
    [0,1], Maximum ln 72) und die Norm. Vektorisiert über n Startzustände:
    psi0 darf [72] oder [n,72] sein.
    """
    single = psi0.ndim == 1
    P0 = psi0[None, :] if single else psi0
    lam, V = eig if eig is not None else torch.linalg.eigh(H)
    Vc = V.to(torch.complex128)
    c = Vc.T @ P0.T.to(torch.complex128)                  # [72, n]
    decay = torch.exp(-dtau * lam.to(torch.complex128)
                      - 1j * alpha * lam.to(torch.complex128))   # [72]

    ent: List[torch.Tensor] = []
    nrm: List[torch.Tensor] = []
    for _ in range(steps + 1):
        psi = Vc @ c
        if renormalize:
            psi = psi / psi.norm(dim=0, keepdim=True)
        pr = psi.abs() ** 2
        p = pr / pr.sum(dim=0, keepdim=True)
        ent.append(-(p * p.clamp_min(1e-30).log()).sum(dim=0) / math.log(72.0))
        nrm.append(psi.norm(dim=0))
        c = c * decay[:, None]
        if strobe is not None:
            psi = Vc @ c
            if renormalize:
                psi = psi / psi.norm(dim=0, keepdim=True)
            psi = strobe.to(torch.complex128) @ psi
            c = Vc.T @ psi

    E = torch.stack(ent, dim=0)                            # [steps+1, n]
    N = torch.stack(nrm, dim=0)
    return {"H_norm": E[:, 0] if single else E,
            "norm": N[:, 0] if single else N}


def crossing_time(ent, theta: float):
    """Erster Schritt mit H_norm < theta, linear interpoliert (stetig).

    ent: Liste/Tensor [steps+1] oder [steps+1, n] -> float oder np.ndarray.
    """
    E = ent if isinstance(ent, torch.Tensor) else torch.as_tensor(ent, dtype=torch.float64)
    out = []
    for j in range(E.shape[1] if E.ndim == 2 else 1):
        e = E[:, j] if E.ndim == 2 else E
        below = (e < theta).nonzero()
        if len(below) == 0:
            out.append(float("nan"))
            continue
        t = int(below[0])
        if t == 0:
            out.append(0.0)
            continue
        denom = float(e[t - 1] - e[t])
        frac = float((e[t - 1] - theta) / denom) if denom > 0 else 1.0
        out.append(float(t - 1 + max(0.0, min(1.0, frac))))
    if E.ndim == 1:
        return None if math.isnan(out[0]) else out[0]
    return np.array(out)


def matched_spectrum(H: torch.Tensor, seed: int = 11) -> torch.Tensor:
    """H' = Q H Q^T mit zufälligem orthogonalem Q — identisches Spektrum,
    andere Eigenbasis. Prüft, ob die *Struktur* (S4) oder nur das Spektrum zählt."""
    g = torch.Generator(device=H.device).manual_seed(seed)
    A = torch.randn(72, 72, generator=g, dtype=H.dtype, device=H.device)
    Q, _ = torch.linalg.qr(A)
    return Q @ H @ Q.T


def operator_suite(n_seeds: int = 200, steps: int = 40, dtau: float = 1.0,
                   device: str = "cuda", dtype=torch.float64) -> Dict[str, object]:
    """O2: die fünf Operator-Behauptungen auf der GPU, mit Kontrollen.

    Vorab fixiert (PREREG_Eval_O):
      dtau = 1.0, steps = 40  -> die behauptete Wide->Narrow-Transition liegt
                                 im Fenster (sonst wäre "kein Crossing" ein
                                 reines Parameterartefakt).
      theta = Phi^-1 = 0.618...
      psi0  = Haar-zufällig auf der komplexen Sphäre (unitär invariant)
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    theta = 1.0 / phi()

    H_s4 = build_hamiltonian("s4_funnel", dtype=dtype).to(device)
    H_nf = build_hamiltonian("s4_nofunnel", dtype=dtype).to(device)
    H_pure = build_hamiltonian("s4_pure", dtype=dtype).to(device)
    H_fn = build_hamiltonian("funnel", dtype=dtype).to(device)
    H_gue = build_hamiltonian("gue", seed=7, dtype=dtype).to(device)
    H_match = matched_spectrum(H_s4, seed=11).to(device)

    perms = list(itertools.permutations(range(4)))
    P = _s4_permutation_matrices(dtype=dtype)[perms.index((1, 0, 2, 3))]
    strobe72 = torch.kron(P, torch.eye(3, dtype=dtype)).to(device)

    g = torch.Generator(device=device).manual_seed(20261115)
    re = torch.randn(n_seeds, 72, generator=g, dtype=dtype, device=device)
    im = torch.randn(n_seeds, 72, generator=g, dtype=dtype, device=device)
    z = torch.complex(re, im)
    psi_haar = z / z.norm(dim=1, keepdim=True)
    psi_uniform = (torch.ones(72, dtype=torch.complex128, device=device) / math.sqrt(72))

    variants = [
        ("s4_alphaTCI", H_s4, alpha_tci(), True, None),
        ("s4_alphaWyler", H_s4, 1.0 / alpha_inv_wyler(), True, None),
        ("s4_alpha0", H_s4, 0.0, True, None),
        ("s4_alpha1", H_s4, 1.0, True, None),
        ("s4_no_renorm", H_s4, alpha_tci(), False, None),
        ("s4_nofunnel", H_nf, alpha_tci(), True, None),
        ("s4_pure", H_pure, alpha_tci(), True, None),
        ("s4_strobe_alphaTCI", H_s4, alpha_tci(), True, strobe72),
        ("funnel_only", H_fn, alpha_tci(), True, None),
        ("gue_generic", H_gue, alpha_tci(), True, None),
        ("s4_matched_spectrum", H_match, alpha_tci(), True, None),
    ]

    out: Dict[str, object] = {"config": {"n_seeds": n_seeds, "steps": steps,
                                         "dtau": dtau, "theta_phi_inv": theta,
                                         "device": device}}
    traces: Dict[str, List[float]] = {}
    tcross: Dict[str, np.ndarray] = {}
    for name, H, alpha, renorm, strobe in variants:
        eig = torch.linalg.eigh(H)
        r = run_trajectory(H, alpha, dtau, steps, psi_haar, renorm, strobe, eig=eig)
        E = r["H_norm"].cpu().numpy()                      # [steps+1, n]
        Nn = r["norm"].cpu().numpy()
        tc = crossing_time(r["H_norm"], theta)
        ok = tc[~np.isnan(tc)]
        out[name] = {
            "n": n_seeds,
            "t_cross_mean": round(float(np.mean(ok)), 3) if len(ok) else None,
            "t_cross_std": round(float(np.std(ok)), 3) if len(ok) else None,
            "crossed_fraction": round(float(len(ok) / n_seeds), 3),
            "H_norm_final_mean": round(float(E[-1].mean()), 6),
            "H_norm_min_mean": round(float(E.min(axis=0).mean()), 6),
            "norm_loss_final_mean": round(float((1.0 - Nn[-1]).mean()), 6),
        }
        traces[name] = [round(float(x), 6) for x in E.mean(axis=1)]
        tcross[name] = tc

    # Headline: ein fester Gleichverteilungs-Zustand (die "Wide"-Initialisierung)
    headline = {}
    for name, H, alpha, renorm, strobe in variants:
        eig = torch.linalg.eigh(H)
        r = run_trajectory(H, alpha, dtau, steps, psi_uniform, renorm, strobe, eig=eig)
        headline[name] = {
            "H_norm_t": [round(float(x), 6) for x in r["H_norm"]],
            "t_cross": crossing_time(r["H_norm"], theta),
            "norm_start": float(r["norm"][0]), "norm_end": float(r["norm"][-1]),
        }

    # ── Kontrollen ──────────────────────────────────────────────────────
    from scipy import stats as _st

    def _ks(a, b):
        a = np.asarray([x for x in a if not (isinstance(x, float) and math.isnan(x))])
        b = np.asarray([x for x in b if not (isinstance(x, float) and math.isnan(x))])
        if len(a) < 3 or len(b) < 3:
            return None
        return float(_st.ks_2samp(a, b).pvalue)

    # Spektrum-gematchte Zufalls-Hamiltonians: hat die S4-Struktur eine
    # Signatur, die über "irgendeine Eigenbasis" hinausgeht?
    g2 = torch.Generator(device=device).manual_seed(23)
    ens_tc, ens_final = [], []
    for _ in range(200):
        A = torch.randn(72, 72, generator=g2, dtype=dtype, device=device)
        Q, _ = torch.linalg.qr(A)
        Hx = Q @ H_s4 @ Q.T
        rx = run_trajectory(Hx, alpha_tci(), dtau, steps, psi_uniform)
        ens_tc.append(crossing_time(rx["H_norm"], theta))
        ens_final.append(float(rx["H_norm"][-1]))
    ens_tc = np.array([x if x is not None else steps + 1.0 for x in ens_tc])
    ens_final = np.array(ens_final)
    tc_s4_uniform = headline["s4_alphaTCI"]["t_cross"]
    pct_tc = float((ens_tc < (tc_s4_uniform if tc_s4_uniform is not None else steps + 1)).mean())
    pct_final = float((ens_final < headline["s4_alphaTCI"]["H_norm_t"][-1]).mean())

    spread = float(np.std(tcross["s4_alphaTCI"][~np.isnan(tcross["s4_alphaTCI"])]))
    aT = np.array(headline["s4_alphaTCI"]["H_norm_t"])
    a0 = np.array(headline["s4_alpha0"]["H_norm_t"])
    a1 = np.array(headline["s4_alpha1"]["H_norm_t"])
    alpha_effect = float(np.max(np.abs(aT - a0)))
    alpha_effect_1 = float(np.max(np.abs(a1 - a0)))

    # Schwellen-Scan: ist Phi^-1 eine ausgezeichnete Schwelle?
    eig_s4 = torch.linalg.eigh(H_s4)
    scan = []
    for th in np.arange(0.30, 0.901, 0.01):
        tc = crossing_time(run_trajectory(H_s4, alpha_tci(), dtau, steps, psi_haar,
                                          eig=eig_s4)["H_norm"], float(th))
        censored = np.where(np.isnan(tc), steps + 1.0, tc)   # Zensierung
        scan.append({"theta": round(float(th), 4),
                     "t_cross_mean": round(float(np.mean(censored)), 3),
                     "n_uncensored": int((~np.isnan(tc)).sum())})
    valid = [s["t_cross_mean"] for s in scan if s["t_cross_mean"] is not None]
    d2 = np.abs(np.diff(valid, 2)) if len(valid) > 3 else np.array([])
    thetas = [s["theta"] for s in scan if s["t_cross_mean"] is not None]
    idx_phi = int(np.argmin(np.abs(np.array(thetas) - theta)))
    out["threshold_scan"] = {
        "points": scan,
        "mean_second_difference_median": float(np.median(d2)) if len(d2) else None,
        "second_difference_at_phi_inv": float(d2[idx_phi]) if len(d2) > idx_phi else None,
        "max_second_difference": float(np.max(d2)) if len(d2) else None,
        "t_cross_monotone": bool(np.all(np.diff(valid) <= 1e-9)),
    }
    out["controls"] = {
        "s4_percentile_t_cross_in_matched_ensemble": pct_tc,
        "s4_percentile_final_entropy_in_matched_ensemble": pct_final,
        "matched_ensemble_t_cross_mean": round(float(ens_tc.mean()), 3),
        "matched_ensemble_final_entropy_mean": round(float(ens_final.mean()), 6),
        "s4_no_funnel_final_entropy": headline["s4_nofunnel"]["H_norm_t"][-1],
        "s4_pure_final_entropy": headline["s4_pure"]["H_norm_t"][-1],
        "s4_pure_min_entropy": min(headline["s4_pure"]["H_norm_t"]),
        "haar_spread_t_cross": spread,
        "alpha_TCI_effect_max_abs": alpha_effect,
        "alpha_TCI_effect_over_spread": alpha_effect / spread if spread > 0 else None,
        "alpha_1_effect_max_abs": alpha_effect_1,
        "alpha_TCI_vs_alpha_Wyler_rel_diff": abs(alpha_tci() - 1.0 / alpha_inv_wyler()) / alpha_tci(),
        "ks_p_s4_vs_matched_spectrum": _ks(tcross["s4_alphaTCI"], tcross["s4_matched_spectrum"]),
        "ks_p_s4_vs_gue": _ks(tcross["s4_alphaTCI"], tcross["gue_generic"]),
        "ks_p_s4_vs_funnel_only": _ks(tcross["s4_alphaTCI"], tcross["funnel_only"]),
    }
    out["traces_mean"] = traces
    out["headline_trajectory"] = headline
    return out


# ── Selbsttest ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    a = audit_constants()
    print(json.dumps(a, indent=1)[:4000])
    s = operator_suite(n_seeds=20, steps=20)
    print(json.dumps({k: v for k, v in s.items() if k != "traces_mean"}, indent=1)[:4000])
