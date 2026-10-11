"""Compare equiv_dump outputs: master vs branch, and branch 1 vs 4 threads.

Usage: compare.py <out_dir>   (expects <out_dir>/{master,branch}/t{1,4}/*.bin)
Prints a Markdown report; exits 1 if leaf ids differ from master, if the
branch's 1- and 4-thread outputs are not bit-identical, or if any prediction
differs from master by more than 1e-6 relative.
"""
import os
import struct
import sys

REL_TOL = 1e-6
SHAP_ULP_FLOOR = 1e-3  # SHAP ulp ignores |v| < 1e-3 * max |SHAP| in the config


def load(path):
    data, out, i = open(path, "rb").read(), {}, 0
    while i < len(data):
        j = data.index(b"\n", i)
        tag, n = data[i:j].split()
        n, i = int(n), j + 1
        out[tag.decode()] = struct.unpack(f"<{n}I", data[i:i + 4 * n])
        i += 4 * n + 1
    return out


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ordered(bits):
    # Map float bits to integers whose difference is the ulp distance.
    return bits if bits < 0x80000000 else 0x80000000 - (bits & 0x7FFFFFFF)


def diff(a, b, ulp_floor_frac=0.0):
    """(max abs, max ulp, max rel) between two float bit arrays.

    Elements whose magnitude is below ulp_floor_frac times the array's largest
    magnitude are left out of the ulp figure (cancellation noise near zero
    gives huge ulp counts for tiny absolute differences); they still count
    toward max abs.
    """
    mabs = mulp = mrel = 0.0
    floor = 0.0
    if ulp_floor_frac:
        floor = ulp_floor_frac * max((abs(v) for v in map(f32, a) if v == v), default=0.0)
    for x, y in zip(a, b):
        if x == y:
            continue
        fx, fy = f32(x), f32(y)
        if fx != fx or fy != fy:
            return float("inf"), float("inf"), float("inf")
        d = abs(fx - fy)
        mabs = max(mabs, d)
        if max(abs(fx), abs(fy)) >= floor:
            mulp = max(mulp, abs(ordered(x) - ordered(y)))
        mrel = max(mrel, d / max(abs(fx), abs(fy), 1e-30))
    return mabs, mulp, mrel


def main():
    root = sys.argv[1]
    names = sorted(os.listdir(f"{root}/master/t1"))
    rows, failures = [], []
    worst = {"P_abs": 0.0, "P_ulp": 0, "S_abs": 0.0, "S_ulp": 0}
    for name in names:
        cfg = name[:-4]
        b1, b4 = load(f"{root}/branch/t1/{name}"), load(f"{root}/branch/t4/{name}")
        cells = []
        for t in ("1", "4"):
            m, b = load(f"{root}/master/t{t}/{name}"), (b1 if t == "1" else b4)
            leaves_eq = m["L"] == b["L"] and len(m["L"]) == len(b["L"])
            pa, pu, pr = diff(m["P"], b["P"])
            sa, su, _ = diff(m["S"], b["S"], SHAP_ULP_FLOOR)
            worst["P_abs"], worst["P_ulp"] = max(worst["P_abs"], pa), max(worst["P_ulp"], pu)
            worst["S_abs"], worst["S_ulp"] = max(worst["S_abs"], sa), max(worst["S_ulp"], su)
            if not leaves_eq:
                failures.append(f"{cfg} t{t}: leaf ids differ from master")
            if pr > REL_TOL:
                failures.append(f"{cfg} t{t}: prediction rel diff {pr:.2e} > {REL_TOL}")
            cells.append(f"{'yes' if leaves_eq else '**NO**'} | {pa:.1e} / {pu:.0f} | {sa:.1e} / {su:.0f}")
        det = all(b1[k] == b4[k] for k in "PLS")
        if not det:
            failures.append(f"{cfg}: branch 1-thread and 4-thread outputs differ")
        rows.append(f"| {cfg} | {cells[0]} | {cells[1]} | {'yes' if det else '**NO**'} |")
    print(f"### Equivalence vs master ({len(names)} configs)\n")
    print(f"Worst over all configs and thread counts: predictions max abs {worst['P_abs']:.2e}, "
          f"max {worst['P_ulp']:.0f} ulp; SHAP max abs {worst['S_abs']:.2e}, max {worst['S_ulp']:.0f} ulp.\n")
    print("<details><summary>Per-config table</summary>\n")
    print("| config | t1 leaves = | t1 pred abs / ulp | t1 SHAP abs / ulp "
          "| t4 leaves = | t4 pred abs / ulp | t4 SHAP abs / ulp | branch t1 == t4 |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n</details>\n")
    if failures:
        print(f"**FAILED ({len(failures)})**\n")
        print("\n".join(f"- {f}" for f in failures))
        sys.exit(1)
    print("**PASS**: leaf ids identical to master, branch bit-identical at 1 and 4 threads, "
          f"prediction diffs within {REL_TOL} relative.")


if __name__ == "__main__":
    main()
