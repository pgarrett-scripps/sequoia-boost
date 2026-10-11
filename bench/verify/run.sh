#!/usr/bin/env bash
# Equivalence check and rough timing of a branch checkout against a baseline
# checkout (normally master).
# Usage: bench/verify/run.sh <baseline_checkout> <branch_checkout> <out_dir>
# Writes a Markdown report to stdout and, on GitHub Actions, the job summary.
set -euo pipefail
base=$(cd "$1" && pwd); branch=$(cd "$2" && pwd); out=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
report="$out/report.md"; : > "$report"

# The harness builds against ../../crates/sequoia-boost, so copy it into the
# baseline checkout too (the baseline need not contain it).
for name in master branch; do
  src=$([ $name = master ] && echo "$base" || echo "$branch")
  if [ "$src/bench/verify" != "$here" ]; then
    mkdir -p "$src/bench/verify"
    cp -R "$here/Cargo.toml" "$here/src" "$src/bench/verify/"
  fi
  echo "::group::build $name"
  cargo build --release --manifest-path "$src/bench/verify/Cargo.toml" \
    --target-dir "$out/target-$name"
  echo "::endgroup::"
done

for name in master branch; do
  for t in 1 4; do
    RAYON_NUM_THREADS=$t "$out/target-$name/release/equiv_dump" "$out/$name/t$t" $t 2> "$out/dispatch-$name.txt"
  done
done

{
  echo "## sequoia-boost verify: $(uname -sm)"
  echo
  echo "Runtime SIMD dispatch: master \`$(cat "$out/dispatch-master.txt")\`, branch \`$(cat "$out/dispatch-branch.txt")\`"
  echo
} >> "$report"
status=0
python3 "$here/compare.py" "$out" >> "$report" || status=$?

{
  echo
  echo "### Rough timing (shared CI runner, not a benchmark)"
  echo
  echo "Hist training 50k x 20, depth 6, 50 rounds (median of 3); predict 100 trees on 50k rows (mean of 3)."
  echo
  echo "| objective | threads | master train ms | branch train ms | train speedup | master predict ms | branch predict ms | predict speedup |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|"
} >> "$report"
for t in 1 4; do
  RAYON_NUM_THREADS=$t "$out/target-master/release/timing_rough" $t > "$out/timing-master-t$t.txt"
  RAYON_NUM_THREADS=$t "$out/target-branch/release/timing_rough" $t > "$out/timing-branch-t$t.txt"
  paste -d' ' "$out/timing-master-t$t.txt" "$out/timing-branch-t$t.txt" \
    | awk '{printf "| %s | %s | %s | %s | %.2fx | %s | %s | %.2fx |\n", $2, $3, $4, $9, $4/$9, $5, $10, $5/$10}' >> "$report"
done

cat "$report"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && cat "$report" >> "$GITHUB_STEP_SUMMARY"
exit $status
