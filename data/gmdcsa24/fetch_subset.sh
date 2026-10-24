#!/bin/bash
# Deterministic subset: every 3rd clip (01,04,07,...) per subject/class. GMDCSA-24 (CC BY 4.0, Zenodo 10.5281/zenodo.13354453)
R=https://raw.githubusercontent.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/master
for s in 1 2 3 4; do for c in ADL Fall; do
  n=$(($(wc -l < "Subject $s/$c.csv")))
  for i in $(seq 1 3 30); do f=$(printf "%02d.mp4" $i)
    grep -q "^$f" "Subject $s/$c.csv" || continue
    mkdir -p "Subject $s/$c"; [ -s "Subject $s/$c/$f" ] || curl -sfL "$R/Subject%20$s/$c/$f" -o "Subject $s/$c/$f"
  done; done; done
echo FETCH_DONE
