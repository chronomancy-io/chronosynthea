//! E1 — pairwise comorbidity extraction for the council Wave 2 falsification.
//!
//! Generates `N` patients with chronosynthea's `generate_compact` path and
//! emits a CSV with per-condition-pair joint counts:
//!
//!   cond_a_code, cond_a_display, cond_b_code, cond_b_display,
//!   joint_count, joint_prev, marginal_a, marginal_b, expected_under_indep
//!
//! Read in tandem with the Java Synthea `conditions.csv` produced by E2 to
//! compute Pearson r over top-50 pairs (Empiricist falsification threshold
//! r ≥ 0.90).

use chronosynthea_mss::{BatchConfig, BatchGenerator, CalibratedRegistry};
use std::collections::HashMap;
use std::path::PathBuf;

fn calibrated_registry_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("data");
    p.push("prevalence");
    p.push("calibrated_registry.json");
    p
}

#[test]
#[ignore]
fn e1_emit_pairwise_csv() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found at {:?}", path);
        return;
    }

    let registry = CalibratedRegistry::load(&path).unwrap();
    let fingerprint = registry.to_fingerprint();
    let n_patients: usize = 10_000;

    // Snapshot the condition code/display table so we can map index → SNOMED
    let code_table: Vec<(String, String)> = fingerprint
        .conditions
        .iter()
        .map(|c| (c.code.clone(), c.display.clone()))
        .collect();
    let num_conditions = code_table.len();

    let config = BatchConfig {
        seed: 42,
        ..Default::default()
    };
    let generator = BatchGenerator::new(fingerprint, config);
    let patients = generator.generate_compact(n_patients);
    assert_eq!(patients.len(), n_patients);

    // Marginal and pairwise counts.
    let mut marginal = vec![0u64; num_conditions];
    let mut pairwise: HashMap<(u16, u16), u64> = HashMap::new();
    for p in &patients {
        let conds = p.conditions.as_slice();
        for &c in conds {
            marginal[c as usize] += 1;
        }
        for i in 0..conds.len() {
            for j in (i + 1)..conds.len() {
                let a = conds[i].min(conds[j]);
                let b = conds[i].max(conds[j]);
                if a == b {
                    continue;
                }
                *pairwise.entry((a, b)).or_insert(0) += 1;
            }
        }
    }

    let n_f = n_patients as f64;

    // Build a sorted list of pairs by joint count (descending) and write a CSV.
    let mut pairs: Vec<((u16, u16), u64)> = pairwise.into_iter().collect();
    pairs.sort_by(|x, y| y.1.cmp(&x.1));

    println!(
        "chronosynthea_pairwise_csv_path=/tmp/e1-chronosynthea-pairwise.csv"
    );
    println!("chronosynthea_n_patients={n_patients}");
    println!("chronosynthea_num_conditions={num_conditions}");
    println!("chronosynthea_num_pairs_observed={}", pairs.len());

    use std::fs::File;
    use std::io::Write;
    let mut out = File::create("/tmp/e1-chronosynthea-pairwise.csv").unwrap();
    writeln!(
        out,
        "cond_a_code,cond_a_display,cond_b_code,cond_b_display,joint_count,joint_prev,marginal_a,marginal_b,expected_under_indep,lift"
    )
    .unwrap();

    for ((a, b), count) in &pairs {
        let pa = marginal[*a as usize] as f64 / n_f;
        let pb = marginal[*b as usize] as f64 / n_f;
        let joint = *count as f64 / n_f;
        let expected = pa * pb;
        let lift = if expected > 0.0 { joint / expected } else { 0.0 };
        let (code_a, disp_a) = &code_table[*a as usize];
        let (code_b, disp_b) = &code_table[*b as usize];
        writeln!(
            out,
            "{},\"{}\",{},\"{}\",{},{:.6},{:.6},{:.6},{:.6},{:.4}",
            code_a, disp_a, code_b, disp_b, count, joint, pa, pb, expected, lift
        )
        .unwrap();
    }

    // Print the top-25 by joint count to stdout for the casebook.
    println!();
    println!("=== Top 25 pairs by joint count (chronosynthea n={n_patients}, seed=42) ===");
    println!(
        "{:>8} {:>8} {:<60} {:<60} {:>10} {:>10} {:>10}",
        "joint", "lift", "cond_a", "cond_b", "P(a)", "P(b)", "P(a∩b)"
    );
    for ((a, b), count) in pairs.iter().take(25) {
        let pa = marginal[*a as usize] as f64 / n_f;
        let pb = marginal[*b as usize] as f64 / n_f;
        let joint = *count as f64 / n_f;
        let expected = pa * pb;
        let lift = if expected > 0.0 { joint / expected } else { 0.0 };
        let disp_a = &code_table[*a as usize].1;
        let disp_b = &code_table[*b as usize].1;
        println!(
            "{:>8} {:>8.3} {:<60} {:<60} {:>10.6} {:>10.6} {:>10.6}",
            count,
            lift,
            disp_a.chars().take(60).collect::<String>(),
            disp_b.chars().take(60).collect::<String>(),
            pa,
            pb,
            joint
        );
    }
}
