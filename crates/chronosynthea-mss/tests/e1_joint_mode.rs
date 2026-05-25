//! E1 — Joint-mode regression. Documents the tradeoff between marginal
//! fidelity and pairwise correlation that the d5 'Joint Structure' axis
//! exposes.
//!
//! Set `CHRONOSYNTHEA_COOCCURRENCE_PATH` to an empirical pairwise file
//! (see `tests/scripts/extract_cooccurrence.py` for how that file is
//! produced from a Java Synthea run) and the joint sampling path activates.
//!
//! This test is `#[ignore]` because it requires an environment variable
//! and an empirical cooccurrence file; run it explicitly via
//! `CHRONOSYNTHEA_COOCCURRENCE_PATH=… cargo test … --ignored e1_joint_mode`.

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
fn e1_joint_mode_emit_pairwise() {
    let path = calibrated_registry_path();
    if !path.exists() {
        eprintln!("Skipping: calibrated_registry.json not found");
        return;
    }

    let coocc_env = std::env::var("CHRONOSYNTHEA_COOCCURRENCE_PATH").ok();
    println!(
        "CHRONOSYNTHEA_COOCCURRENCE_PATH = {}",
        coocc_env.as_deref().unwrap_or("<unset>")
    );

    let registry = CalibratedRegistry::load(&path).unwrap();
    println!(
        "loaded cooccurrence pairs: {}",
        registry.cooccurrence_pairs.len()
    );

    let fingerprint = registry.to_fingerprint();
    let mode = if fingerprint.cooccurrence.is_empty() {
        "marginal-only"
    } else {
        "pairwise-empirical"
    };
    println!("d5 Joint Structure mode: {mode}");

    let n_patients: usize = 10_000;
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
    use std::fs::File;
    use std::io::Write;
    let out_path = format!("/tmp/e1-chronosynthea-pairwise-{mode}.csv");
    let mut out = File::create(&out_path).unwrap();
    writeln!(
        out,
        "cond_a_code,cond_a_display,cond_b_code,cond_b_display,joint_count,joint_prev,marginal_a,marginal_b,expected_under_indep,lift"
    )
    .unwrap();

    let mut pairs: Vec<((u16, u16), u64)> = pairwise.into_iter().collect();
    pairs.sort_by(|x, y| y.1.cmp(&x.1));
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
    println!("Wrote pairwise CSV to {out_path}");
    println!("n_patients={n_patients} num_pairs_observed={}", pairs.len());

    // Lift summary
    let lifts: Vec<f64> = pairs
        .iter()
        .map(|((a, b), count)| {
            let pa = marginal[*a as usize] as f64 / n_f;
            let pb = marginal[*b as usize] as f64 / n_f;
            let joint = *count as f64 / n_f;
            let expected = pa * pb;
            if expected > 0.0 {
                joint / expected
            } else {
                0.0
            }
        })
        .collect();
    let mut sorted = lifts.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "lift distribution: min={:.3}  median={:.3}  p95={:.3}  max={:.3}",
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[(sorted.len() as f64 * 0.95) as usize],
        sorted.last().copied().unwrap_or(0.0)
    );
}
