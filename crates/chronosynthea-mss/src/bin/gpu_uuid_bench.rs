//! Microbenchmark: GPU UUID emission vs CPU `stable_uuid`.
//!
//! Build/run:
//!     cargo run --release --features gpu --bin gpu_uuid_bench -- 100000
//!
//! Reports CPU baseline (single-thread, then rayon-parallel) and GPU
//! throughput. Use the result to decide whether scaling the GPU path
//! up to full per-patient CSV emission is worth the engineering cost.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gpu_uuid_bench requires --features gpu (wgpu not compiled in)"
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() {
    use chronosynthea_mss::gpu::GpuUuidEmitter;
    use rayon::prelude::*;
    use std::time::Instant;

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    eprintln!("[bench] emitting {n} UUIDs");

    // Deterministic patient ids.
    let ids: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();

    // CPU baseline (single-threaded) — match the per-patient byte work.
    let t0 = Instant::now();
    let mut cpu_out_serial = vec![0u8; n * 36];
    for (i, &id) in ids.iter().enumerate() {
        let s = chronosynthea_mss::patient_uuid(id);
        cpu_out_serial[i * 36..i * 36 + 36]
            .copy_from_slice(s.as_bytes());
    }
    let cpu_serial_dt = t0.elapsed();
    eprintln!(
        "[cpu serial]   {} UUIDs in {:.3}ms = {:.0}M UUIDs/s",
        n,
        cpu_serial_dt.as_secs_f64() * 1000.0,
        n as f64 / cpu_serial_dt.as_secs_f64() / 1e6
    );

    // CPU baseline (rayon parallel).
    let t1 = Instant::now();
    let cpu_out_parallel: Vec<u8> = ids
        .par_iter()
        .flat_map_iter(|&id| {
            chronosynthea_mss::patient_uuid(id).into_bytes()
        })
        .collect();
    let cpu_par_dt = t1.elapsed();
    eprintln!(
        "[cpu parallel] {} UUIDs in {:.3}ms = {:.0}M UUIDs/s ({:.2}× vs serial)",
        n,
        cpu_par_dt.as_secs_f64() * 1000.0,
        n as f64 / cpu_par_dt.as_secs_f64() / 1e6,
        cpu_serial_dt.as_secs_f64() / cpu_par_dt.as_secs_f64()
    );

    // GPU path. Includes adapter init time (one-time).
    let t_init = Instant::now();
    let emitter = match GpuUuidEmitter::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU init failed: {err}");
            std::process::exit(3);
        }
    };
    let init_dt = t_init.elapsed();
    eprintln!("[gpu] init: {:.1}ms", init_dt.as_secs_f64() * 1000.0);

    // Warm-up dispatch (driver/kernel-cache cost shouldn't pollute the measurement).
    let _ = emitter.emit(&ids[..n.min(64)]);
    let (gpu_out, gpu_dt) = emitter.emit(&ids);
    eprintln!(
        "[gpu]          {} UUIDs in {:.3}ms = {:.0}M UUIDs/s ({:.2}× vs cpu serial, {:.2}× vs cpu parallel)",
        n,
        gpu_dt.as_secs_f64() * 1000.0,
        n as f64 / gpu_dt.as_secs_f64() / 1e6,
        cpu_serial_dt.as_secs_f64() / gpu_dt.as_secs_f64(),
        cpu_par_dt.as_secs_f64() / gpu_dt.as_secs_f64()
    );

    // First-UUID byte-equivalence check between CPU and GPU.
    let cpu_first = std::str::from_utf8(&cpu_out_parallel[..36]).unwrap();
    let gpu_first = std::str::from_utf8(&gpu_out[..36]).unwrap_or("<non-utf8>");
    eprintln!("  cpu[0] = {cpu_first}");
    eprintln!("  gpu[0] = {gpu_first}");
    if cpu_first != gpu_first {
        eprintln!(
            "  NOTE: GPU UUIDs do not byte-match CPU yet (kernel uses a 64-bit \
             multiply approximation; tuning needed for exact parity)."
        );
    }
}
