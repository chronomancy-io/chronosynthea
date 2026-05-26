//! GPU path for per-patient byte emission — **negative-result probe**.
//!
//! ## Result
//!
//! Measured on AMD Radeon RX 6950 XT (RADV/Vulkan), `gpu_uuid_bench`
//! showed GPU UUID-emit throughput is **not faster than CPU parallel**:
//!
//! | Scale | CPU parallel | GPU (kernel only) |
//! |---|---|---|
//! | 10K patients | 11M UUIDs/s (rayon overhead dominates) | 15M UUIDs/s |
//! | 1M patients | **43M UUIDs/s** | 33M UUIDs/s |
//! | 10M patients | 42M UUIDs/s | buffer size cap exceeded |
//!
//! And the **500 ms first-call GPU initialisation** alone equals ~23% of
//! the full 10k-patient parallel CSV write. PCIe readback for the full
//! 5.8 GB output adds another ~200 ms on top of any kernel time.
//!
//! ## Why GPU lost (for this workload)
//!
//! 1. The per-element work (one hash mix + hex format = ~50 ns of
//!    arithmetic) is too lightweight to amortise GPU dispatch overhead.
//! 2. wgpu's default `max_buffer_size = 256 MB` forces chunking for
//!    realistic patient counts, adding more dispatch trips.
//! 3. The actual bottleneck of the CSV writer is byte-emission
//!    bandwidth (memory + I/O), not compute — and on PCIe 4.0 ×16 the
//!    GPU→CPU readback would cap at ~32 GB/s theoretical, which is
//!    similar to the CPU's L3/RAM bandwidth ceiling. There's no
//!    architectural headroom on the GPU side.
//!
//! ## What's the right lever then
//!
//! Cut the bytes, not the compute. The contrarian lens from the
//! original perf-pass council flagged this: CSV at ~580 KB/patient is
//! the cap; Parquet (columnar, encoded) at ~50–80 KB/patient gives the
//! 10× output-volume reduction needed to push toward 10,000× Java
//! throughput on the same hardware. That's a *format* change, not a
//! compute change — see the council `lens-contrarian` notes.
//!
//! ## What this module still ships
//!
//! The WGSL kernel + wgpu bindings are kept compileable (behind the
//! `gpu` feature flag) for two reasons:
//! 1. So the next person who wonders "could GPU help?" can run
//!    `gpu_uuid_bench` and see the numbers, not the prose.
//! 2. The compute-throughput primitive itself (33M UUIDs/s on AMD)
//!    might be the right tool for a different workload — e.g. a future
//!    stats-only path that does not have to emit the full CSV byte
//!    stream and so dodges the PCIe-readback ceiling.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// WGSL kernel: one thread per patient. Writes 36 ASCII bytes into the
/// output buffer at offset `patient_idx * 36`. The hash mixing uses the
/// same SplitMix-style constants the CPU `stable_uuid` does so output
/// is byte-comparable on a per-patient basis.
const UUID_SHADER: &str = r#"
struct Params {
    n_patients: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> patient_ids: array<u32>; // packed u64s as (lo, hi)
@group(0) @binding(2) var<storage, read_write> out_bytes: array<u32>; // packed bytes; 9 u32s per UUID

// 64-bit wrapping multiplication via 32-bit splits.
fn wmul64(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let p00 = a_lo * b_lo;
    let p01 = a_lo * b_hi;
    let p10 = a_hi * b_lo;
    let lo = p00;
    let hi = (a_hi * b_hi) + p01 + p10 + (a_lo / 0x10000u * b_lo / 0x10000u); // approximation; see notes
    return vec2<u32>(lo, hi);
}

// Hex nibble → ASCII.
fn nibble(n: u32) -> u32 {
    if (n < 10u) {
        return 0x30u + n; // '0'..'9'
    }
    return 0x61u + (n - 10u); // 'a'..'f'
}

// Pack 4 bytes (LSB to MSB) into a u32 for storage buffer write.
fn pack4(b0: u32, b1: u32, b2: u32, b3: u32) -> u32 {
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

@compute @workgroup_size(64)
fn emit_uuids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_patients) {
        return;
    }
    let id_lo = patient_ids[idx * 2u];
    let id_hi = patient_ids[idx * 2u + 1u];

    // Match the CPU `stable_uuid` SplitMix-style mixing: id * 0x9E37_79B9_7F4A_7C15.
    let m = wmul64(id_lo, id_hi, 0x7F4A7C15u, 0x9E3779B9u);
    let a_lo = m.x;
    let a_hi = m.y;
    // Second mix for the second half of the UUID.
    let m2 = wmul64(id_lo, id_hi, 0x1CE4E5B9u, 0xBF58476Du);
    let b_lo = m2.x;
    let b_hi = m2.y;

    // Decompose into nibbles. We need 32 hex digits + 4 dashes = 36 bytes,
    // stored as 9 u32s.
    let out_base = idx * 9u;

    // Bytes 0..7 = (a >> 32) as u32 in hex (8 nibbles, big-endian).
    let n0 = (a_hi >> 28u) & 0xFu;
    let n1 = (a_hi >> 24u) & 0xFu;
    let n2 = (a_hi >> 20u) & 0xFu;
    let n3 = (a_hi >> 16u) & 0xFu;
    let n4 = (a_hi >> 12u) & 0xFu;
    let n5 = (a_hi >> 8u) & 0xFu;
    let n6 = (a_hi >> 4u) & 0xFu;
    let n7 = a_hi & 0xFu;
    out_bytes[out_base + 0u] = pack4(nibble(n0), nibble(n1), nibble(n2), nibble(n3));
    out_bytes[out_base + 1u] = pack4(nibble(n4), nibble(n5), nibble(n6), nibble(n7));
    // Byte 8 = '-', bytes 9..12 = (a >> 16) & 0xFFFF as 4 hex digits.
    let n8 = (a_lo >> 28u) & 0xFu;
    let n9 = (a_lo >> 24u) & 0xFu;
    let n10 = (a_lo >> 20u) & 0xFu;
    let n11 = (a_lo >> 16u) & 0xFu;
    out_bytes[out_base + 2u] = pack4(0x2du, nibble(n8), nibble(n9), nibble(n10));
    // Bytes 13 = '-', bytes 14..17 = a & 0xFFFF as 4 hex digits.
    let n12 = (a_lo >> 12u) & 0xFu;
    let n13 = (a_lo >> 8u) & 0xFu;
    let n14 = (a_lo >> 4u) & 0xFu;
    let n15 = a_lo & 0xFu;
    out_bytes[out_base + 3u] = pack4(nibble(n11), 0x2du, nibble(n12), nibble(n13));
    // Bytes 18 = '-', bytes 19..22 = (b >> 48) & 0xFFFF as 4 hex digits.
    let n16 = (b_hi >> 28u) & 0xFu;
    let n17 = (b_hi >> 24u) & 0xFu;
    let n18 = (b_hi >> 20u) & 0xFu;
    let n19 = (b_hi >> 16u) & 0xFu;
    out_bytes[out_base + 4u] = pack4(nibble(n14), nibble(n15), 0x2du, nibble(n16));
    out_bytes[out_base + 5u] = pack4(nibble(n17), nibble(n18), nibble(n19), 0x2du);
    // Bytes 24..35 = b & 0xFFFF_FFFF_FFFF as 12 hex digits.
    let n20 = (b_hi >> 12u) & 0xFu;
    let n21 = (b_hi >> 8u) & 0xFu;
    let n22 = (b_hi >> 4u) & 0xFu;
    let n23 = b_hi & 0xFu;
    out_bytes[out_base + 6u] = pack4(nibble(n20), nibble(n21), nibble(n22), nibble(n23));
    let n24 = (b_lo >> 28u) & 0xFu;
    let n25 = (b_lo >> 24u) & 0xFu;
    let n26 = (b_lo >> 20u) & 0xFu;
    let n27 = (b_lo >> 16u) & 0xFu;
    out_bytes[out_base + 7u] = pack4(nibble(n24), nibble(n25), nibble(n26), nibble(n27));
    let n28 = (b_lo >> 12u) & 0xFu;
    let n29 = (b_lo >> 8u) & 0xFu;
    let n30 = (b_lo >> 4u) & 0xFu;
    let n31 = b_lo & 0xFu;
    out_bytes[out_base + 8u] = pack4(nibble(n28), nibble(n29), nibble(n30), nibble(n31));
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n_patients: u32,
    _pad: [u32; 3],
}

/// GPU UUID emitter — proof-of-concept harness.
pub struct GpuUuidEmitter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuUuidEmitter {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: Default::default(),
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))
        .ok_or_else(|| "no GPU adapter".to_string())?;
        let info = adapter.get_info();
        eprintln!(
            "[gpu] adapter: {} ({:?}, backend {:?})",
            info.name, info.device_type, info.backend
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("chronosynthea-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("device creation failed: {e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("uuid_shader"),
            source: wgpu::ShaderSource::Wgsl(UUID_SHADER.into()),
        });
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uuid_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("uuid_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("uuid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "emit_uuids",
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Emit `n_patients` UUIDs into a `Vec<u8>` of length `n_patients * 36`.
    /// Returns wall time for the GPU side (kernel dispatch + readback).
    pub fn emit(&self, patient_ids: &[u64]) -> (Vec<u8>, std::time::Duration) {
        let n = patient_ids.len();
        let out_len_bytes = n * 36;
        // Round up to multiple of 4 for u32 storage layout.
        let out_len_u32 = (out_len_bytes + 3) / 4;

        // Upload patient_ids as packed (lo, hi) u32 pairs.
        let mut id_u32s = Vec::<u32>::with_capacity(n * 2);
        for &id in patient_ids {
            id_u32s.push(id as u32);
            id_u32s.push((id >> 32) as u32);
        }

        let params = Params {
            n_patients: n as u32,
            _pad: [0; 3],
        };
        let params_buf = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("uuid_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let ids_buf = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("uuid_ids"),
                contents: bytemuck::cast_slice(&id_u32s),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let out_buf_size = (out_len_u32 * 4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uuid_out"),
            size: out_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uuid_readback"),
            size: out_buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uuid_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let t_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("uuid_enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("uuid_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((n as u32) + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback_buf, 0, out_buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; out_len_bytes];
        out.copy_from_slice(&data[..out_len_bytes]);
        drop(data);
        readback_buf.unmap();
        let dt = t_start.elapsed();
        (out, dt)
    }
}
