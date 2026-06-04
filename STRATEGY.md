# ChronoSynthea: Strategic Analysis & Market Positioning

> **Honesty note (MSS pass, 2026-06-04).** This is a forward-looking strategy/marketing document. Several claims in earlier drafts (a fixed "16,000x faster" multiplier, "1M patients in under 1 second," "statistically indistinguishable from Java Synthea," and the cost/P&L figures derived from them) were **not supported by anything measured in this repo** and have been corrected or flagged. What is actually measured: ChronoSynthea generates patient *statistics* at roughly **8.4–9.0 M/sec (stats-only)** and **3.7–4.0 M/sec (full-stats)** on an 8-core/16-thread Ryzen 7 5800X (see PERFORMANCE.md), producing no materialized records on those paths. There is **no Java Synthea baseline measured here**, so no speedup multiple is asserted. The generator samples conditions independently from precomputed base rates (no causal model, no co-occurrence by default), so output is **not** demonstrated to be statistically equivalent to Java Synthea. Market-size, pricing, and P&L numbers below are unverified business projections, not facts.

## Executive Summary

ChronoSynthea is a high-throughput synthetic **patient-statistics** generator. On the machine in PERFORMANCE.md it produces aggregate statistics for millions of synthetic patients per second on a single multi-core CPU, with no per-record materialization on the fast paths. Its positioning rests on:

- **Very high statistics-generation throughput** (measured millions/sec; see PERFORMANCE.md) — the speed advantage vs Java Synthea is plausible but **unquantified** here (no Java baseline measured)
- **Low marginal compute cost** per generated population (qualitative; the dollar figures below are estimates, not measured)
- **Self-consistent sampling**: generated prevalences reproduce the registry's base rates to within ~0.31% at n=100K — this is sampling noise vs its own base rates, **not** proven equivalence to Java Synthea

This document outlines the technical foundations, market implications, competitive positioning, and monetization strategy. Treat all numeric business claims as projections to be validated.

---

## Table of Contents

1. [The Breakthrough](#1-the-breakthrough)
2. [Why This Matters](#2-why-this-matters)
3. [Technical Foundation](#3-technical-foundation)
4. [Market Analysis](#4-market-analysis)
5. [Competitive Landscape](#5-competitive-landscape)
6. [Go-to-Market Strategy](#6-go-to-market-strategy)
7. [Monetization Model](#7-monetization-model)
8. [Cost Structure & Unit Economics](#8-cost-structure--unit-economics)
9. [References & Citations](#9-references--citations)

---

## 1. The Breakthrough

### Measured Throughput (ChronoSynthea, this repo)

| Path | What it produces | Throughput (measured) | 1M patients |
|------|------------------|-----------------------|-------------|
| `generate_stats_only` | atomic counts, no records | ~8.4–9.0 M pts/sec | ~0.11–0.12 s |
| `generate_full_stats_only` | counts incl. meds/obs/procs | ~3.7–4.0 M pts/sec | ~0.25–0.27 s |

Measured on AMD Ryzen 7 5800X (8 cores / 16 threads), Rust 1.94.0, `--release`, real 214-condition registry, 2026-06-04. **No Java Synthea run was performed here**, so the "vs Java" column and any speedup multiplier have been removed — that comparison is an Unknown (see PERFORMANCE.md §7). Memory and cost-per-million figures were not measured and have been removed from this table.

### Self-Consistency Check (n=100,000)

```
Statistical Comparison (n=100000)
  Max Deviation: 0.31%
  KL Divergence: -0.006132
  Chi-Squared:   181.17
  Failure Rate:  0.0% (0/214 conditions, 10% tolerance)
```

This compares generated stats against **ChronoSynthea's own base rates** (the same fingerprint used to generate them). It demonstrates low sampling noise — **not** equivalence to Java Synthea. Because conditions are sampled independently (no co-occurrence) and demographic multipliers are uniform, comorbidity structure and demographic variation are explicitly *not* reproduced. Do not claim "statistically indistinguishable from Java Synthea."

---

## 2. Why This Matters

### 2.1 The Synthetic Data Imperative

Healthcare organizations face a fundamental tension:

1. **Data is essential** for AI/ML training, software testing, analytics, and research
2. **Real patient data** is protected by HIPAA, GDPR, and other regulations
3. **De-identification is insufficient**—studies show re-identification rates of 87%+ for supposedly anonymized data [1]

Synthetic data solves this by generating clinically realistic but entirely fabricated patient records. The challenge has always been **generation speed and cost**.

### 2.2 Previous Limitations

Java Synthea, the de facto standard for synthetic patient generation, is a full causal simulation (a week-by-week state machine over each patient's lifespan) and is correspondingly compute-intensive.

> **Unverified.** Earlier drafts quoted a specific "~7 hours per 1M patients" / "~75 patients/sec" Java Synthea figure attributed to a "Synthea Wiki." No such measurement, source, or version is recorded in this repo, and reference [2] is the JAMIA paper, not a wiki page with that number. The figure has been removed. The qualitative point stands — a per-patient lifespan simulation is far slower than direct statistical sampling — but the exact ratio is **unmeasured** (Unknown).

The practical implications of a slow generator (impractical for CI/CD test fixtures, costly at large scale, unsuitable for real-time endpoints) remain the motivation; the specific cost figures below are estimates, not measurements.

### 2.3 What Sub-Second Changes

| Use Case | Before ChronoSynthea | After ChronoSynthea |
|----------|---------------------|---------------------|
| **Sales demos** | Pre-generate data, hope it matches customer needs | Generate custom populations in real-time |
| **CI/CD testing** | Maintain static test fixtures | Fresh, varied data for every test run |
| **ML training** | Fixed dataset, risk of overfitting | Unlimited varied training data |
| **Research** | Budget limits population size | Large synthetic-statistics runs become cheap (note: these are marginal-rate samples, not full causal simulations) |
| **API endpoints** | Return pre-cached data | True on-demand generation |

---

## 3. Technical Foundation

### 3.1 Minimally Sufficient Statistic (MSS) Approach

Traditional simulation (Java Synthea) works by:
1. Initializing a patient with demographics
2. Running a week-by-week state machine for their entire life
3. Recording every transition and event
4. Outputting to FHIR/CSV

This is computationally expensive because it simulates **causation**.

Our approach captures **marginal** statistics (not the full causal/joint structure):
1. Pre-compute a base-rate fingerprint (the bundled `calibrated_registry.json`)
2. Build per-demographic marginals; the joint demographic distribution is an *independence approximation* (product of marginals), and the condition co-occurrence map is left empty by default
3. Sample each condition independently against its base rate; one O(1) Vose-alias draw selects the demographic archetype per patient
4. Output records (or, on the fast paths, only aggregate counts)

This is a deliberate fidelity trade-off: it makes generation fast and the marginals exact, but it does **not** reproduce comorbidity clustering, demographic-conditioned prevalence, or causal timing.

### 3.2 Key Optimizations

"Impact" describes the mechanism; per-technique speedups were not isolated in this repo.

| Technique | Where it actually applies | Reference |
|-----------|---------------------------|-----------|
| **SIMD threshold draws** | meds/obs/procs in the full-stats path (the default condition path is scalar) | [3] |
| **Vose alias method** | O(1) archetype selection vs O(n) linear CDF search | [5] |
| **Lock-free atomic statistics** | parallel count aggregation without a mutex | [6] |
| **u16 code indices** | hot-path references avoid `Arc<str>` refcounting (code *strings* are still owned `String`s, not compile-time-interned) | [7] |
| **Arena allocation (`bumpalo`)** | defined but **not** on the measured throughput paths — those materialize no records | [4] |

### 3.3 Self-Consistency Validation (not Java equivalence)

We compare generated stats against the fingerprint's own base rates using:

1. **Kullback-Leibler Divergence** [8]
2. **Chi-Squared** [9] (against an ad-hoc in-code threshold `sqrt(n)·c/10 ≈ 6767` at n=100K, c=214 — a project heuristic, not a standard critical value)
3. **Per-Condition Prevalence** vs base rate

Measured (n=100,000, 2026-06-04):
- **KL Divergence: -0.006132** (≈ 0 vs base rates)
- **Chi-Squared: 181.17**
- **Max Deviation: 0.31%** (vs the registry's base rates)

These show the sampler reproduces its own base rates with low noise. They are **not** a comparison to Java Synthea output and do not establish clinical or distributional equivalence to it.

---

## 4. Market Analysis

### 4.1 Total Addressable Market

The synthetic data market is projected to reach **$3.1 billion by 2030**, with healthcare being the largest vertical [10].

| Segment | Market Size (2024) | Growth Rate | Key Drivers |
|---------|-------------------|-------------|-------------|
| Healthcare AI/ML | $1.2B | 35% CAGR | FDA guidance on AI, clinical trial simulation |
| Health IT Testing | $800M | 18% CAGR | Interoperability mandates, EHR adoption |
| Pharma R&D | $600M | 22% CAGR | Reduced trial costs, rare disease simulation |
| Academic Research | $300M | 15% CAGR | Open science, reproducibility requirements |

### 4.2 Target Customer Segments

**Tier 1: Enterprise Health IT (Epic, Cerner, Meditech)**
- Pain: Need massive test datasets for EHR development
- Budget: $500K-5M/year for testing infrastructure
- Value prop: Eliminate batch processing, enable real-time test data

**Tier 2: Pharma & Clinical Research (Pfizer, Roche, Novartis)**
- Pain: Insufficient real-world data for rare disease research
- Budget: $1-10M/year for synthetic data initiatives
- Value prop: Generate 100M+ patient populations for Monte Carlo simulations

**Tier 3: Healthcare AI Startups**
- Pain: Need large training datasets without HIPAA exposure
- Budget: $10K-100K/year
- Value prop: Unlimited training data at negligible cost

**Tier 4: Academic Institutions**
- Pain: Limited compute budgets, long IRB processes for real data
- Budget: $5K-50K/year
- Value prop: Democratized access to population-scale synthetic data

### 4.3 Regulatory Tailwinds

1. **FDA Guidance on AI/ML-Based SaMD (2021)** [11]: Encourages use of synthetic data for algorithm validation
2. **21st Century Cures Act**: Mandates interoperability, driving need for test data
3. **HIPAA Safe Harbor**: Synthetic data is not PHI, no compliance burden
4. **EU AI Act (2024)**: Requires diverse training data, synthetic data qualifies

---

## 5. Competitive Landscape

### 5.1 Direct Competitors

Competitor speeds and pricing below are **rough public estimates, not measured by us**; the Synthea speed cell is intentionally left unquantified because no Java baseline was measured here.

| Competitor | Approach | Speed | Pricing (est.) | Weakness |
|------------|----------|-------|----------------|----------|
| **Synthea (Open Source)** | Causal state-machine simulation | not measured here (lifespan sim → much slower than direct sampling) | Free (compute costs) | Slow, resource-intensive |
| **MDClone** | Real data transformation | N/A (batch) | $500K-2M/year | Requires source data, privacy concerns |
| **Syntegra** | GAN-based synthesis | Minutes per cohort | $100K-500K/year | Slow, no FHIR native |
| **Gretel.ai** | ML-based synthesis | Seconds-minutes | $50K-200K/year | Not healthcare-specific |
| **Hazy** | Differential privacy | Batch processing | $100K+/year | Requires source data |

### 5.2 Competitive Advantages

**1. Speed**

ChronoSynthea generates patient *statistics* at measured millions/sec (PERFORMANCE.md). It is very likely orders of magnitude faster than a lifespan simulator, but the exact ratio is **unmeasured** — no head-to-head Java Synthea run exists in this repo, so the previous "1,600,000 pts/sec vs 75 pts/sec, 3-orders-of-magnitude" bar chart has been removed as unsubstantiated.

**2. Statistical Fidelity — *marginals only*, not proven Java equivalence**

ChronoSynthea reproduces its registry's marginal base rates with low sampling noise (0.31% max deviation, chi-squared 181.17 at n=100K). This is **not** mathematical equivalence to Java Synthea, and it explicitly does **not** model comorbidity or demographic-conditioned prevalence. Earlier "mathematically equivalent to the gold standard / confirming distributional equivalence" language overstated this and has been removed.

**3. Zero Privacy Risk (No Source Data)**

We generate from a pre-computed base-rate fingerprint rather than transforming real records, so there is no source PHI to re-identify. (This is a property of the *approach*; it is not a formal privacy guarantee and no de-identification proof is provided in this repo.)

**4. Cost Structure**

The earlier per-million dollar figures ($0.00001/1M, "99.99% lower", "1,500,000x") were derived from the now-removed ~600ms/1.6M-per-second numbers and were not measured. Qualitatively, marginal compute per generated population is low because the fast paths allocate no per-patient records; concrete cloud costs should be benchmarked before being quoted to customers.

### 5.3 Barriers to Entry

These are arguable strategic claims, not measured facts:

1. **Domain expertise** to choose "sufficient" statistics for a given use case
2. **Rust systems programming** for the SIMD / lock-free / cache-friendly implementation
3. **Validation methodology** (note: the current in-repo validation is self-consistency, not equivalence to an external gold standard — a true equivalence study would be additional work)
4. **Calibration data**: the bundled `calibrated_registry.json` is a precomputed base-rate file. Its provenance as the output of "extensive Java Synthea runs" is asserted by its `description` field but is **not** demonstrated in this repo.

---

## 6. Go-to-Market Strategy

### 6.1 Positioning Statement

> **For healthcare organizations that need fast synthetic patient data, ChronoSynthea generates patient statistics at millions of patients per second on a single CPU—enabling real-time, on-demand generation for testing, AI training, and research.**

> Avoid the claims "clinically accurate," "1M records in under 1 second," and "16,000x faster" in customer-facing copy until they are substantiated: the fast paths emit aggregate statistics (not per-patient records), output reflects marginal base rates only (no comorbidity/causal modeling), and no Java baseline has been measured.

### 6.2 Launch Strategy

**Phase 1: Developer Adoption (Months 1-6)**
- Open-source the core MSS library
- Publish benchmarks and validation methodology
- Offer generous free tier (10M patients/month)
- Target: 1,000 developers, 100 organizations

**Phase 2: Enterprise Pilot (Months 6-12)**
- Partner with 5-10 enterprise customers
- Develop FHIR R4 export, custom demographics
- Obtain HITRUST/SOC 2 certification
- Target: $500K ARR

**Phase 3: Market Expansion (Months 12-24)**
- Launch self-serve SaaS platform
- Add custom module support
- Expand to international markets (GDPR compliance)
- Target: $5M ARR

### 6.3 Marketing Channels

| Channel | Tactics | Expected CAC |
|---------|---------|--------------|
| **Content Marketing** | Technical blogs, benchmark comparisons, whitepapers | $50-100 |
| **Developer Relations** | Conference talks, open-source contributions, tutorials | $100-200 |
| **Direct Sales** | Enterprise outreach, POC programs, RFP responses | $5,000-20,000 |
| **Partnerships** | EHR vendor integrations, cloud marketplace listings | Variable |

### 6.4 Key Messages by Audience

**For Developers:**
> "Generate a million test patients faster than a database query. Zero infrastructure, zero waiting."

**For IT Leadership:**
> "Eliminate synthetic data bottlenecks. Real-time generation means faster releases and better testing."

**For Researchers:**
> "Population-scale synthetic *statistics* without population-scale wait times." (Cost-per-population claims should be benchmarked before use; the "1 billion patients for the cost of a coffee" line was based on unmeasured cost figures and is removed.)

**For Compliance Officers:**
> "True synthetic data with zero privacy risk. No source data, no re-identification, no HIPAA concerns."

---

## 7. Monetization Model

### 7.1 Pricing Tiers

| Tier | Monthly Volume | Price | Target Customer |
|------|---------------|-------|-----------------|
| **Free** | 0-10M patients | $0 | Developers, students, POCs |
| **Starter** | 10-100M patients | $99/month | Small teams, startups |
| **Pro** | 100M-1B patients | $499/month | Mid-market, research labs |
| **Enterprise** | Unlimited | Custom ($5K-50K/month) | Large healthcare orgs |

### 7.2 Value Metric Justification

We price on **patient volume** because:
1. It's intuitive and predictable for customers
2. It scales with value delivered
3. It allows generous free tier for adoption
4. Our marginal cost is near-zero, enabling high margins at all tiers

### 7.3 Upsell Opportunities

| Feature | Free | Starter | Pro | Enterprise |
|---------|------|---------|-----|------------|
| FHIR R4 export | ✓ | ✓ | ✓ | ✓ |
| Custom demographics | - | ✓ | ✓ | ✓ |
| Custom conditions | - | - | ✓ | ✓ |
| On-premise deployment | - | - | - | ✓ |
| SLA guarantee | - | - | 99.9% | 99.99% |
| Dedicated support | - | - | - | ✓ |

### 7.4 Competitive Undercut Strategy

**Objective:** Make ChronoSynthea the default choice by offering more free than competitors charge for.

| Competitor Paid Tier | ChronoSynthea Equivalent |
|---------------------|-------------------------|
| Synthea hosting: $500/month for 1M patients/month | **Free** (10M patients/month) |
| MDClone: $50K/month minimum | **$499/month** for equivalent volume |
| Syntegra: $100K+/year | **$5,988/year** (Pro annual) |

---

## 8. Cost Structure & Unit Economics

> **Caveat:** the cost and unit-economics tables in this section are illustrative business projections. They were originally derived from an unmeasured ~600ms/1M figure; the measured stats-only time on the reference machine is ~0.11–0.12s/1M (full-stats ~0.25–0.27s), but cloud-instance pricing, output serialization, and storage were **not** benchmarked. Re-derive these from real measurements before quoting them.

### 8.1 Compute Cost Analysis

For 1M patients (stats-only) at ~0.11–0.12s measured on a Ryzen 7 5800X — cloud-hardware timing not separately measured:

| Resource | Usage | Cost |
|----------|-------|------|
| **CPU** | 0.0002 CPU-hours | $0.000006 |
| **Memory** | 500 MB peak (ephemeral) | ~$0 |
| **Network** | ~1 GB output (if stored) | $0.02 |
| **Storage** | Optional (S3) | $0.023/GB/month |

**Total compute cost per 1M patients: ~$0.00001 to $0.02** (depending on output storage)

### 8.2 Unit Economics by Tier

| Tier | Revenue | Volume | Compute Cost | Gross Margin |
|------|---------|--------|--------------|--------------|
| Free | $0 | 10M | $0.0002 | N/A (acquisition) |
| Starter | $99 | 100M | $0.002 | **99.998%** |
| Pro | $499 | 1B | $0.02 | **99.996%** |
| Enterprise | $10,000 | 50B | $1.00 | **99.99%** |

### 8.3 Infrastructure Recommendations

**Option A: Serverless (AWS Lambda / Cloud Run)**

Best for: API-first SaaS with variable demand

```
Request → API Gateway → Lambda (Rust) → Response/S3
                            ↓
                    ~600ms for 1M patients
                    ~$0.0002 per invocation
```

Pros:
- Zero idle cost
- Automatic scaling
- Sub-second cold starts with Rust

Cons:
- 15-minute max execution (3+ billion patients per invocation)
- Payload size limits for response body

**Option B: Container-Based (ECS Fargate / Cloud Run)**

Best for: High-volume, predictable workloads

```
Load Balancer → Fargate Tasks (min: 1, max: 100) → Response
                            ↓
                    ~100ms for 1M patients (warm)
                    ~$30/month minimum
```

Pros:
- No cold starts
- Consistent performance
- Larger memory available

Cons:
- Minimum cost even at zero usage

**Option C: Edge Deployment (Cloudflare Workers / Fastly)**

Best for: Global low-latency API

```
Edge PoP → WASM Worker (Rust compiled) → Response
                            ↓
                    Sub-50ms global latency
                    Cloudflare pricing (~$0.50/M requests)
```

Pros:
- Lowest latency globally
- Edge caching for repeated requests
- DDoS protection included

Cons:
- WASM has some performance overhead
- Memory constraints

### 8.4 Projected P&L (Year 1)

| Line Item | Q1 | Q2 | Q3 | Q4 | Year 1 |
|-----------|----:|----:|----:|----:|-------:|
| **Customers** | 50 | 150 | 400 | 1,000 | - |
| **MRR** | $5K | $20K | $60K | $150K | - |
| **ARR (Exit)** | - | - | - | - | **$1.8M** |
| **Compute Costs** | $50 | $200 | $600 | $1,500 | $2,350 |
| **Gross Margin** | 99% | 99% | 99% | 99% | **99%** |

---

## 9. References & Citations

[1] Sweeney, L. (2000). "Simple Demographics Often Identify People Uniquely." Carnegie Mellon University, Data Privacy Working Paper 3. *Demonstrated that 87% of US population can be uniquely identified by ZIP, birthdate, and gender.*

[2] Walonoski, J., et al. (2017). "Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record." Journal of the American Medical Informatics Association, 25(3), 230-238. DOI: 10.1093/jamia/ocx079

[3] Lemire, D., & Boytsov, L. (2015). "Decoding billions of integers per second through vectorization." Software: Practice and Experience, 45(1), 1-29. *Foundational work on SIMD optimization for data processing.*

[4] Emery, D. (2018). "Bumpalo: A fast bump allocation arena for Rust." https://github.com/fitzgen/bumpalo *Arena allocator enabling O(1) batch deallocation.*

[5] Vose, M. D. (1991). "A linear algorithm for generating random numbers with a given distribution." IEEE Transactions on Software Engineering, 17(9), 972-975. *O(n) preprocessing, O(1) sampling algorithm.*

[6] Herlihy, M., & Shavit, N. (2012). "The Art of Multiprocessor Programming." Morgan Kaufmann. *Foundational text on lock-free concurrent data structures.*

[7] Matsakis, N., & Klock, F. (2014). "The Rust Programming Language." *Rust's ownership model enables zero-cost abstractions for memory safety.*

[8] Kullback, S., & Leibler, R. A. (1951). "On Information and Sufficiency." The Annals of Mathematical Statistics, 22(1), 79-86. *Defines KL divergence for measuring distributional difference.*

[9] Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 50(302), 157-175.

[10] Grand View Research (2024). "Synthetic Data Generation Market Size Report, 2024-2030." *Market sizing and growth projections.*

[11] U.S. Food and Drug Administration (2021). "Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan." *FDA guidance encouraging synthetic data for AI validation.*

[12] Chen, R., et al. (2023). "Synthetic Data in Healthcare: A Systematic Review." npj Digital Medicine, 6, 89. *Comprehensive review of synthetic data approaches and validation methods.*

[13] El Emam, K., et al. (2020). "Evaluating the Risk of Re-identification of Patients from Hospital Prescription Records." BMC Medical Informatics and Decision Making, 20, 113. *Demonstrates re-identification risks in de-identified data.*

---

## Appendix A: Benchmark Reproducibility

All benchmarks can be reproduced with:

```bash
# Clone and build
git clone https://github.com/chronomancy-io/chronosynthea
cd chronosynthea
cargo build --release

# Run validation tests
cargo test --package chronosynthea-mss --test java_validation --release -- --nocapture

# Run performance tests
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_high_volume_generation_with_validation --nocapture   # stats-only
cargo test --package chronosynthea-mss --test java_validation --release \
    -- test_full_generation_performance --nocapture              # full-stats
```

Measured output (AMD Ryzen 7 5800X, 16 threads, Rust 1.94.0, 2026-06-04):

```
Statistical Comparison (n=100000)
  KL Divergence: -0.006132
  Max Deviation: 0.0031
  Chi-Squared:   181.17
  Passed:        true

# stats-only path
Generated 1M patients in 110.79ms (9.03M patients/sec)
# full-stats path
Full stats generation (avg of 5 runs): 200K patients in 52.26ms (3827.01K patients/sec)
Projected time for 1M patients: 261.30ms
```

(Throughput here is the count-generation path, which materializes no per-patient records; see PERFORMANCE.md.)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **MSS** | Minimally Sufficient Statistic—the minimal set of statistics needed to reproduce a distribution |
| **SIMD** | Single Instruction, Multiple Data—CPU instructions that operate on multiple values simultaneously |
| **FHIR** | Fast Healthcare Interoperability Resources—healthcare data exchange standard |
| **KL Divergence** | Kullback-Leibler divergence—measure of how one probability distribution differs from another |
| **Synthea** | Open-source synthetic patient generator from MITRE Corporation |
| **Arena Allocation** | Memory allocation strategy where objects share a memory region and are freed together |
| **Vose Alias** | Algorithm for O(1) sampling from discrete probability distributions |

---

*Document Version: 2.0.0 (MSS-honesty pass)*  
*Last Updated: 2026-06-04*  
*Classification: Internal Strategy Document — contains forward-looking projections; technical claims reconciled to measured numbers in PERFORMANCE.md*
