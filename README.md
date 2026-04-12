# 4x RTX 5090 Open-Air Rig for Local LLM Inference

> A filmmaker's journey building a 128 GB VRAM beast for local AI inference and fine-tuning on a prosumer budget.

## Why This Build

I'm a filmmaker working with AI-generated imagery (LoRA training, Flux/Deforum/ComfyUI) and increasingly running large language models locally for creative research. Cloud GPU costs add up fast, and I needed a machine that could:

- Run **70B+ parameter models** quantized (GGUF Q4/Q5) at interactive speeds
- Fine-tune LoRAs and run ComfyUI workflows with large batch sizes
- Handle multi-model inference via vLLM or Ollama with tensor parallelism
- Serve as a persistent local inference server accessible over my Tailscale network

The answer: **4x RTX 5090** (128 GB total VRAM) on an open-air frame, built incrementally with parts I already had from previous builds and some smart sourcing.

---

## Full Parts List

| Component | Model | Qty | Unit Price (TTC) | Total (TTC) |
|-----------|-------|-----|-------------------|-------------|
| **CPU** | AMD Ryzen 9 9950X (4.3 / 5.7 GHz) | 1 | €649.95 | €649.95 |
| **Motherboard** | ASUS ProArt X870E-CREATOR WIFI | 1 | €549.95 | €549.95 |
| **RAM** | G.Skill Flare X5 Low Profile 96 GB (2×48 GB) DDR5-6000 CL30 | 2 | €1,299.95 | €2,599.90 |
| **GPU** | MSI GeForce RTX 5090 32G VANGUARD SOC | 4 | ~€2,200* | ~€8,800* |
| **Storage** | Samsung 9100 PRO 8 TB M.2 NVMe PCIe 5.0 | 2 | €1,249.95 | €2,499.90 |
| **CPU Cooler** | Noctua NH-D15 Chromax Black | 1 | €149.95 | €149.95 |
| **AIO (spare)** | Cooler Master MasterLiquid 240 Core II ARGB | 1 | €79.95 | €79.95 |
| **Fan** | Noctua NF-A12x25 PWM chromax.black.swap | 1 | €49.96 | €49.96 |
| **Fan** | Noctua NF-A12x15 PWM chromax.black.swap | 1 | €34.96 | €34.96 |
| **PSU (ATX)** | Corsair SF1000 SFX 80+ Platinum | 1 | €214.95 | €214.95 |
| **PSU (server)** | HP 1200W Server PSU (80+ Platinum) | 2 | ~€40 (used) | ~€80 |
| **PSU (octopus)** | Chinese 1600W breakout PSU (AliExpress) | 1 | ~€60 | ~€60 |
| **Case** | Open-air mining/GPU rig frame (6-8 GPU) | 1 | ~€120 | ~€120 |
| **Risers** | PCIe 5.0 risers (LINKUP AVA5 or equivalent) | 4 | ~€70 | ~€280 |
| **Fans (extra)** | Arctic P12/P14 PWM (for GPU airflow) | 6 | ~€10 | ~€60 |
| **Breakout boards** | HP PSU breakout + 16-pin cables | 2 | ~€35 | ~€70 |

**\*GPU prices as of mid-2026, subject to stock/availability.**

### Total Estimated Cost

| Category | Cost |
|----------|------|
| Components (new, TTC) | ~€15,830 |
| Additional parts (risers, fans, cables) | ~€410 |
| **Total** | **~€16,240** |

Already owning the server PSUs, open-air frame, and octopus PSU saved ~€260.

---

## Architecture Deep Dive

### The AM5 PCIe Lane Problem

This is the most important thing to understand about this build, and what most guides gloss over.

**AM5 (Ryzen 9000 series) provides 28 PCIe 5.0 lanes from the CPU:**

| Allocation | Lanes | Generation |
|------------|-------|------------|
| GPU slots (PCIEX16_1 + _2) | 16 | PCIe 5.0 |
| M.2 NVMe (2 slots) | 8 | PCIe 5.0 |
| Chipset uplink | 4 | PCIe 5.0 |

The **X870E chipset** then adds 36-44 PCIe 4.0 lanes for secondary expansion slots, additional M.2, USB, SATA, etc. But the CPU itself provides only **16 lanes** for GPU slots total.

**What the ASUS ProArt X870E-CREATOR WIFI actually offers:**

- **Slot 1 (PCIEX16_1):** PCIe 5.0 x16 from CPU — full bandwidth (~64 GB/s bidirectional)
- **Slot 2 (PCIEX16_2):** PCIe 5.0 from CPU — shares the same 16-lane pool. When both slots are populated, they auto-bifurcate to **x8/x8**
- **Slot 3:** PCIe 4.0 x4 from chipset — usable but limited (~8 GB/s)

**Key insight:** Slots 1 and 2 share the CPU's 16 GPU lanes. You get either 1 GPU at x16 or 2 GPUs at x8/x8. The BIOS also supports bifurcation modes including x8/x4/x4 and even x4/x4/x4/x4 on a single slot (for specialized riser splitters).

### How to Actually Fit 4 GPUs

For a 4-GPU setup on this board, the realistic lane allocation is:

**Option A — Best balance (recommended):**
- Slot 1 + Slot 2: x8/x8 Gen5 → **GPU 1 + GPU 2** (via risers, since the cards are too thick to sit side by side)
- Slot 3 (chipset): x4 Gen4 → **GPU 3**
- **GPU 4:** Requires a bifurcation riser on Slot 1 or 2 (splitting x8 into x4/x4), or using a PCIe switch. Alternatively, some builders use the M.2 slot with an M.2-to-PCIe x4 adapter for the 4th card.

**Option B — Maximum density via bifurcation:**
- Slot 1: x4/x4/x4/x4 bifurcation → **4 GPUs at x4 Gen5** each (~16 GB/s bidirectional)
- This requires a quad-bifurcation riser card and BIOS support

**Option C — Practical compromise (most common in real builds):**
- Slot 1: x8 Gen5 → **GPU 1**
- Slot 2: x8 Gen5 → **GPU 2**
- Slot 3 (chipset): x4 Gen4 → **GPU 3**
- **GPU 4:** on a riser via a bifurcation split of Slot 1 or Slot 2 (x4 Gen5), accepting one GPU drops to x4

All options work for LLM inference. Here's why:

### Why This Barely Matters for LLM Inference

PCIe bandwidth is mostly irrelevant for inference workloads. Here's why:

1. **Model loading is a one-time operation.** You load 70B weights into VRAM once. Even at x4 Gen4 (~8 GB/s), loading 32 GB of weights takes ~4 seconds instead of ~0.5 seconds. You do this once.

2. **Token generation is GPU-compute bound**, not PCIe-transfer bound. The bottleneck is matrix multiplication inside the GPU, not data moving over the bus.

3. **Multi-GPU tensor parallelism** (vLLM, llama.cpp) does communicate between GPUs during inference, but the volume is small compared to training. On AM5 without NVLink, inter-GPU communication goes through CPU/PCIe — slower than NVLink, but sufficient for inference at dozens of tokens/second.

4. **Benchmarks show 0-4% difference** between Gen5 x16 and Gen4 x4 for LLM inference throughput on single-GPU. For multi-GPU, the gap can widen to 5-15% depending on model size and parallelism strategy, but it's rarely the bottleneck.

**The Threadripper Alternative:** If PCIe lanes genuinely become a bottleneck (e.g., heavy training workloads with large gradient syncs), the proper platform is AMD Threadripper PRO 7000 with WRX90 (128 PCIe 5.0 lanes — 32 per GPU for 4 cards, 8-channel DDR5 up to 2 TB). Confirmed quad-5090 builds exist on Threadripper with full x16 per GPU and custom liquid cooling. But the CPU alone costs ~€3,000+ and boards are ~€1,000+. For inference-focused builds, AM5 is the pragmatic choice — you sacrifice lane width but keep 95%+ of the inference throughput at a fraction of the platform cost.

---

## Will All 4 GPUs Run at the Same Speed?

**Short answer: no.** This is the most common question about multi-GPU on AM5, and it deserves a straight answer.

### The Asymmetry

On the ProArt X870E, the 4 GPUs get unequal PCIe bandwidth:

| GPU | Slot | PCIe Link | Bandwidth | Relative Speed |
|-----|------|-----------|-----------|----------------|
| GPU 1 | CPU Slot 1 | x8 Gen5 | ~32 GB/s | Fast |
| GPU 2 | CPU Slot 2 | x8 Gen5 | ~32 GB/s | Fast |
| GPU 3 | Chipset Slot | x4 Gen4 | ~8 GB/s | Slower |
| GPU 4 | Bifurcation/adapter | x4 Gen5 or Gen4 | ~8-16 GB/s | Slower |

The GPUs themselves (CUDA cores, GDDR7 VRAM, clock speeds) are identical. It's the **pipe between GPU and CPU** that varies.

### When Does It Matter?

**Single-model inference (Ollama, one model on one GPU):** Not at all. The model loads into VRAM once, then it's pure GPU-internal compute. A GPU at x4 Gen4 generates tokens at the exact same speed as one at x8 Gen5.

**Multi-GPU tensor parallelism (vLLM, llama.cpp with 4 GPUs on one model):** It matters. GPUs exchange activation data at every token. The slowest link (x4 Gen4 at ~8 GB/s) becomes the bottleneck because faster GPUs wait for it. Estimated impact: **5-15% lower throughput** vs. 4 GPUs at equal bandwidth.

**Multi-model serving (different models on different GPUs):** Doesn't matter. Each GPU works independently. Put your most-used model on a fast-lane GPU if you want optimal single-model latency.

### Three Ways to Equalize

**Option 1: Threadripper (the real answer) — +€3,500-4,000**

AMD Threadripper PRO 7975WX + WRX90 board gives 128 PCIe 5.0 lanes. Each GPU gets a full x16 slot. No bifurcation, no compromises, no asymmetry. This is what commercial AI workstations (BIZON, Lambda, Puget Systems) use. If you're spending €8,800 on GPUs, the extra €3,500 for Threadripper is arguably worth it for a properly balanced system.

| Component | Model | Est. Price |
|-----------|-------|------------|
| CPU | AMD Threadripper PRO 7975WX (32C/64T) | ~€3,500 |
| Motherboard | ASUS WRX90E SAGE or Gigabyte WRX90 AERO D | ~€1,000-1,200 |
| RAM | Your DDR5 kits work (4 of 8 channels populated) | €0 (reuse) |

**Option 2: Force all slots to Gen4 x4 (free, all equal)**

In BIOS, force every PCIe slot to Gen4. All 4 GPUs now run at the same ~8 GB/s. You lose absolute bandwidth but gain **symmetry** — no GPU waits for a slower neighbor during tensor parallelism. For inference, the ~8 GB/s pipe is rarely the bottleneck anyway.

This is the cheapest and simplest equalization strategy. Counterintuitive but effective.

**Option 3: Software compensation with tensor-split (free, smart)**

llama.cpp and vLLM both support weighted tensor splitting. Give more model layers to fast-lane GPUs and fewer to slow-lane GPUs:

```bash
# llama.cpp: weight distribution across 4 GPUs
# GPUs 1-2 (x8 Gen5) get more layers, GPUs 3-4 (x4) get fewer
./llama-server \
  -m models/llama-3.1-70b-Q5_K_M.gguf \
  --n-gpu-layers 99 \
  --tensor-split 30,30,20,20

# vLLM: uses tensor parallelism automatically
# Less fine-grained control, but handles asymmetry reasonably well
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```

The slow GPU receives less work, finishes at the same time as the fast ones, and nobody waits. This is the most pragmatic approach on AM5.

### Comparison Table

| Approach | Extra Cost | All 4 Equal? | Overall Throughput |
|----------|-----------|--------------|-------------------|
| Threadripper WRX90 | +€3,500-4,000 | Yes (4×x16 Gen5) | 100% |
| AM5 + force Gen4 x4 | €0 | Yes (4×x4 Gen4) | ~92-95% |
| AM5 + tensor-split tuning | €0 | Compensated | ~90-95% |
| AM5 + do nothing | €0 | No | ~85-95% |

### My Recommendation

For **inference** (my primary use case): stay on AM5, use `--tensor-split` to compensate. The 5-10% difference vs. Threadripper doesn't justify €3,500+.

For **training** (LoRA/QLoRA is fine, full fine-tuning is not): if you ever need to do serious multi-GPU training with large gradient syncs, sell the AM5 combo and upgrade to Threadripper. The PCIe asymmetry hurts much more during training than inference.

---

## PCIe 5.0 Risers: The Full Story

### Why PCIe 5.0 Risers Have a Bad Reputation

PCIe 5.0 runs at **32 GT/s** (gigatransfers per second), double Gen4's 16 GT/s. At these frequencies:

- **Signal integrity degrades faster** with cable length, bends, and poor shielding
- **Cheap risers introduce noise** that causes link negotiation failures, crashes under load, or automatic downgrade to Gen4/Gen3
- The **RTX 5090's multi-PCB design** (especially Founders Edition) already acts as an internal riser of sorts, compounding signal integrity challenges

Reviewers like der8auer, Igor's Lab, and Hardware Canucks documented instabilities with budget risers on RTX 50-series cards, especially when BIOS is set to "Auto" PCIe generation.

### But It's Not a Fatality

With **quality PCIe 5.0 risers**, it works fine. The key factors:

| Factor | Good | Bad |
|--------|------|-----|
| Cable length | 15-30 cm | >40 cm |
| Shielding | Multi-layer, full coverage | Thin or partial |
| Connectors | Gold-plated, reinforced | Loose or cheap |
| Bending | Gentle curves only | Sharp folds, crimped |
| Brand | LINKUP, Thermaltake, ADT-Link | No-name AliExpress |

### My Recommendation

**Primary choice:** LINKUP AVA5 PCIe 5.0 (20-30 cm versions). As of 2026, these are the most widely validated for RTX 5090 builds. They maintain full bandwidth (128 GB/s on 3DMark tests) even with moderate bends. Price: ~€55-80 each.

**Budget alternative:** Quality PCIe 4.0 risers + force Gen4 in BIOS. Performance loss: 1-3% in inference. Stability gain: significant. Price: ~€25-40 each. Many multi-5090 builds run this way without issues.

**Fallback strategy:** Start with Gen5 risers, set BIOS to Auto. If you see instabilities (GPU not detected, crashes under sustained load, GPU-Z showing unexpected Gen3/4 link), force Gen4 in BIOS. You lose almost nothing for LLM workloads.

### Validation Protocol

After assembling, verify with:

```bash
# Check PCIe link speed and width
nvidia-smi -q | grep -A 5 "PCI"

# Stress test: sustained inference for 24h
# (use your actual workload, not synthetic benchmarks)
ollama run llama3.1:70b-instruct-q4_K_M

# Monitor thermals during stress
watch -n 1 nvidia-smi
```

Check that GPU-Z shows the expected generation (Gen5 x8 or Gen4 x16/x8) and that it doesn't fluctuate during load.

---

## Power Delivery: Multi-PSU Strategy

### Power Budget

| Component | Peak Draw |
|-----------|-----------|
| RTX 5090 Vanguard SOC (×4) | 4 × 575W = 2,300W |
| Ryzen 9 9950X (PBO) | ~200W |
| RAM, SSDs, fans, board | ~80W |
| **Total peak** | **~2,580W** |
| **With transient spikes** | **~2,800-3,000W** |

A single ATX PSU cannot handle this. You need a multi-PSU setup.

### My PSU Configuration

```
PSU 1: HP 1200W Server (80+ Platinum)
  → Motherboard 24-pin (via ATX breakout)
  → CPU 8-pin
  → GPU 1 (16-pin native)
  → GPU 2 (16-pin native)

PSU 2: HP 1200W Server (80+ Platinum)
  → GPU 3 (16-pin native)
  → GPU 4 (16-pin native)

PSU 3: Corsair SF1000 (backup / headroom)
  → Available if load balancing needed
  → Or used for fans + peripherals
```

### HP Server PSU Notes

HP ProLiant 1200W units (DPS-1200FB or HSTNS-PL11) are **excellent** for multi-GPU builds:

- **80+ Platinum efficiency** — designed for 24/7 datacenter operation
- **12V single rail at 100A** — clean, high-current output
- **Dirt cheap** on the used market (~€30-50 each)
- Widely used in mining and AI rig builds since 2017

**What you need to make them work:**

- **Breakout boards** (~€15-30 on AliExpress/Amazon) that convert the server PSU connector into standard ATX/PCIe cables
- **Quality 16-pin (12V-2×6) cables** for the RTX 5090. Do NOT use cheap Molex-to-16-pin adapters — these can melt under 575W sustained load. Get cables rated for the current.
- A way to **sync startup** between PSUs (some breakout boards handle this, or use a jumper wire on the ATX 24-pin green wire)

### The Octopus PSU (AliExpress 1600W)

These "pieuvre" PSUs are essentially server PSU breakout boards with built-in cables, often sold for mining. They work, but:

- **Quality varies wildly** — test voltage stability under load before trusting it with 5090s
- Use as **supplementary** power, not primary
- Good for powering 1-2 GPUs while the HP units handle the rest

### Electrical Infrastructure

**Critical:** 4×5090 at full load draws ~2.5-3 kW from the wall. Standard European 16A / 230V circuits provide 3.68 kW max — you're close to the limit.

Recommendations:
- **Dedicated 20A or 32A circuit** to the rig
- Never share the circuit with space heaters, kettles, or other high-draw appliances
- A **UPS is optional but smart** — not for runtime, but for clean shutdown protection
- Consider **undervolting the GPUs** (see below) to reduce draw by 20-30% with minimal performance loss

---

## Cooling: Open-Air vs. Closed Case

### Why Open-Air Wins for This Build

The MSI Vanguard SOC is a **quad-slot card** (~76 mm thick, 336 mm long). Four of these in a closed case, even a full tower, creates a thermal nightmare:

- Cards would be physically touching or overlapping in standard ATX cases
- No airflow between GPUs
- Ambient case temperature rises to 50-60°C under sustained load
- GPU thermal throttling kicks in, reducing inference speed

**Open-air advantages:**

- Space each GPU 2-3 slot widths apart
- Direct ambient air access on all sides
- Easy to add directional fans between cards
- Trivial maintenance and card swaps
- 10-15°C cooler GPU temps vs. closed case (measured across similar builds)

### The Crypto Mining Approach (Closed Case + Industrial Fans)

Some miners use enclosed cases with **high-CFM industrial blowers** (200-300 mm, 100+ CFM) creating massive positive pressure. This works for:

- Dust-heavy environments (warehouses, garages)
- Forced hot-air exhaust through ducting
- 24/7/365 unattended operation

For a **home/studio LLM inference rig**, open-air is simpler, cheaper, and quieter. The industrial blower approach adds complexity and noise for marginal benefit in a clean indoor environment.

### My Cooling Setup

```
[INTAKE FANS]  →  [GPU 1]  →  [FAN]  →  [GPU 2]  →  [FAN]  →  [GPU 3]  →  [FAN]  →  [GPU 4]  →  [EXHAUST]

Bottom of frame: Motherboard + CPU (Noctua NH-D15)
Between each GPU: 1-2× 120mm fans (Arctic P12 or Noctua NF-A12x25)
Side/top: Additional 140mm fans for general airflow
```

**Tip:** Mount GPUs horizontally (flat) if your frame allows it. This prevents GPU sag and keeps the heatsinks working optimally (heat rises naturally).

### Undervolting: Free Performance

The RTX 5090 Vanguard SOC ships with aggressive factory overclock curves. Undervolting reduces power draw and heat significantly:

```bash
# Example: cap power to 80% (saves ~115W per card)
nvidia-smi -i 0 -pl 460  # GPU 0
nvidia-smi -i 1 -pl 460  # GPU 1
nvidia-smi -i 2 -pl 460  # GPU 2
nvidia-smi -i 3 -pl 460  # GPU 3
```

Expected results at 80% power limit:
- Power draw: ~460W vs. 575W per card (saves 460W total across 4 GPUs)
- Performance loss: 3-8% in inference (barely noticeable in tokens/s)
- Temperature drop: 5-10°C per card
- Total rig draw drops from ~2,600W to ~2,100W — much friendlier to your circuit

For long inference runs, this is the **single best optimization** you can make.

---

## BIOS Configuration

### Essential Settings

| Setting | Value | Why |
|---------|-------|-----|
| PCIe Generation | Gen 4 (or Auto → Gen 4 fallback) | Stability over bandwidth |
| XMP / EXPO | Enabled (DDR5-6000) | Use your RAM's rated speed |
| Resizable BAR | Enabled | Allows GPU to access full VRAM mapping |
| Above 4G Decoding | Enabled | Required for multi-GPU + large VRAM |
| SR-IOV | Disabled (unless using vGPU) | Simplicity |
| CSM | Disabled | UEFI boot, modern OS |
| PCIe Bifurcation | x8x8 on Slot 1 (if available) | Allows 2 GPUs on CPU lanes |
| C-States | Enabled | Power saving when idle |
| PBO | Enabled, eco mode optional | Balance CPU perf vs. heat |

### RAM Configuration

With 4×48 GB (192 GB total), all four DIMM slots are populated:

- Slots **A2 + B2** first (dual-channel primary) — your first kit
- Slots **A1 + B1** second — your second kit
- XMP/EXPO profile should detect DDR5-6000 CL30 automatically
- If instability occurs with 4 DIMMs at 6000 MHz, drop to 5600 MHz — 4-DIMM configs are harder to run at high frequencies on AM5

**Why 192 GB RAM matters:** For LLM inference, system RAM is used for:
- Model loading buffer before GPU transfer
- KV cache overflow (when VRAM is full)
- CPU offloading of model layers (llama.cpp `--n-gpu-layers` partial offload)
- Running ComfyUI, vLLM server, monitoring, and OS simultaneously

192 GB gives comfortable headroom for all of this. 96 GB would work but gets tight with multiple large models loaded.

---

## Software Stack

### Core Inference

```bash
# Ollama (easiest to start with)
ollama serve  # starts server
ollama run llama3.1:70b-instruct-q4_K_M  # fits in ~40 GB VRAM

# vLLM (better for serving + tensor parallelism)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90

# llama.cpp (best flexibility, GGUF format)
./llama-server \
  -m models/llama-3.1-70b-Q5_K_M.gguf \
  --n-gpu-layers 99 \
  --tensor-split 25,25,25,25  # even split across 4 GPUs
```

### What Can 128 GB VRAM Run?

| Model | Quantization | VRAM Usage | Fits? |
|-------|-------------|------------|-------|
| Llama 3.1 70B | Q4_K_M | ~42 GB | Yes (1 GPU spare) |
| Llama 3.1 70B | Q8_0 | ~75 GB | Yes (across 3 GPUs) |
| Llama 3.1 70B | FP16 | ~140 GB | No (need offload) |
| Mixtral 8×22B | Q4_K_M | ~80 GB | Yes (across 3 GPUs) |
| Llama 3.1 405B | Q4_K_M | ~230 GB | Partial (CPU offload) |
| DeepSeek-V3 | Q4 | ~200 GB | Partial (CPU offload) |
| Any 7-34B model | Q4-Q8 | 5-22 GB | Yes (single GPU) |

The sweet spot is **70B Q5-Q8** models — they run entirely in VRAM across 2-4 GPUs with fast inference.

### Image/Video Generation

```bash
# ComfyUI (already configured on my Windows GPU machines)
# 4×5090 = massive batch sizes, fast LoRA training
# SDXL/Flux workflows run 4× faster than single GPU
```

### Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Detailed power/temp monitoring
nvidia-smi dmon -s pucvmet -d 5

# Per-GPU utilization for multi-model serving
gpustat --watch
```

---

## Network Integration

This rig joins my existing Tailscale mesh:

| Machine | Role | Tailscale IP |
|---------|------|-------------|
| MacBook Air M3 | Primary workstation | 100.118.137.41 |
| Mac Mini M4 | Server (Paris) | 100.84.223.88 |
| **4×5090 Rig** | GPU compute | TBD |
| PC Windows (existing) | GPU + ComfyUI | 100.82.222.123 |

Ollama or vLLM exposes an API on the Tailscale network, accessible from any machine:

```bash
# From MacBook, query the 4×5090 rig
curl http://<rig-tailscale-ip>:11434/api/generate \
  -d '{"model": "llama3.1:70b", "prompt": "..."}'
```

---

## Remaining Budget (What I Still Need to Buy)

Already owned: open-air frame, HP server PSUs ×2, octopus PSU, Corsair SF1000, extra Noctua fans.

| Item | Est. Cost | Priority |
|------|-----------|----------|
| 4× PCIe 5.0 risers (LINKUP AVA5 20-30cm) | €280-350 | Critical |
| 4-6× Arctic P12 PWM fans | €50-70 | High |
| 2× HP breakout boards + 16-pin cables | €60-80 | High |
| Misc cables, zip ties, standoffs | €20-30 | Medium |
| **Total remaining** | **€410-530** | |

---

## Honest Caveats

### What I Would Change If Starting Over

1. **Threadripper PRO 7000 WRX90** instead of AM5. The PCIe lane situation on AM5 is workable but not ideal for 4 GPUs. Threadripper gives 128 PCIe 5.0 lanes (32 per GPU, so 4 cards at full x16 each without bifurcation hacks), 8-channel DDR5 (up to 2 TB), and is designed for exactly this use case. Confirmed multi-5090 builds exist on Threadripper with custom liquid cooling. Cost difference: ~€3,000-4,000 more for CPU+board. Worth it if this is a production machine or if you plan to train (not just infer).

2. **ECC RAM** for long inference jobs. Consumer DDR5 can have rare bit flips during 24+ hour runs. ECC catches these. Available on Threadripper; technically supported but not validated on most AM5 boards.

3. **Dedicated 32A circuit** from the start. I'm running close to the limit of a standard European 16A / 230V circuit (3.68 kW max). With undervolting it's fine, but at full power all 4 GPUs can trip the breaker.

4. **Reference-design / blower-style GPUs** instead of thick open-air cooler cards. The Vanguard SOC quad-slot design means physical spacing is a challenge even on open-air frames. Reference PCB cards are thinner, and blower-style coolers exhaust heat directionally rather than radiating into adjacent cards. Some 4-GPU builders specifically seek reference cards with waterblock compatibility for this reason.

### Known Risks

- **PCIe lane allocation is the #1 complexity.** AM5's 16 GPU lanes across 2 slots means you're juggling bifurcation, chipset slots, and possibly PCIe adapters to seat 4 cards. Test each GPU individually before the full 4-card config. If a GPU fails to initialize, it's likely a lane/slot issue, not a dead card.
- **PCIe riser stability** is the #2 risk. Budget risers + Gen5 + RTX 5090 = potential for frustrating intermittent crashes. Spend the money on quality risers.
- **HP server PSUs are loud.** The stock 40mm fans ramp to jet-engine levels under load. Common mod: replace stock fans with Noctua 40mm (NF-A4x20 PWM) for dramatic noise reduction while maintaining safe airflow.
- **4-DIMM DDR5-6000 on AM5** can be finicky. The IMC (integrated memory controller) on Zen 5 works harder with 4 populated DIMMs. If you get memory errors or POST failures, drop to 5600 MHz and tighten timings.
- **Dust** in open-air setups. Monthly cleaning with compressed air is mandatory. Keep the rig in a clean, ventilated room away from pets and fabric.
- **M.2 slot sharing with GPU lanes.** On the ProArt X870E, some M.2 slots share bandwidth with PCIe GPU slots. Check the manual for which M.2 port uses CPU lanes vs. chipset lanes. Prioritize chipset M.2 for your second SSD to avoid stealing GPU bandwidth.

### What This Build Is NOT For

- **Large-scale training** (pre-training, full fine-tuning of 70B+ models) — you need NVLink or InfiniBand for efficient gradient sync. This build is inference and LoRA/QLoRA focused.
- **Silence** — 4×5090 under load + server PSUs + extra fans = significant noise. This is a utility machine, not a living room PC.
- **Plug-and-play** — multi-PSU, risers, and open-air means more assembly, troubleshooting, and maintenance than a standard PC.

---

## Build Log / Assembly Order

1. Mount motherboard on open-air frame
2. Install CPU + NH-D15 cooler
3. Install 4×48 GB RAM (all slots)
4. Install 2× M.2 SSDs
5. Connect risers to PCIe slots (test one GPU first before all four)
6. Mount GPU 1 directly or on short riser — boot, install OS + drivers
7. Add GPUs 2-4 one at a time, testing stability after each
8. Connect PSUs (HP units via breakout boards)
9. BIOS: enable Above 4G, Resizable BAR, set PCIe gen, XMP
10. Install Ubuntu Server 24.04 or similar (headless, SSH)
11. Install NVIDIA drivers + CUDA toolkit
12. Install Ollama / vLLM / llama.cpp
13. Run 24h stress test with target model
14. Undervolt, verify temps, final tune

**Golden rule:** Add one GPU at a time. If something is unstable with 4 GPUs, remove GPUs until stable, then add back. Isolate the problem before debugging.

---

## Expected Performance

Based on comparable 4×5090 builds:

| Workload | Expected Performance |
|----------|---------------------|
| Llama 3.1 70B Q4_K_M (single GPU) | ~35-45 tok/s |
| Llama 3.1 70B Q4_K_M (4 GPU tensor parallel) | ~80-120 tok/s |
| Llama 3.1 70B Q8 (4 GPU) | ~50-70 tok/s |
| SDXL image generation | ~2-3 sec/image (512×512) |
| Flux.1 image generation | ~5-8 sec/image |
| LoRA training (SDXL) | ~4× faster than single GPU |

These are rough estimates. Actual numbers depend on quantization, batch size, prompt length, KV cache, and software version.

---

## Cost vs. Cloud: Is It Worth It?

### Cloud GPU Pricing (mid-2026 estimates)

| Provider | GPU Config | $/hour | Monthly (24/7) |
|----------|-----------|--------|-----------------|
| RunPod | 4×A100 80GB | ~$8-12/hr | ~$6,000-9,000 |
| Lambda | 4×H100 80GB | ~$12-16/hr | ~$9,000-12,000 |
| vast.ai | 4×RTX 5090 | ~$4-6/hr | ~$3,000-4,500 |
| This build | 4×RTX 5090 (owned) | ~€0.50/hr electricity | ~€360/month |

### Break-Even Analysis

Total build cost: ~€16,500. Monthly electricity at ~2 kW average (undervolted, not 24/7 full load): ~€360/month at €0.18/kWh.

Compared to renting 4×5090 on vast.ai at ~$5/hr:
- If you use the rig **8 hours/day**: cloud cost = ~$1,200/month vs. your electricity = ~€120/month
- **Break-even: ~14-16 months** of regular use
- After that, every month of usage is essentially free (minus electricity)

For a filmmaker/researcher who runs inference daily, fine-tunes LoRAs weekly, and needs a persistent local API server, this build pays for itself within the first year and a half.

### The Real Advantage: Availability and Privacy

Beyond cost, owning the hardware means:
- **No queue, no cold starts** — your models are always loaded and warm
- **Full data privacy** — nothing leaves your network
- **No API rate limits** — run as many requests as your GPUs can handle
- **Experimentation freedom** — try weird models, custom quantizations, and bleeding-edge software without per-hour anxiety

---

## Resources

- [LINKUP AVA5 PCIe 5.0 Risers](https://linkup.one) — validated for RTX 5090
- [Ollama](https://ollama.com) — simplest local LLM runtime
- [vLLM](https://github.com/vllm-project/vllm) — high-throughput inference with tensor parallelism
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — flexible GGUF inference
- [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface) — `nvidia-smi` for monitoring and power management
- [GPU-Z](https://www.techpowerup.com/gpuz/) — verify PCIe link speed and generation

---

## License

This build guide is shared for educational purposes. Hardware choices reflect my specific needs (filmmaker + AI researcher) and budget constraints. Your mileage may vary.

Built with help from Claude Code. Mistakes are mine.
