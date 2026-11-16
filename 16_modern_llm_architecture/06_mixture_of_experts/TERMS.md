# Terms — Mixture of Experts

Folder: `16_modern_llm_architecture/06_mixture_of_experts` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Mixture of Experts (MoE)](#mixture-of-experts-moe) · [Expert](#expert) · [Router (gate)](#router-gate) · [Top-k gating](#top-k-gating) · [Shared expert](#shared-expert) · [Total vs active parameters](#total-vs-active-parameters) · [Load imbalance and expert collapse](#load-imbalance-and-expert-collapse) · [Auxiliary load-balancing loss](#auxiliary-load-balancing-loss) · [Auxiliary-loss-free balancing](#auxiliary-loss-free-balancing) · [Capacity factor](#capacity-factor) · [register_buffer](#register_buffer) · [torch.topk and gather](#torchtopk-and-gather) · [torch.bincount](#torchbincount) · [Expert parallelism](#expert-parallelism)

---

## Mixture of Experts (MoE)  `(concept)`
**1. Definition:** A layer with many expert MLPs and a router that sends each token to a few of them.
**2. In plain words:** A hospital with specialists; each patient sees two.
**3. Example to quote:** > "MoE grows total parameters without growing compute per token."
**Seen in:** `main.py`, `MoE`

## Expert  `(concept)`
**1. Definition:** One of the parallel MLPs in an MoE layer, usually a SwiGLU network.
**2. In plain words:** A small ordinary MLP.
**3. Example to quote:** > "Our expert has 24,576 parameters."
**Seen in:** `main.py`, `Expert`

## Router (gate)  `(concept)`
**1. Definition:** A linear layer mapping a token vector to one score per expert.
**2. In plain words:** The receptionist.
**3. Example to quote:** > "Our router has 512 parameters for 8 experts at d=64."
**Seen in:** `main.py`, `self.router`

## Top-k gating  `(concept)`
**1. Definition:** Select the k experts with the highest router scores and renormalise their weights.
**2. In plain words:** Choose the best two specialists.
**3. Example to quote:** > "Mixtral top-2, DeepSeek-V3 top-8."
**Seen in:** `main.py`, `topk`

## Shared expert  `(concept)`
**1. Definition:** An expert applied to every token in addition to the routed ones.
**2. In plain words:** A generalist everyone sees.
**3. Example to quote:** > "DeepSeek-V3 has 1 shared and 256 routed experts."
**Seen in:** `main.py`, `self.shared`

## Total vs active parameters  `(concept)`
**1. Definition:** Total = all weights stored; active = weights used per token.
**2. In plain words:** Memory scales with total, compute with active.
**3. Example to quote:** > "DeepSeek-V3: 671B total, 37B active."
**Seen in:** `main.py`, section 1

## Load imbalance and expert collapse  `(concept)`
**1. Definition:** A few experts receive most tokens while others starve.
**2. In plain words:** Some specialists are overloaded and some idle.
**3. Example to quote:** > "Untrained router: loads from 4 to 1,263 slots of 4,096."
**Seen in:** `main.py`, section 2

## Auxiliary load-balancing loss  `(concept)`
**1. Definition:** Switch/GShard loss n x sum(f_e x P_e) added to training to spread tokens evenly.
**2. In plain words:** A fine for lopsided traffic.
**3. Example to quote:** > "It equals 1.0 at perfect balance (1.505 for the untrained router)."
**Seen in:** `main.py`, `balance_loss`

## Auxiliary-loss-free balancing  `(concept)`
**1. Definition:** DeepSeek-V3: a per-expert bias used only to choose the top-k, increased for underloaded and decreased for overloaded experts, with no extra loss.
**2. In plain words:** A thermostat on expert popularity.
**3. Example to quote:** > "The router weights stay untouched: True."
**Seen in:** `main.py`, section 5

## Capacity factor  `(concept)`
**1. Definition:** Max tokens per expert = capacity_factor x tokens x k / n_experts; overflow is dropped.
**2. In plain words:** A size limit per specialist.
**3. Example to quote:** > "30.2% of slots would drop at 1.25 for the untrained router."
**Seen in:** `main.py`, section 3

## register_buffer  `(syntax)`
**1. Definition:** Registers a tensor in a module that is saved and moved with it but is not a trainable parameter.
**2. In plain words:** State that no optimizer touches.
**3. Example to quote:** > "The balancing bias is a buffer."
**Seen in:** `main.py`, `MoE.__init__`

## torch.topk and gather  `(library)`
**1. Definition:** topk returns the k largest values and indices; gather picks values at given indices.
**2. In plain words:** Find the best two, then fetch their scores.
**3. Example to quote:** > "probs.gather(-1, top_idx)."
**Seen in:** `main.py`, `forward`

## torch.bincount  `(library)`
**1. Definition:** Counts occurrences of each non-negative integer.
**2. In plain words:** Tokens per expert.
**3. Example to quote:** > "bincount(top_idx.flatten(), minlength=8)."
**Seen in:** `main.py`, `counts`

## Expert parallelism  `(concept)`
**1. Definition:** Placing different experts on different GPUs and exchanging tokens with all-to-all communication.
**2. In plain words:** Each specialist works in a different building.
**3. Example to quote:** > "The busiest expert sets the speed."
**Seen in:** `WALKTHROUGH.md`

