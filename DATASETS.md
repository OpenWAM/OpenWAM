# Open-WAM Datasets

This document outlines the core datasets for training different layers of the Open-WAM architecture.

## Layer Definitions
- **Layer 2 (Robotic Video Adaptation / World Model)**: A generative video backbone adapted to understand robotic embodiment, multi-object interactions, physical common sense, and visual dynamics. **Input:** Video (+ Text/Language). **Output:** Video predictions.
- **Layer 3 (Action Policy)**: A specialized policy head attached to the Video World Model that maps visual representations or predictive latents into low-level robotic control. **Input:** Video + State. **Output:** Actions (7D pose + gripper, etc.).
- **Layer 4 (High-Level Planning / Cognitive Control)**: Long-horizon reasoning, task decomposition, and semantic understanding. **Input:** High-level goals (Language/Image). **Output:** Sub-tasks or high-level commands to Layer 3.

---

## 1. Datasets for Layer 2 (Video World Model / Pure Video)

For Layer 2, we rely entirely on **pure video sequences** (stripping explicit action labels if necessary) to teach the model general physics and embodiment priors.

### 🎯 Tier 1: Target Domain (评测域绝对对齐)
| Dataset | Link | Type | Size | Quality | Processing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Libero** | [Website](https://libero-project.github.io/) | Simulation | 130 Tasks, ~10k+ traj | Exact domain match for Layer 3 evaluation | Strip action labels, extract visual views (wrist/egocentric) to `.mp4` |

### 🦾 Tier 2: Embodiment (真实机器人视觉先验)
| Dataset | Link | Type | Size | Quality | Processing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DROID** | [Website](https://droid-dataset.github.io/) | Real Robot | 76k traj (350+ hours) | High res, Franka arm, rich scenes, diverse materials | Unpack HDF5/TFRecord, convert to `.mp4`, extract text instructions |
| **BridgeV2** | [Website](https://rail.eecs.berkeley.edu/datasets/bridge_v2/) | Real Robot | 25k+ traj | WidowX arm, consistent kitchen scenes, high policy success | Filter out bad crops, extract `.mp4` |
| **RH20T** | [Website](https://rh20t.github.io/) | Real Robot | ~110k traj | Extremely complex contacts (plug, unplug, twist caps) | Parse camera views, generate dense text captions |
| **Open X-Embodiment** | [Website](https://robotics-transformer-x.github.io/) | Real Robot | Massive | The largest aggregate robotics dataset | Filter heavily to extract high-resolution, interaction-heavy trajectories |

### 🌍 Tier 3: Physics & World (泛化物理常识)
| Dataset | Link | Type | Size | Quality | Processing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Epic-Kitchens 100** | [Website](https://epic-kitchens.github.io/2020) | Human Egocentric | 100 hours | Egocentric human hands, diverse tasks, state changes | Chunk videos (2-4s), run VLM for dense captions |
| **Something-Something v2** | [Website](https://developer.qualcomm.com/software/ai-datasets/something-something) | Human Egocentric | 220k short clips | Pure physical actions (pushing, tearing, dropping) | Native `.mp4` format, direct reuse of standard captions |

---

## 2. Datasets for Layer 3 (Action Policy)

Layer 3 is strictly evaluated on control metrics. It requires high-quality, perfectly aligned **(Video, Action)** pairs.

| Dataset | Type | Notes |
| :--- | :--- | :--- |
| **LIBERO-10 / 90** | Simulation | Primary benchmark. Ground truth 7D reference-relative EEF targets and gripper commands. |
| **DROID / BridgeV2** | Real Robot | Real-world manipulation. Reliable end-effector pose tracking and proprioceptive joint data mapped to visual frames. |

---

## 3. Datasets for Layer 4 (High-Level Planning / VLM)

Layer 4 relies on semantic data, VQA, and hierarchical planning traces to decompose complex user instructions.

| Dataset | Focus | Link |
| :--- | :--- | :--- |
| **EgoSchema** | Video QA / Reasoning | [Website](https://egoschema.github.io/) |
| **EgoTaskQA** | Intent Understanding | [Website](https://egotaskqa.github.io/) |
| **RoboVQA** | Robotics VQA | [via Open-X](https://robotics-transformer-x.github.io/) |
| **Language-annotated trajectories** | Sub-task breakdown | Extracted from Libero/DROID via LLMs (e.g., Gemini) |
