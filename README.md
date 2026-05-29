# agent-native-language

A communication protocol where agents exchange spectral structure instead of text or JSON. The Laplacian IS the message. The conservation ratio IS the confidence. The Fiedler vector IS the routing.

## What This Gives You

- **SpectralMessage**: a message IS a Laplacian — encode agent state as a spectral object
- **Spectral communication bus**: agents register with capability graphs, send tasks as spectral projections, receive results as Laplacians
- **Conservation-based confidence**: the conservation ratio of the Laplacian tells you how well-structured the agent's response is
- **Fiedler-based routing**: tasks are routed to agents based on eigenvector alignment, not string matching
- **No serialization overhead**: pure NumPy linear algebra replaces JSON/text parsing

## Quick Start

```bash
pip install numpy
python demo.py
```

This runs 4 agents communicating through spectral messages — registering capabilities, sending tasks, receiving results, all without a single JSON object.

## Architecture

```
spectral_message.py   — SpectralMessage: a message IS a Laplacian
comm_bus.py           — AgentNativeComm: the spectral communication bus
demo.py               — 4 agents communicating purely through spectral messages
```

### Core Concepts

| Concept | What it replaces |
|---------|-----------------|
| Laplacian | JSON payload — encodes full capability structure |
| Eigenvalues | Type/schema — spectral fingerprint identifies the agent |
| Conservation ratio | Status code — confidence in the message |
| Fiedler vector | Routing table — which agents should handle this |
| Spectral alignment | Content negotiation — cosine similarity of spectra |

### The Protocol

```
# Traditional:
Agent A → "Please analyze this data" → Agent B → {"result": 0.95} → Agent A

# Agent-native:
Agent A → [Laplacian L_A + task vector t] → Agent B → [Laplacian L_B + result vector r] → Agent A
```

## How It Fits

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem. This is the communication layer for:

- **agent-spectrum-os** — the OS that schedules agents spectrally
- **agent-manifest-rs** — manifests define the capability graphs that become Laplacians
- **agent-handshake-rs** — traditional handshake can upgrade to spectral communication

## Testing

Integration tests via `demo.py` — 4 agents exchange spectral messages and assert successful communication, conservation ratios, and eigenvalue alignment.

## Installation

```bash
pip install numpy
```

Python 3 with NumPy. No other dependencies.
