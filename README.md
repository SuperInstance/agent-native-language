# Agent Native Language

**Agents don't exchange JSON or text. They exchange spectral structure.**

The input and output ARE the tension graph.

## The Idea

Current agent communication:
```
Agent A → "Please analyze this data" → Agent B → {"result": 0.95} → Agent A
```

Agent-native communication:
```
Agent A → [Laplacian L_A + task vector t] → Agent B → [Laplacian L_B + result vector r] → Agent A
```

The Laplacian IS the message. The conservation ratio IS the confidence. The Fiedler vector IS the routing.

## Core Concepts

| Concept | What it means in agent comms |
|---|---|
| **Laplacian** | The agent's capability graph — its full state as a spectral object |
| **Eigenvalues** | Spectral fingerprint — identifies the agent's "shape" |
| **Conservation ratio** | Confidence in the message — how well-structured the agent's state is |
| **Fiedler vector** | Routing — which agents should receive and process this message |
| **Spectral alignment** | Compatibility — cosine similarity of eigenvalue spectra determines if an agent can handle a task |

## Quick Start

```bash
pip install numpy
python demo.py
```

## Architecture

```
spectral_message.py   — SpectralMessage: a message IS a Laplacian
comm_bus.py           — AgentNativeComm: the spectral communication bus
demo.py               — 4 agents communicating purely through spectral messages
```

## No JSON. No Text. Pure Mathematics.

Agents register with capability graphs. Tasks are projected onto eigenvectors. Routing uses the Fiedler vector. Confidence is a conservation ratio. Composition is block-diagonal Laplacian assembly.

## License

MIT
