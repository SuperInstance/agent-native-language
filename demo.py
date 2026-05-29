#!/usr/bin/env python3
"""
Demo: 4 agents communicating purely through spectral messages.

No JSON schemas. No API specs. Just spectral mathematics as the language.
"""

import numpy as np
from spectral_message import SpectralMessage
from comm_bus import AgentNativeComm


def main():
    np.random.seed(42)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         AGENT NATIVE LANGUAGE — Spectral Communication         ║")
    print("║         No text. No JSON. Only Laplacians.                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    bus = AgentNativeComm()

    # ── Step 1: Register 4 agents with different capabilities ──
    print("━" * 70)
    print("STEP 1: Registering agents with spectral fingerprints")
    print("━" * 70)

    # Agent A: Data analyst — strong numerical, statistical capabilities
    bus.register("Analyst", {
        "data_processing": {"strength": 0.9, "domains": ["numerical", "statistical", "temporal"]},
        "pattern_recognition": {"strength": 0.85, "domains": ["numerical", "visual", "statistical"]},
        "forecasting": {"strength": 0.8, "domains": ["temporal", "numerical"]},
        "anomaly_detection": {"strength": 0.7, "domains": ["statistical", "numerical"]},
    }, {
        "data_processing": 1.0,
        "pattern_recognition": 0.8,
        "forecasting": 0.5,
        "anomaly_detection": 0.3,
    })

    # Agent B: Language model — strong text, reasoning capabilities
    bus.register("Linguist", {
        "text_generation": {"strength": 0.95, "domains": ["textual", "semantic", "logical"]},
        "summarization": {"strength": 0.9, "domains": ["textual", "semantic"]},
        "translation": {"strength": 0.85, "domains": ["textual", "semantic", "multilingual"]},
        "reasoning": {"strength": 0.8, "domains": ["logical", "semantic"]},
    }, {
        "text_generation": 1.0,
        "summarization": 0.7,
        "translation": 0.4,
        "reasoning": 0.9,
    })

    # Agent C: Code generator — strong programming, system capabilities
    bus.register("Coder", {
        "code_generation": {"strength": 0.95, "domains": ["programming", "logical", "system"]},
        "debugging": {"strength": 0.85, "domains": ["programming", "logical"]},
        "architecture": {"strength": 0.8, "domains": ["system", "programming"]},
        "testing": {"strength": 0.75, "domains": ["programming", "logical", "numerical"]},
    }, {
        "code_generation": 1.0,
        "debugging": 0.6,
        "architecture": 0.8,
        "testing": 0.5,
    })

    # Agent D: Creative — strong generative, artistic capabilities
    bus.register("Artist", {
        "image_generation": {"strength": 0.9, "domains": ["visual", "creative", "spatial"]},
        "style_transfer": {"strength": 0.85, "domains": ["visual", "creative"]},
        "composition": {"strength": 0.8, "domains": ["visual", "spatial", "creative"]},
        "color_theory": {"strength": 0.7, "domains": ["visual", "numerical"]},
    }, {
        "image_generation": 1.0,
        "style_transfer": 0.5,
        "composition": 0.7,
        "color_theory": 0.4,
    })

    # Print spectral fingerprints
    for name in ["Analyst", "Linguist", "Coder", "Artist"]:
        agent = bus.agents[name]
        print(f"\n  [{name}]")
        print(agent["spectral"].spectral_summary())

    # ── Step 2: Analyst broadcasts a numerical analysis task ──
    print("\n" + "━" * 70)
    print("STEP 2: Analyst broadcasts 'time series anomaly detection' task")
    print("━" * 70)

    responses = bus.broadcast("Analyst", {
        "components": {"0": 0.9, "1": 0.7, "2": 0.6, "3": 0.3},
    })

    print(f"\n  Broadcast results (sorted by spectral alignment):")
    for name, alignment in responses:
        print(f"    {name}: alignment = {alignment:.4f} {'✓ SELECTED' if alignment == responses[0][1] else ''}")

    if responses:
        best_agent = responses[0][0]
        best_alignment = responses[0][1]
        print(f"\n  → Best match: {best_agent} (alignment {best_alignment:.4f})")
        print(f"  → Reason: eigenvalue spectra of Analyst and {best_agent} are most aligned")
        print(f"  → This means their capability graphs have similar spectral structure")

    # ── Step 3: Direct send — Analyst → Linguist with a text task ──
    print("\n" + "━" * 70)
    print("STEP 3: Analyst → Linguist with a 'summarize findings' task")
    print("━" * 70)

    result = bus.send("Analyst", "Linguist", {
        "components": {"0": 0.3, "1": 0.9, "2": 0.7, "3": 0.5},
    })

    print(f"\n  Status: {result['status']}")
    print(f"  Alignment: {result['alignment']:.4f}")
    if result['status'] == 'ACCEPTED':
        print(f"  Conservation ratio (confidence): {result['conservation']:.4f}")
        fiedler = np.array(result['fiedler_routing'])
        print(f"  Fiedler routing (first 5): {np.array2string(fiedler[:5], precision=4)}")
        print(f"  Composed eigenvalues: {np.array2string(np.array(result['composed_eigenvalues']), precision=4, suppress_small=True)}")

    # ── Step 4: Direct send — Analyst → Artist with a numerical task ──
    print("\n" + "━" * 70)
    print("STEP 4: Analyst → Artist with a 'statistical analysis' task (mismatch)")
    print("━" * 70)

    result = bus.send("Analyst", "Artist", {
        "components": {"0": 0.9, "1": 0.8, "2": 0.3, "3": 0.1},
    })

    print(f"\n  Status: {result['status']}")
    print(f"  Alignment: {result['alignment']:.4f}")
    if 'reason' in result:
        print(f"  Reason: {result['reason']}")
        print(f"  → Analyst's eigenvalues are dominated by numerical/statistical modes")
        print(f"  → Artist's eigenvalues are dominated by visual/spatial modes")
        print(f"  → Low spectral alignment = wrong agent for the task")

    # ── Step 5: Broadcast a code task from Coder ──
    print("\n" + "━" * 70)
    print("STEP 5: Coder broadcasts 'generate unit tests' task")
    print("━" * 70)

    responses = bus.broadcast("Coder", {
        "components": {"0": 0.7, "1": 0.9, "2": 0.8, "3": 0.6},
    })

    print(f"\n  Broadcast results:")
    for name, alignment in responses:
        print(f"    {name}: alignment = {alignment:.4f}")

    # ── Step 6: Print full conversation log ──
    print("\n" + "━" * 70)
    print("FULL SPECTRAL CONVERSATION LOG")
    print("━" * 70)

    bus.print_log()

    # ── Summary ──
    print("\n" + "╔══════════════════════════════════════════════════════════════════╗")
    print("║                        SUMMARY                                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Messages exchanged:    spectral objects (Laplacians)           ║")
    print("║  Routing mechanism:     Fiedler vector (2nd eigenvector)        ║")
    print("║  Confidence metric:     Conservation ratio (Rayleigh quotient)  ║")
    print("║  Agent selection:       Eigenvalue cosine similarity            ║")
    print("║  Composition:           Block-diagonal Laplacian assembly       ║")
    print("║  Text exchanged:        NONE                                    ║")
    print("║  JSON exchanged:        NONE                                    ║")
    print("║  Mathematics exchanged: ALL OF IT                               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
