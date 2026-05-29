"""
AgentNativeComm — communication bus where agents speak in spectral messages.

No JSON. No text. Pure mathematics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from spectral_message import SpectralMessage


class AgentNativeComm:
    """
    Communication bus where agents speak in spectral messages.
    """

    def __init__(self):
        self.agents: Dict[str, dict] = {}
        self.message_log: List[dict] = []

    def register(self, name: str, capabilities: dict, initial_state: dict) -> None:
        """Register an agent with capabilities and initial state."""
        msg = SpectralMessage.from_agent_state(capabilities, None, initial_state)
        self.agents[name] = {
            "capabilities": capabilities,
            "spectral": msg,
            "state": initial_state,
        }
        self._log("REGISTER", name, None, {
            "laplacian_shape": msg.laplacian.shape,
            "eigenvalues": msg.eigenvalues.tolist(),
            "conservation_ratio": msg.conservation_ratio,
        })

    def send(self, from_agent: str, to_agent: str, task: dict) -> dict:
        """
        Send a task from one agent to another.
        The task IS a spectral projection.
        """
        sender = self.agents[from_agent]
        receiver = self.agents[to_agent]

        task_msg = SpectralMessage.from_agent_state(
            sender["capabilities"], task, sender["state"]
        )

        alignment = receiver["spectral"].can_handle(task_msg)

        if alignment < 0.3:
            result = {
                "status": "REJECTED",
                "alignment": alignment,
                "reason": f"Spectral alignment {alignment:.3f} too low",
            }
            self._log("REJECTED", from_agent, to_agent, result)
            return result

        composed = receiver["spectral"].compose(task_msg)
        result = {
            "status": "ACCEPTED",
            "alignment": alignment,
            "conservation": composed.conservation_ratio,
            "fiedler_routing": composed.fiedler_vector.tolist(),
            "composed_eigenvalues": composed.eigenvalues.tolist(),
        }
        self._log("ACCEPTED", from_agent, to_agent, result)
        return result

    def broadcast(self, from_agent: str, task: dict) -> List[Tuple[str, float]]:
        """
        Broadcast to ALL agents.
        Fiedler vector determines which agents respond.
        """
        sender = self.agents[from_agent]
        task_msg = SpectralMessage.from_agent_state(
            sender["capabilities"], task, sender["state"]
        )

        self._log("BROADCAST", from_agent, None, {
            "task_eigenvalues": task_msg.eigenvalues.tolist(),
            "conservation": task_msg.conservation_ratio,
        })

        responses: List[Tuple[str, float]] = []
        for name, agent in self.agents.items():
            if name == from_agent:
                continue
            alignment = agent["spectral"].can_handle(task_msg)
            self._log("ALIGNMENT_CHECK", from_agent, name, {
                "alignment": alignment,
                "accepted": alignment > 0.5,
            })
            if alignment > 0.5:
                responses.append((name, alignment))

        responses.sort(key=lambda x: -x[1])
        return responses

    def print_log(self) -> None:
        """Print the full conversation log."""
        print("=" * 80)
        print("SPECTRAL CONVERSATION LOG — No text exchanged, only Laplacians")
        print("=" * 80)
        for i, entry in enumerate(self.message_log, 1):
            print(f"\n--- Message #{i} ---")
            print(f"  Type: {entry['type']}")
            print(f"  From: {entry['from']}")
            if entry["to"]:
                print(f"  To:   {entry['to']}")
            for k, v in entry["details"].items():
                if isinstance(v, list):
                    arr = np.array(v)
                    print(f"  {k}: {np.array2string(arr, precision=4, suppress_small=True)}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    def _log(self, msg_type: str, from_agent: str, to_agent: Optional[str], details: dict) -> None:
        self.message_log.append({
            "type": msg_type,
            "from": from_agent,
            "to": to_agent,
            "details": details,
        })
