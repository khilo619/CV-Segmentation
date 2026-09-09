"""Global Sequence Trellis Optimization using Viterbi Dynamic Programming.

Assigned Engineer: Engineer 3 & Engineer 1
"""

from typing import List, Tuple
import numpy as np


class ViterbiSequenceOptimizer:
    """Enforces structural document transition constraints and CV length priors."""

    STATES = ["START", "PAGE_2", "PAGE_3", "PAGE_4_PLUS"]
    NUM_STATES = 4

    def __init__(self) -> None:
        # State index mapping:
        # 0: START
        # 1: PAGE_2
        # 2: PAGE_3
        # 3: PAGE_4_PLUS

        # Log transition matrix: log P(S_t | S_{t-1})
        # Note: Impossible transitions (e.g. START -> PAGE_3) are assigned -inf
        self.log_transitions = np.full((self.NUM_STATES, self.NUM_STATES), -np.inf)

        # From START:
        self.log_transitions[0, 0] = np.log(0.45)  # 1-page CV finished, next is START
        self.log_transitions[0, 1] = np.log(0.55)  # Continues to PAGE_2

        # From PAGE_2:
        self.log_transitions[1, 0] = np.log(0.70)  # 2-page CV finished, next is START
        self.log_transitions[1, 2] = np.log(0.30)  # Continues to PAGE_3

        # From PAGE_3:
        self.log_transitions[2, 0] = np.log(0.85)  # 3-page CV finished, next is START
        self.log_transitions[2, 3] = np.log(0.15)  # Continues to PAGE_4_PLUS

        # From PAGE_4_PLUS:
        self.log_transitions[3, 0] = np.log(0.92)  # Long CV finished, next is START
        self.log_transitions[3, 3] = np.log(0.08)  # Continues beyond 4 pages

    def decode(self, split_probabilities: list[float]) -> list[int]:
        """Runs Viterbi dynamic programming trellis over stream page emissions.

        Args:
            split_probabilities: List of N-1 probabilities where p_i is P(split between page i and i+1)

        Returns:
            List of 1-indexed page numbers where a split boundary occurs.
        """
        num_transitions = len(split_probabilities)
        if num_transitions == 0:
            return []

        num_pages = num_transitions + 1

        # Trellis tables: V[t, s] stores max log probability up to page t in state s
        trellis = np.full((num_pages, self.NUM_STATES), -np.inf)
        backpointers = np.zeros((num_pages, self.NUM_STATES), dtype=int)

        # Initial condition: Page 0 is unconditionally START (State 0)
        trellis[0, 0] = 0.0

        for t in range(1, num_pages):
            # p_split is emission evidence for transitioning into START
            p_split = float(np.clip(split_probabilities[t - 1], 1e-6, 1.0 - 1e-6))
            p_cont = 1.0 - p_split

            log_emit_start = np.log(p_split)
            log_emit_cont = np.log(p_cont)

            for curr_state in range(self.NUM_STATES):
                # Emission log-likelihood depends on whether current state is START or Continuation
                emit_log_prob = log_emit_start if curr_state == 0 else log_emit_cont

                # Find best previous state
                candidates = trellis[t - 1, :] + self.log_transitions[:, curr_state]
                best_prev_state = int(np.argmax(candidates))
                max_candidate_val = float(candidates[best_prev_state])

                if max_candidate_val > -np.inf:
                    trellis[t, curr_state] = max_candidate_val + emit_log_prob
                    backpointers[t, curr_state] = best_prev_state

        # Backtracking to reconstruct optimal path
        best_last_state = int(np.argmax(trellis[num_pages - 1, :]))
        state_path = [best_last_state]

        for t in range(num_pages - 1, 0, -1):
            prev_state = backpointers[t, state_path[-1]]
            state_path.append(prev_state)

        state_path.reverse()

        # Boundaries occur wherever state_path transitions into START (state 0) after page 0
        boundaries: list[int] = []
        for page_idx in range(1, num_pages):
            if state_path[page_idx] == 0:
                # 1-indexed boundary: page_idx is the last page of previous candidate
                boundaries.append(page_idx)

        return boundaries
