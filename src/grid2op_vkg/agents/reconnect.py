"""The "simple reconnect rule" baseline from docs/Grid2Op基线方法调查.pdf §3.1.

Reconnect any powerline whose cooldown has expired, otherwise do nothing.
It is the same heuristic as ``baselines.utils.GymEnvWithReco`` but packaged as a
plain grid2op agent so it can be evaluated with the same loop as the others.
"""

import numpy as np
from grid2op.Agent import BaseAgent


class ReconnectAgent(BaseAgent):
    def act(self, observation, reward, done=False):
        to_reco = (observation.time_before_cooldown_line == 0) & (~observation.line_status)
        if not np.any(to_reco):
            return self.action_space({})
        # Game rules allow one line-status change per step; take the first.
        line_id = int(np.where(to_reco)[0][0])
        return self.action_space({"set_line_status": [(line_id, +1)]})
