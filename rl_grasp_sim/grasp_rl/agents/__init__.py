"""Agents share one small interface:

    act(obs, explore)                         -> action index
    observe(obs, a, r, next_obs, terminated)  -> dict of training stats
    end_episode()                             -> decay schedules
    save(path) / load(path)
"""


def make_agent(algo, cfg, env, seed=0):
    if algo == "qlearning":
        from grasp_rl.agents.q_learning import QLearningAgent
        return QLearningAgent(cfg.qlearning, env.observation_space.n, env.action_space.n, seed)
    if algo == "dqn":
        from grasp_rl.agents.dqn import DQNAgent
        return DQNAgent(cfg.dqn, env.observation_space, env.action_space.n, seed)
    if algo == "scripted":
        from grasp_rl.agents.scripted import ScriptedAgent
        return ScriptedAgent(cfg, env)
    if algo == "random":
        from grasp_rl.agents.scripted import RandomAgent
        return RandomAgent(env.action_space.n, seed)
    raise ValueError(f"unknown algorithm {algo!r}")


def observation_mode(algo, cfg):
    return {"qlearning": cfg.qlearning.observation, "dqn": cfg.dqn.observation}.get(algo, "discrete")
