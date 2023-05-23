from gymnasium.envs.registration import register

register(
    id="NMI-v0",
    entry_point="src.RL.gym_hotspot.envs:NMI_v0", 
)
register(
    id="NMI-v1",
    entry_point="src.RL.gym_hotspot.envs:NMI_v1", 
)
register(
    id="NMI-v2",
    entry_point="src.RL.gym_hotspot.envs:NMI_v2", 
)
register(
    id="NMI-v3",
    entry_point="src.RL.gym_hotspot.envs:NMI_v3", 
)
register(
    id="NMI_evaluate-v0",
    entry_point="src.RL.gym_hotspot.envs:NMI_evaluate_v0", 
)