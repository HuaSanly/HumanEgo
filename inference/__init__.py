"""HumanEgo serving plugin; importing it never constructs models or hardware."""

# The original inference import used by simulation remains independent of LeRobot.
if __name__ == "lerobot_policy_humanego":
    from .configuration_humanego import HumanEgoConfig as HumanEgoConfig
