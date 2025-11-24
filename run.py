#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

import hydra

from omegaconf import DictConfig, OmegaConf
from hora.utils.misc import set_np_formatting, set_seed
from hora.algo.deploy.deploy_ros2 import HardwarePlayer
from hora.algo.deploy.deploy_ros2_two_hands import HardwarePlayerTwoHands


# OmegaConf & Hydra Config
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting()
    config.seed = set_seed(config.seed)

    # Detect mode based on config parameters
    checkpoint_right = config.get('checkpoint_right', None)
    checkpoint_left = config.get('checkpoint_left', None)
    checkpoint_single = config.get('checkpoint', None)

    # Two-hand mode: if checkpoint_right or checkpoint_left is specified
    if checkpoint_right is not None or checkpoint_left is not None:
        # Two-hand mode
        debug = config.get('debug', False)

        # Use checkpoint_single as fallback for checkpoint_right if not specified
        if checkpoint_right is None:
            checkpoint_right = checkpoint_single

        if checkpoint_right is None:
            raise ValueError("checkpoint_right (or checkpoint) must be specified for two-hand mode")

        agent = HardwarePlayerTwoHands(debug=debug)
        agent.restore(checkpoint_right, checkpoint_left)
        agent.deploy()
    else:
        # Single-hand mode
        if checkpoint_single is None:
            raise ValueError("checkpoint must be specified for single-hand mode")

        agent = HardwarePlayer()
        agent.restore(checkpoint_single)
        agent.deploy()


if __name__ == '__main__':
    main()
