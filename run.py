import hydra

from omegaconf import DictConfig, OmegaConf
from hora.utils.misc import set_np_formatting, set_seed
# noinspection PyUnresolvedReferences
from hora.algo.deploy.deploy_ros2 import HardwarePlayer


# ---- OmegaConf & Hydra Config
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
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
    agent = HardwarePlayer()
    agent.restore(config.checkpoint)
    agent.deploy()


if __name__ == '__main__':
    main()
