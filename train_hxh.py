import yaml
from types import SimpleNamespace
from algs.ippo_runner import IPPORunner
from algs.mappo_runner import MAPPORunner
from env.maeos_env import MAEOSEnv


if __name__ == '__main__':
    # 读取配置文件
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)

    # 初始化环境和训练类
    env = MAEOSEnv(config)
    if config.shared:
        #mappo
        runner = MAPPORunner(env, config)
    else:
        #ippo
        runner = IPPORunner(env, config)

    # ma2ppo

    runner.run()