import numpy as np
from env import sats, act, obs, scene, data, comm
from env.sim import dyn, fsw, world
from env.utils.orbital import walker_delta
from env.gym import GeneralSatelliteTasking
from Basilisk.utilities import orbitalMotion

class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.Time(),
        obs.SatProperties(
            dict(prop="omega_BP_P", module="dynamics", norm=0.03),
            dict(prop="c_hat_P"),
            dict(prop="r_BN_P", module="dynamics", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", module="dynamics", norm=7616.5, name="velocity"),
            dict(prop="battery_charge_fraction")
        ),
        # obs.Eclipse(),
        obs.OpportunityProperties(
            dict(prop="priority"), 
            dict(prop="r_LP_P",norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="opportunity_open", norm=5700.0),
            dict(prop="opportunity_mid", norm=6000.0),
            dict(prop="opportunity_close", norm=6300.0),
            dict(prop="target_angle"),
            n_ahead_observe=10,
        )
    ]
    action_spec = [act.Image(n_ahead_image=10),
                   act.Charge(duration=60.0),
                #    act.Downlink(duration=60.0),
                #    act.Desat(duration=60.0)
                ]
    dyn_type = dyn.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel

class MAEOSEnv:
    def __init__(self, config):
        # 配置卫星轨道参数
        self.oes = walker_delta(
            n_spacecraft=config.n_spacecraft,
            n_planes=config.n_planes,
            rel_phasing=0,
            altitude=config.altitude,
            inc=config.inc,
            clustersize=config.clustersize,
            clusterspacing=config.clusterspacing,
        )
        self.satellites = self._create_satellites()
        self.env = self._create_environment(config)

    def _create_satellites(self):
        satellites = []
        sat_type = ImagingSatellite
        for i, oe in enumerate(self.oes):
            # 获取卫星默认配置参数，并可根据需要覆盖默认值
            sat_args = sat_type.default_sat_args(
                oe=oe,
                imageAttErrorRequirement=0.01,
                imageRateErrorRequirement=0.01,
                panelEfficiency=lambda: 0.2 + np.random.uniform(-0.01, 0.01),
            )
            # 输出第一颗卫星的配置参数以供检查
            if i == 0:
                print(sat_args)

            # 实例化卫星对象，并添加到卫星列表中
            satellite = sat_type("EO" + str(i + 1), sat_args)
            satellites.append(satellite)
        return satellites

    def _create_environment(self, config):
        # 创建任务环境
        env = GeneralSatelliteTasking(
            satellites=self.satellites,
            world_type=world.GroundStationWorldModel,
            world_args=world.GroundStationWorldModel.default_world_args(),
            scenario=scene.UniformTargets(1000),
            rewarder=data.UniqueImageReward(),
            communicator=comm.LOSCommunication(),
            sim_rate=config.sim_rate,
            max_step_duration=config.max_step_duration,
            time_limit=config.time_limit,
            log_level=config.log_level,
        )
        return env

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space