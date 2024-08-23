import numpy as np
from env import sats, act, obs, scene, data, comm
from env.sim import dyn, fsw,world
from env.utils.orbital import random_orbit,walker_delta
from env.gym import GeneralSatelliteTasking, ConstellationTasking
from Basilisk.utilities import orbitalMotion
from ray.rllib.algorithms.appo import APPO, APPOConfig

class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.Time(),
        obs.SatProperties(
            dict(prop="omega_BP_P", module="dynamics", norm=0.03),
            dict(prop="c_hat_P"),
            dict(prop="r_BN_P", module="dynamics", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", module="dynamics", norm=7616.5, name="velocity"),
            # dict(prop="battery_charge_fraction")
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
                #    act.Charge(duration=60.0),
                #    act.Downlink(duration=60.0),
                #    act.Desat(duration=60.0)
                ]
    dyn_type = dyn.FullFeaturedDynModel
    fsw_type = fsw.SteeringImagerFSWModel

def unpack_config(env):
    """Create a wrapped version of an env class that unpacks env_config from Ray into kwargs"""

    class UnpackedEnv(env):
        def __init__(self, env_config):
            super().__init__(**env_config)

    UnpackedEnv.__name__ = f"{env.__name__}_Unpacked"

    return UnpackedEnv
def run():
    oes = walker_delta(
        n_spacecraft=3,  # Number of satellites
        n_planes=3,  # Number of orbital planes
        rel_phasing=0,
        altitude=500 * 1e3,
        inc=45,
        clustersize=3,  # Cluster all 3 satellites together
        clusterspacing=30,  # Space satellites by a true anomaly of 30 degrees
    )
    satellites = []
    sat_type = ImagingSatellite
    for i, oe in enumerate(oes):

        sat_args = sat_type.default_sat_args(
            oe=oe,
            imageAttErrorRequirement=0.01,  # Change a default parameter
            imageRateErrorRequirement=0.01,
            panelEfficiency=lambda: 0.2 + np.random.uniform(-0.01, 0.01),
        )

        satellite = sat_type(
            "EO" + str(i + 1), sat_args    
            )
        satellites.append(satellite)

    env_args = dict(
        satellites=satellites,
        world_type=world.GroundStationWorldModel,
        world_args=world.GroundStationWorldModel.default_world_args(),
        scenario=scene.UniformTargets(1000, priority_distribution=lambda: 1),#np.random.randint(4)+1),
        rewarder=data.UniqueImageReward(),
        communicator=comm.NoCommunication(),
        sim_rate=0.5,
        max_step_duration=600.0,
        time_limit=95 * 60,
        terminate_on_time_limit=True,
        log_level="warning",
    )

    training_args = dict(
        lr=0.003,
        gamma=0.999,
        train_batch_size=5000,

        num_sgd_iter=10,
        # model=dict(fcnet_hiddens=[512, 512],vf_share_layers=True),
        # model = dict(vf_share_layers=True),
        model = dict(use_lstm=True),

        lambda_=0.95,
        use_kl_loss=False,
        clip_param=0.1,
        grad_clip=0.1,
    )


    config = (
        APPOConfig()
        .training(**training_args)
        .environment(
            env=unpack_config(GeneralSatelliteTasking),
            env_config=env_args,
        )
        .framework("torch") # tf2, tf, torch
        .resources(
            num_gpus=1,
            )   
        .env_runners(
            num_env_runners=30,
            num_envs_per_env_runner=1,
            )
        
    )
    checkpoint_dir = "./checkpoints/lstm"

    trainer = config.build()
    # trainer.restore(checkpoint_dir)

    for _ in range(100000):
        trainer.train()
        trainer.save(checkpoint_dir)

if __name__ == "__main__":
    run()
