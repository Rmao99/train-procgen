import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

LOG_DIR = 'log'

def main():
    num_envs = 64
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    # timesteps_per_proc = 50_000_000
    use_vf_clipping = True

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--total_timesteps', type=int, default=0)

    args = parser.parse_args()

    test_worker_interval = args.test_worker_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else args.num_levels

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=LOG_DIR, 
                     format_strs=format_strs,
                     log_suffix="_total_timesteps_{}_num_levels_{}".format(args.total_timesteps,
                                                                           num_levels))

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating evaluation environment")
    eval_venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=100, start_level=2000, distribution_mode=args.distribution_mode)
    eval_venv = VecExtractDictObs(eval_venv, "rgb")

    eval_venv = VecMonitor(
        venv=eval_venv, filename=None, keep_buf=100,
    )

    eval_venv = VecNormalize(venv=eval_venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    model = ppo2.learn(
                    env=venv,
                    eval_env=eval_venv,
                    network=conv_fn,
                    total_timesteps=args.total_timesteps,
                    save_interval=0,
                    nsteps=nsteps,
                    nminibatches=nminibatches,
                    lam=lam,
                    gamma=gamma,
                    noptepochs=ppo_epochs,
                    log_interval=1,
                    ent_coef=ent_coef,
                    mpi_rank_weight=mpi_rank_weight,
                    clip_vf=use_vf_clipping,
                    comm=comm,
                    lr=learning_rate,
                    cliprange=clip_range,
                    update_fn=None,
                    init_fn=None,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                )

    # Save the model
    model.save("model/model_total_timesteps_{}_num_levels_{}".format(args.total_timesteps,
                                                                     num_levels))

if __name__ == '__main__':
    main()
