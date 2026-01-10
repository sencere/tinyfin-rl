from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.replay import ReplayBuffer
from tinyfin_rl.algos.dqn import DQNTrainer, QNetwork, hard_update
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.seed import set_seed
from tinyfin_rl.checkpoint import save_q_checkpoint, load_q_checkpoint


def main():
    set_seed(0)
    backend = TinyfinBackend(device="cpu")
    env = BanditEnv([0.1, 0.9])
    action_space = Discrete(2)
    q_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    target_net = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    hard_update(target_net, q_net)
    trainer = DQNTrainer(env=env, q_net=q_net, target_net=target_net, replay=ReplayBuffer(capacity=500))
    trainer.train(steps=200)
    save_q_checkpoint("dqn_checkpoint.npz", q_net, target_net, meta={"algo": "dqn"})

    q_net2 = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    target_net2 = QNetwork(backend=backend, obs_dim=1, action_space=action_space)
    load_q_checkpoint("dqn_checkpoint.npz", q_net2, target_net2)
    trainer2 = DQNTrainer(env=env, q_net=q_net2, target_net=target_net2, replay=ReplayBuffer(capacity=500))
    trainer2.train(steps=200)


if __name__ == "__main__":
    main()
