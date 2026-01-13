from tinyfin_rl.bindings import EnvPlugin, TrainLib, DqnConfig


def main():
    plugin = EnvPlugin("environments/lineworld/liblineworld.so")
    train = TrainLib()
    cfg = DqnConfig(obs_n=7, action_n=2, steps=1000, gamma=0.99, lr=0.05, epsilon=0.1)
    status, metrics = train.train_dqn(plugin, cfg)
    print(status, metrics.return_mean, metrics.loss)


if __name__ == "__main__":
    main()
