from env import PendulumEnv

if __name__ == "__main__":
    env = PendulumEnv(manual_control=True)
    while True:
        if env.step(0)[2]:
            env.close()
            break
