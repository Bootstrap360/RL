
import gym, helpers, argparse, os
from models import fully_connected
import tensorflow as tf
import environments
import matplotlib.pyplot as plt
import datetime

def main(args):
    BATCH_SIZE = args.batch_size
    MAX_EPSILON = args.max_epsilon
    MIN_EPSILON = args.min_epsilon
    decay = args.decay
    gamma = args.gamma

    env_name = args.env_name
    if env_name in ['MountainCar-v0']:
        env = gym.make(env_name)
        num_states = env.env.observation_space.shape[0]
        num_actions = env.env.action_space.n
    else:
        env = environments.make(env_name)
        num_states = env.get_num_states()
        num_actions = env.get_num_actions()


    model = fully_connected.Model(num_states, num_actions, BATCH_SIZE, layer_sizes=[10,10])
    mem = helpers.Memory(1000)

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    saver = tf.train.Saver()
    model_save_dir = os.path.join('.', 'saved_models', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(model_save_dir, exist_ok=True)
    with tf.Session(config=config) as sess:
        sess.run(model.var_init)
        gr = helpers.GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        decay, gamma)
        num_episodes = 300
        cnt = 0
        while cnt < num_episodes:
            if cnt % 50 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                gr._render = True
                gr.run()
                save_path = saver.save(sess, os.path.join(model_save_dir,"model_{:05d}.ckpt".format(cnt)))
                print("Model saved in path: %s" % save_path)
            else:
                gr._render = True
                gr.run()
            cnt += 1
        # plt.plot(gr.reward_store)
        # plt.show()
        # plt.close("all")
        # plt.plot(gr.max_x_store)
        # plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 3, type=int)
    parser.add_argument('--max_epsilon', default = 1, type=float)
    parser.add_argument('--min_epsilon', default = 0.01, type=float)
    parser.add_argument('--decay', default =  0.001, type = float)
    parser.add_argument('--gamma', default =  0.99, type = float)
    parser.add_argument('--env_name', default =  'MountainCar-v0', type = str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)