from src.logger import Logger
from src.algorithms.utils import validate_policy, get_model_weights_from_file

def validate_queries(conf, weight_file, num_samples = 1000):
    policy_model = conf.policy_model
    policy_model.set_weights(get_model_weights_from_file(weight_file))

    env = conf.val_env
    logger = Logger(conf.experiment_name, create_tensorboard=False, validation_only=True)

    for i in range(num_samples):
        validate_policy(policy_model, env, i, logger, conf, execution = True)
