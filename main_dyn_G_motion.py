import argparse
import tensorflow as tf
from src.dynamic_generator import Dynamic_Generator

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CONFIG = tf.app.flags.FLAGS

# training mode
tf.flags.DEFINE_boolean('isTraining', 'True', 'training or Not')
tf.flags.DEFINE_string('training_mode', 'complete', 'training from [incomplete] data or [complete] data:')
tf.flags.DEFINE_string('dynamic_mode', 'nonlinear', 'dynamic model for hidden states [linear] or [nonlinear]')
tf.flags.DEFINE_string('frame_generator_mode', '64', 'generator model for image frames [64] or [150]: 64 is used for synthesis, 150 is for recovery')

# model hyper-parameters
tf.flags.DEFINE_integer('image_size', 64, 'Image size to rescale images')

# training hyper-parameters
tf.flags.DEFINE_integer('num_epochs', 7000, 'Number of epochs')
tf.flags.DEFINE_integer('num_frames', 30, 'number of frames used in training data')
tf.flags.DEFINE_integer('batch_size', 30, 'number of training examples (videos) in each batch')
tf.flags.DEFINE_integer('truncated_backprop_length', 30, 'truncated length for back propagation') #20


tf.flags.DEFINE_integer('state_size', 3, 'state dimension')
tf.flags.DEFINE_integer('content_size', 100, 'content dimension')
tf.flags.DEFINE_integer('motion_type_size', 0, 'motion type dimension')

# latent variables
tf.flags.DEFINE_integer('z_size', 100, 'channel of latent variables')  # 5

tf.flags.DEFINE_float('lr_gen', 0.002, 'learning rate')
tf.flags.DEFINE_float('beta1_gen', 0.5, 'momentum term in Adam')

# langevin hyper-parameters
tf.flags.DEFINE_float('refsig', 1, 'sigma')  # 0.003
tf.flags.DEFINE_float('step_size', 0.03, 'delta')
tf.flags.DEFINE_integer('sample_steps', 15, 'number of steps of Langevin sampling')

# misc
tf.flags.DEFINE_string('output_dir', './output_synthesis', 'output directory')
tf.flags.DEFINE_string('category', 'animal30_running', 'name of category')
tf.flags.DEFINE_string('data_path', './trainingVideo/action_dataset/', 'path of the training data')
tf.flags.DEFINE_integer('log_step', 10, 'number of steps to output synthesized image')
tf.flags.DEFINE_string('mask_file', 'missing_frame_type.mat', 'name of the mask file [missing_frame_type.mat | region_type.mat]')
tf.flags.DEFINE_string('mask_type', 'randomRegion', 'type of masks: [randomRegion | missingFrames | external]')
tf.flags.DEFINE_string('mask_path', './trainingVideo/recovery_dataset/mask/', 'mask path')

# testing
tf.flags.DEFINE_integer('num_sections_in_test', 2, 'total number of truncations in testing')
tf.flags.DEFINE_integer('num_batches_in_test', 2, 'number of batches generated in testing')
tf.flags.DEFINE_string('ckpt_name', 'model.ckpt-6990', 'name of the checkpoint')

def main():
    with tf.Session() as sess:

        model = Dynamic_Generator(sess, CONFIG)

        if CONFIG.isTraining:

            if CONFIG.training_mode == 'complete':
                model.train()
            elif CONFIG.training_mode == 'incomplete':
                model.train_from_incomplete()
            else:
                return NotImplementedError
        else:

            ckpt_path = os.path.join(CONFIG.output_dir, CONFIG.category, 'model', CONFIG.ckpt_name)
            appearance_path = os.path.join(CONFIG.output_dir, CONFIG.category, 'model', 'content_and_state.dat')
            state_initial_path = os.path.join(CONFIG.output_dir, CONFIG.category, 'model', 'state_initial.dat')
            motion_path = os.path.join(CONFIG.output_dir, CONFIG.category, 'model', 'motion_type_vectors.dat')
            model.test(ckpt_path,  appearance_path, motion_path,  state_initial_path, is_random_content=False,
                       is_random_motion_type=True, is_random_state_initial=True)


if __name__ == '__main__':
    main()
