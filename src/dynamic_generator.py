from __future__ import division

import tensorflow as tf
import numpy as np
import shutil
import os
import time
from ops import *
from util import *
from progressbar import ETA, Bar, Percentage, ProgressBar
import scipy.io

class Dynamic_Generator(object):
    def __init__(self, sess, config):

        self.sess = sess

        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.num_frames = config.num_frames
        # self.num_chain = config.num_chain

        self.num_epochs = config.num_epochs
        self.truncated_backprop_length = config.truncated_backprop_length

        self.num_truncation = self.num_frames // self.truncated_backprop_length

        # parameters for optimizer
        self.lr_gen = config.lr_gen
        self.beta1_gen = config.beta1_gen

        # parameters for inference
        self.refsig = config.refsig
        self.step_size = config.step_size
        self.sample_steps = config.sample_steps

        # parameters for MLP
        self.z_size = config.z_size
        self.state_size = config.state_size
        self.content_size = config.content_size
        self.motion_type_size = config.motion_type_size

        self.mask_path = os.path.join(config.mask_path, config.category)
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.training_mode = config.training_mode
        self.dynamic_mode = config.dynamic_mode  # linear or nonlinear dynamics for states
        self.frame_generator_mode = config.frame_generator_mode # network type for image frame generator

        # testing parameters
        self.num_sections_in_test = config.num_sections_in_test
        self.num_batches_in_test = config.num_batches_in_test

        # recovery parameters
        self.mask_file = config.mask_file
        self.mask_type = config.mask_type


        self.log_dir = os.path.join(self.output_dir, 'log')
        self.train_dir = os.path.join(self.output_dir, 'observed_sequence')
        self.sample_dir = os.path.join(self.output_dir, 'synthesis_sequence')
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.result_dir = os.path.join(self.output_dir, 'final_result')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        if tf.gfile.Exists(self.sample_dir):
            tf.gfile.DeleteRecursively(self.sample_dir)
        tf.gfile.MakeDirs(self.sample_dir)

    # build all computational graph
    def build_model(self):

        self.truncated_batch_z_placeholder = tf.placeholder(shape=[None, self.truncated_backprop_length, self.z_size],
                                                            dtype=tf.float32, name='z')
        self.batch_state_initial_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32,
                                                              name='state_initial')
        self.truncated_batch_obs_placeholder = tf.placeholder(
            shape=[None, self.truncated_backprop_length, self.image_size, self.image_size, 3], dtype=tf.float32,
            name='obs')

        if self.training_mode == "incomplete":
            # mask placeholder will be used in the setting of learning from incomplete data only
            self.truncated_batch_visible_placeholder = tf.placeholder(
                shape=[None, self.truncated_backprop_length, self.image_size, self.image_size, 3], dtype=tf.bool,
                name='mask')

        self.batch_content_placeholder = tf.placeholder(shape=[None, self.content_size], dtype=tf.float32,
                                                        name='content')
        self.batch_motion_type_placeholder = tf.placeholder(shape=[None, self.motion_type_size], dtype=tf.float32,
                                                            name='motion_type')

        self.images_syn, self.next_state, self.content_and_state = self.dyn_generator(
            self.truncated_batch_z_placeholder,
            self.batch_state_initial_placeholder,
            self.batch_content_placeholder,
            self.batch_motion_type_placeholder,
            dynamic_type=self.dynamic_mode,
            frame_generator_type=self.frame_generator_mode)

        if self.training_mode == "incomplete":

            diff_visible = tf.boolean_mask(self.truncated_batch_obs_placeholder - self.images_syn,
                                          self.truncated_batch_visible_placeholder)
            self.dyn_gen_loss = tf.reduce_mean(
                1.0 / (2 * self.refsig * self.refsig) * tf.square(diff_visible))

        elif self.training_mode == "complete":

            self.dyn_gen_loss = tf.reduce_mean(
                1.0 / (2 * self.refsig * self.refsig) * tf.square(
                    self.truncated_batch_obs_placeholder - self.images_syn),
                axis=[0, 1])
        else:
            return NotImplementedError

        self.dyn_gen_loss_mean, self.dyn_gen_loss_update = tf.contrib.metrics.streaming_mean(self.dyn_gen_loss)

        dyn_gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('dyn_gen')]
        dyn_gen_optimizer = tf.train.AdamOptimizer(self.lr_gen, beta1=self.beta1_gen)
        # dyn_gen_optimizer = tf.train.AdagradOptimizer(0.3)

        dyn_gen_grads_vars = dyn_gen_optimizer.compute_gradients(self.dyn_gen_loss, var_list=dyn_gen_vars)

        self.apply_dyn_gen_grads = dyn_gen_optimizer.apply_gradients(dyn_gen_grads_vars)

        self.langevin_dyn_generator = self.langevin_dynamics_dynamic_generator(self.truncated_batch_z_placeholder,
                                                                               self.batch_state_initial_placeholder,
                                                                               self.batch_content_placeholder,
                                                                               self.batch_motion_type_placeholder)

        self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean_squared_error(self.images_syn,
                                                                                                     self.truncated_batch_obs_placeholder)

    # main model
    def dyn_generator(self, z, state_initial, content, motion_type, dynamic_type="nonlinear", frame_generator_type="64", reuse=False):

        # dynamic model for hidden states
        all_states_stacked, next_state = self.state_dynamic_model(z, state_initial, content, motion_type,
                                                                  dynamic_type=dynamic_type, reuse=reuse)

        all_states = tf.reshape(all_states_stacked, shape=(
            tf.shape(all_states_stacked)[0] * tf.shape(all_states_stacked)[1], self.state_size + self.content_size))

        # generator for video frames
        images_syn = self.frame_generator(all_states, reuse=reuse, type=frame_generator_type)

        images_syn = tf.reshape(images_syn, shape=(
            tf.shape(all_states_stacked)[0], tf.shape(all_states_stacked)[1], self.image_size, self.image_size, 3))

        # return synthesized video, next state, and all states (with content vectors if added)
        return images_syn, next_state, all_states_stacked

    def state_dynamic_model(self, z, state_initial, content, motion_type, dynamic_type="nonlinear", reuse=False):

        # z is a batch of latent variables, and state_initial is a batch of initial states
        if dynamic_type == "linear":

            with tf.variable_scope('dyn_gen', reuse=reuse):
                sigma = 0.001
                w_alpha = tf.Variable(
                    sigma * np.random.randn(self.state_size + self.z_size + self.motion_type_size, self.state_size),
                    dtype=tf.float32, name='w_alpha')
                b_alpha = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32, name='b_alpha')

            # unstack the latent variables in each batch over time dimension
            z_series = tf.unstack(z, axis=1)

            current_state = state_initial
            content_and_state_series = []
            for current_z in z_series:
                current_z = tf.reshape(current_z, [tf.shape(z)[0], self.z_size])
                z_motion_type_and_state_concatenated = tf.concat([motion_type, current_z, current_state], axis=1)

                next_state = tf.tanh(tf.matmul(z_motion_type_and_state_concatenated, w_alpha) + b_alpha + current_state)

                # add a fixed sequence-specific content vector to each frame
                content_and_state_concatenated = tf.concat([content, next_state], axis=1)
                #  content_and_state_concatenated = tf.tanh(content_and_state_concatenated)

                content_and_state_series.append(content_and_state_concatenated)
                current_state = next_state

            all_states_stacked = tf.stack(content_and_state_series, axis=1)
            return all_states_stacked, next_state

        elif dynamic_type == "nonlinear":

            # use three layers of MLP
            with tf.variable_scope('dyn_gen', reuse=reuse):

                output_dim1 = 20
                sigma = 0.001
                w_alpha = tf.Variable(
                    sigma * np.random.randn(self.state_size + self.z_size + self.motion_type_size, output_dim1),
                    dtype=tf.float32, name='w_alpha')
                b_alpha = tf.Variable(np.zeros((1, output_dim1)), dtype=tf.float32, name='b_alpha')

                output_dim2 = 20
                w2_alpha = tf.Variable(sigma * np.random.randn(output_dim1, output_dim2),
                                       dtype=tf.float32, name='w2_alpha')
                b2_alpha = tf.Variable(np.zeros((1, output_dim2)), dtype=tf.float32, name='b2_alpha')

                w3_alpha = tf.Variable(sigma * np.random.randn(output_dim2, self.state_size),
                                       dtype=tf.float32, name='w3_alpha')
                b3_alpha = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32, name='b3_alpha')

            # unstack the latent variables in each batch over time dimension
            z_series = tf.unstack(z, axis=1)

            current_state = state_initial
            content_and_state_series = []
            for current_z in z_series:
                current_z = tf.reshape(current_z, [tf.shape(z)[0], self.z_size])
                z_motion_type_and_state_concatenated = tf.concat([motion_type, current_z, current_state], axis=1)

                output1 = tf.matmul(z_motion_type_and_state_concatenated, w_alpha) + b_alpha
                output1 = tf.nn.relu(output1)
                output2 = tf.matmul(output1, w2_alpha) + b2_alpha
                output2 = tf.nn.relu(output2)
                next_state = tf.matmul(output2, w3_alpha) + b3_alpha + current_state
                next_state = tf.tanh(next_state)

                # add a fixed sequence-specific content vector to each frame
                content_and_state_concatenated = tf.concat([content, next_state], axis=1)

                content_and_state_series.append(content_and_state_concatenated)
                current_state = next_state

            all_states_stacked = tf.stack(content_and_state_series, axis=1)
            return all_states_stacked, next_state

        else:
            return NotImplementedError

    # generate a group of image frames. The input size is [num_of_images, state_size + content_size]
    def frame_generator(self, inputs, reuse=False, is_training=True, type="64"):

        # generating a group of image frames
        if type == "64":

            with tf.variable_scope('dyn_gen', reuse=reuse):

                inputs = tf.reshape(inputs, [-1, 1, 1, self.state_size + self.content_size])
                convt1 = convt2d(inputs, (None, self.image_size // 32, self.image_size // 32, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = tf.nn.relu(convt1)

                convt2 = convt2d(convt1, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = tf.nn.relu(convt2)

                convt3 = convt2d(convt2, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = tf.nn.relu(convt3)

                convt4 = convt2d(convt3, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt4")
                convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
                convt4 = tf.nn.relu(convt4)

                convt5 = convt2d(convt4, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt5")
                convt5 = tf.contrib.layers.batch_norm(convt5, is_training=is_training)
                convt5 = tf.nn.relu(convt5)

                convt6 = convt2d(convt5, (None, self.image_size, self.image_size, 3), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt6")
                convt6 = tf.nn.tanh(convt6)

                return convt6

        elif type == "128":

            with tf.variable_scope('dyn_gen', reuse=reuse):

                inputs = tf.reshape(inputs, [-1, 1, 1, self.state_size + self.content_size])

                convt0 = convt2d(inputs, (None, self.image_size // 64, self.image_size // 64, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt0")
                convt0 = tf.contrib.layers.batch_norm(convt0, is_training=is_training)
                convt0 = tf.nn.relu(convt0)

                convt1 = convt2d(convt0, (None, self.image_size // 32, self.image_size // 32, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = tf.nn.relu(convt1)

                convt2 = convt2d(convt1, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = tf.nn.relu(convt2)

                convt3 = convt2d(convt2, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = tf.nn.relu(convt3)

                convt4 = convt2d(convt3, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt4")
                convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
                convt4 = tf.nn.relu(convt4)

                convt5 = convt2d(convt4, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt5")
                convt5 = tf.contrib.layers.batch_norm(convt5, is_training=is_training)
                convt5 = tf.nn.relu(convt5)

                convt6 = convt2d(convt5, (None, self.image_size, self.image_size, 3), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt6")
                convt6 = tf.nn.tanh(convt6)

                return convt6

        elif type == "150":

            with tf.variable_scope('dyn_gen', reuse=reuse):

                inputs = tf.reshape(inputs, [-1, 1, 1, self.state_size + self.content_size])
                convt1 = convt2d(inputs, (None, 2, 2, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = tf.nn.relu(convt1)

                convt2 = convt2d(convt1, (None, 4, 4, 512), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = tf.nn.relu(convt2)

                convt3 = convt2d(convt2, (None, 8, 8, 256), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = tf.nn.relu(convt3)

                convt4 = convt2d(convt3, (None, 16, 16, 128), kernal=(4, 4),
                                 strides=(2, 2), padding="SAME", name="convt4")
                convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
                convt4 = tf.nn.relu(convt4)

                convt5 = convt2d(convt4, (None, 48, 48, 64), kernal=(4, 4),
                                 strides=(3, 3), padding="SAME", name="convt5")
                convt5 = tf.contrib.layers.batch_norm(convt5, is_training=is_training)
                convt5 = tf.nn.relu(convt5)

                convt6 = convt2d(convt5, (None, 144, 144, 64), kernal=(4, 4),
                                 strides=(3, 3), padding="SAME", name="convt6")
                convt6 = tf.contrib.layers.batch_norm(convt6, is_training=is_training)
                convt6 = tf.nn.relu(convt6)
                #
                convt7 = convt2d(convt6, (None, 150, 150, 3), kernal=(7, 7),
                                 strides=(1, 1), padding="VALID", name="convt7")
                convt7 = tf.nn.tanh(convt7)

                return convt7

        else:
            return NotImplementedError

    def langevin_dynamics_frame_generator(self, z, truncated_batch_obs_placeholder):

        def cond(i, z):
            return tf.less(i, self.sample_steps)

        def body(i, z):
            gen_res = self.frame_generator(z, reuse=True)[0]

            gen_loss = tf.reduce_mean(
                1.0 / (2 * self.refsig * self.refsig) * tf.square(truncated_batch_obs_placeholder - gen_res),
                axis=[0, 1])
            grad_z = tf.gradients(gen_loss, z, name='grad_gen_z')[0]
            noise_z = tf.random_normal(shape=tf.shape(z), name='noise_z')
            z = z - 0.5 * self.step_size * self.step_size * (z + grad_z) + self.step_size * noise_z

            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_frame_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z])
            return z


    def langevin_dynamics_dynamic_generator(self, z, state_initial, content, motion_type):

        def cond(i, z, state_initial, content, motion_type):
            return tf.less(i, self.sample_steps)

        def body(i, z, state_initial, content, motion_type):
            gen_res = self.dyn_generator(z, state_initial, content, motion_type, dynamic_type=self.dynamic_mode,
                                                         frame_generator_type=self.frame_generator_mode, reuse=True)[0]

            if self.training_mode == "incomplete":

                diff_visible = tf.boolean_mask(self.truncated_batch_obs_placeholder - gen_res,
                                              self.truncated_batch_visible_placeholder)

                dyn_gen_loss = tf.reduce_mean(
                    1.0 / (2 * self.refsig * self.refsig) * tf.square(diff_visible))

            elif self.training_mode == "complete":
                dyn_gen_loss = tf.reduce_mean(
                    1.0 / (2 * self.refsig * self.refsig) * tf.square(self.truncated_batch_obs_placeholder - gen_res),
                    axis=[0, 1])
            else:
                return NotImplementedError


            grad_z = tf.gradients(dyn_gen_loss, z, name='grad_dyn_gen_z')[0]
            noise_z = tf.random_normal(shape=tf.shape(z), name='noise_z')
            z = z - 0.5 * self.step_size * self.step_size * (z + grad_z) + self.step_size * noise_z

            grad_content = tf.gradients(dyn_gen_loss, content, name='grad_dyn_gen_content')[0]
            noise_content = tf.random_normal(shape=tf.shape(content), name='noise_content')
            content = content - 0.5 * self.step_size * self.step_size * (
                content + grad_content) + self.step_size * noise_content

            grad_motion_type = tf.gradients(dyn_gen_loss, motion_type, name='grad_dyn_gen_motion_type')[0]
            noise_motion_type = tf.random_normal(shape=tf.shape(motion_type), name='noise_motion_type')
            motion_type = motion_type - 0.5 * self.step_size * self.step_size * (
                motion_type + grad_motion_type) + self.step_size * noise_motion_type

            grad_state_initial = tf.gradients(dyn_gen_loss, state_initial, name='grad_dyn_gen_state_initial')[0]
            noise_state_initial = tf.random_normal(shape=tf.shape(state_initial), name='noise_state_initial')
            state_initial = state_initial - 0.5 * self.step_size * self.step_size * (
                state_initial + grad_state_initial) + self.step_size * noise_state_initial

            return tf.add(i, 1), z, state_initial, content, motion_type


        with tf.name_scope("langevin_dynamics_dynamic_generator"):
            i = tf.constant(0)
            i, z, state_initial, content, motion_type = tf.while_loop(cond, body,
                                                                      [i, z, state_initial, content, motion_type])
            return z, content, motion_type, state_initial

    def train_from_incomplete(self):

        # build dynamic generator model
        self.build_model()

        # Prepare training data
        loadVideoToFrames(self.data_path, self.train_dir)
        train_data = getTrainingData(self.train_dir, num_frames=self.num_frames, image_size=self.image_size,
                                     scale_method='tanh')
        num_batches = int(math.ceil(train_data.shape[0] / self.batch_size))

        # create mask: 1 or true value indicates masked position
        mask = np.zeros(shape=train_data.shape, dtype=bool)

        if self.mask_type == 'external':

            assert self.mask_file, "mask file is empty."

            mat = scipy.io.loadmat(os.path.join(self.mask_path, self.mask_file))
            mask1 = mat['masks']
            mask1 = np.transpose(mask1, (2, 0, 1, 3))
            mask1 = np.expand_dims(mask1, axis=0)
            mask1 = mask1[:, 0:self.num_frames, :, :, :]
            mask = mask1.astype(bool)

        elif self.mask_type == 'randomRegion':

            mask = pepper_salt_masks(train_data, mask_ratio=0.6, mask_sz=16, mask_duration=70)
            mask = mask.astype(np.bool)

        elif self.mask_type == 'missingFrames':

            ## mask the middle 1/3 frames in the videos
            mask[0, int(math.ceil(mask.shape[1] / 3)): int(math.ceil(mask.shape[1] / 3 * 2)), :, :, :] = 1

        else:
            return NotImplementedError


        visible_mask = np.invert(mask)
        GT_data=train_data.copy()
        train_data[mask] = 0

        # train_data = np.reshape(train_data[1,:,:,:,:], (1,150,64,64,3))
        print(train_data.shape)
        saveSampleSequence(train_data, self.sample_dir, 'observed', col_num=10, scale_method='tanh')

        # initialize training
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)

        # train_data = train_data.reshape((self.batch_size, -1, self.image_size, self.image_size, 3))
        z = np.random.normal(0, 1, size=(train_data.shape[0], train_data.shape[1], self.z_size)) * self.refsig
        state_initial = np.random.normal(0, 1, size=(train_data.shape[0], self.state_size))  # fixed

        content_vectors = np.random.normal(0, 1, size=(train_data.shape[0], self.content_size))

        motion_type_vectors = np.random.normal(0, 1, size=(train_data.shape[0], self.motion_type_size))

        reconstructed_data = np.zeros(train_data.shape)

        for epoch in xrange(self.num_epochs):

            for iBatch in xrange(num_batches):

                indices_batch = slice(iBatch * self.batch_size,
                                      min(train_data.shape[0], (iBatch + 1) * self.batch_size))
                batch_state_initial = state_initial[indices_batch, :]
                batch_content_vectors = content_vectors[indices_batch, :]
                batch_motion_type_vectors = motion_type_vectors[indices_batch, :]

                for i in xrange(self.num_truncation):
                    start_time = time.time()

                    indices_truncation = slice(i * self.truncated_backprop_length,
                                               (i + 1) * self.truncated_backprop_length)
                    batch_obs = train_data[indices_batch, indices_truncation, :, :, :]
                    batch_z = z[indices_batch, indices_truncation, :]

                    batch_visible = visible_mask[indices_batch, indices_truncation, :, :, :]
                    # inference by Langevin

                    if i == 0:
                        batch_z, batch_content_vectors, batch_motion_type_vectors, batch_state_initial = self.sess.run(
                            self.langevin_dyn_generator,
                            feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                       self.batch_state_initial_placeholder: batch_state_initial,
                                       self.truncated_batch_obs_placeholder: batch_obs,
                                       self.batch_content_placeholder: batch_content_vectors,
                                       self.batch_motion_type_placeholder: batch_motion_type_vectors,
                                       self.truncated_batch_visible_placeholder: batch_visible})

                        state_initial[indices_batch, :] = batch_state_initial

                    else:

                        batch_z, batch_content_vectors, batch_motion_type_vectors, _ = self.sess.run(
                            self.langevin_dyn_generator,
                            feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                       self.batch_state_initial_placeholder: batch_state_initial,
                                       self.truncated_batch_obs_placeholder: batch_obs,
                                       self.batch_content_placeholder: batch_content_vectors,
                                       self.batch_motion_type_placeholder: batch_motion_type_vectors,
                                       self.truncated_batch_visible_placeholder: batch_visible})

                    # learn the model by MLE
                    batch_last_state, batch_content_and_state = self.sess.run(
                        [self.next_state, self.content_and_state, self.dyn_gen_loss, self.dyn_gen_loss_update,
                         self.apply_dyn_gen_grads],
                        feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                   self.batch_state_initial_placeholder: batch_state_initial,
                                   self.truncated_batch_obs_placeholder: batch_obs,
                                   self.batch_content_placeholder: batch_content_vectors,
                                   self.batch_motion_type_placeholder: batch_motion_type_vectors,
                                   self.truncated_batch_visible_placeholder: batch_visible})[0:2]

                    print(batch_last_state[0, 0])
                    recon = self.sess.run(self.images_syn, feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                                                      self.batch_state_initial_placeholder: batch_state_initial,
                                                                      self.batch_content_placeholder: batch_content_vectors,
                                                                      self.batch_motion_type_placeholder: batch_motion_type_vectors})

                    z[indices_batch, indices_truncation, :] = batch_z

                    reconstructed_data[indices_batch, indices_truncation, :, :, :] = recon

                    self.sess.run(self.recon_err_update, feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                                                    self.batch_state_initial_placeholder: batch_state_initial,
                                                                    self.truncated_batch_obs_placeholder: batch_obs,
                                                                    self.batch_content_placeholder: batch_content_vectors,
                                                                    self.batch_motion_type_placeholder: batch_motion_type_vectors})
                    batch_state_initial = batch_last_state

                    [gen_loss_avg, mse_avg] = self.sess.run([self.dyn_gen_loss_mean, self.recon_err_mean])

                    end_time = time.time()
                    print(
                        'Epoch #%d of #%d, batch #%d of #%d, truncation #%d of #%d, dynamic generator loss: %4.4f, Avg MSE: %4.4f, time: %.2fs' % (
                            epoch + 1, self.num_epochs, iBatch + 1, num_batches, i + 1, self.num_truncation,
                            gen_loss_avg,
                            mse_avg, end_time - start_time))

                content_vectors[indices_batch, :] = batch_content_vectors
                motion_type_vectors[indices_batch, :] = batch_motion_type_vectors

            reconstructed_data[visible_mask] = train_data[visible_mask]  # only recover the invisible part

            print('Recovery Error: #%f in scale of [0,255]/pixel' % (np.mean(np.abs((reconstructed_data - GT_data)[mask])) / 2 * 255))

            if epoch % self.log_step == 0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                # saveSampleSequence(sample_videos + img_mean, self.sample_dir, epoch, col_num=10)
                saveSampleSequence(reconstructed_data, self.sample_dir, "%04d" % epoch, col_num=10, scale_method='tanh')
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)
            if epoch % 20 == 0:
                saveSampleVideo(reconstructed_data, self.result_dir, original=(GT_data), global_step=epoch,
                                scale_method='tanh')

    def train(self):

        # build dynamic generator model
        self.build_model()

        # Prepare training data
        loadVideoToFrames(self.data_path, self.train_dir)
        train_data = getTrainingData(self.train_dir, num_frames=self.num_frames, image_size=self.image_size,
                                     scale_method='tanh')

        # Save the actual resized training image frames
        train_dir2 = os.path.join(self.train_dir, 'resized_observed_sequence')
        saveSampleImageFrames(train_data, train_dir2, scale_method='tanh')

        num_batches = int(math.ceil(train_data.shape[0] / self.batch_size))

        # train_data = np.reshape(train_data[1,:,:,:,:], (1,150,64,64,3))
        print(train_data.shape)

        # initialize training
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)

        # train_data = train_data.reshape((self.batch_size, -1, self.image_size, self.image_size, 3))
        z = np.random.normal(0, 1, size=(train_data.shape[0], train_data.shape[1], self.z_size)) * self.refsig
        state_initial = np.random.normal(0, 1, size=(train_data.shape[0], self.state_size))  # fixed
        content_vectors = np.random.normal(0, 1, size=(train_data.shape[0], self.content_size))
        motion_type_vectors = np.random.normal(0, 1, size=(train_data.shape[0], self.motion_type_size))

        reconstructed_data = np.zeros(train_data.shape)
        content_and_state_matrix = np.zeros(
            [train_data.shape[0], train_data.shape[1], self.state_size + self.content_size])

        for epoch in xrange(self.num_epochs):

            for iBatch in xrange(num_batches):

                indices_batch = slice(iBatch * self.batch_size,
                                      min(train_data.shape[0], (iBatch + 1) * self.batch_size))
                batch_state_initial = state_initial[indices_batch, :]
                batch_content_vectors = content_vectors[indices_batch, :]
                batch_motion_type_vectors = motion_type_vectors[indices_batch, :]

                for i in xrange(self.num_truncation):
                    start_time = time.time()

                    indices_truncation = slice(i * self.truncated_backprop_length,
                                               (i + 1) * self.truncated_backprop_length)
                    batch_obs = train_data[indices_batch, indices_truncation, :, :, :]

                    batch_z = z[indices_batch, indices_truncation, :]

                    if i == 0:

                        # inference by Langevin
                        batch_z, batch_content_vectors, batch_motion_type_vectors, batch_state_initial = self.sess.run(
                            self.langevin_dyn_generator,
                            feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                       self.batch_state_initial_placeholder: batch_state_initial,
                                       self.truncated_batch_obs_placeholder: batch_obs,
                                       self.batch_content_placeholder: batch_content_vectors,
                                       self.batch_motion_type_placeholder: batch_motion_type_vectors})

                        state_initial[indices_batch, :] = batch_state_initial
                    else:

                        batch_z, batch_content_vectors, batch_motion_type_vectors, _ = self.sess.run(
                            self.langevin_dyn_generator,
                            feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                       self.batch_state_initial_placeholder: batch_state_initial,
                                       self.truncated_batch_obs_placeholder: batch_obs,
                                       self.batch_content_placeholder: batch_content_vectors,
                                       self.batch_motion_type_placeholder: batch_motion_type_vectors})


                    # learn the model by MLE
                    batch_last_state, batch_content_and_state = self.sess.run(
                        [self.next_state, self.content_and_state, self.dyn_gen_loss, self.dyn_gen_loss_update,
                         self.apply_dyn_gen_grads],
                        feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                   self.batch_state_initial_placeholder: batch_state_initial,
                                   self.truncated_batch_obs_placeholder: batch_obs,
                                   self.batch_content_placeholder: batch_content_vectors,
                                   self.batch_motion_type_placeholder: batch_motion_type_vectors})[0:2]

                    print(batch_last_state[0, 0])
                    recon = self.sess.run(self.images_syn, feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                                                      self.batch_state_initial_placeholder: batch_state_initial,
                                                                      self.batch_content_placeholder: batch_content_vectors,
                                                                      self.batch_motion_type_placeholder: batch_motion_type_vectors})

                    z[indices_batch, indices_truncation, :] = batch_z

                    reconstructed_data[indices_batch, indices_truncation, :, :, :] = recon

                    content_and_state_matrix[indices_batch, indices_truncation, :] = batch_content_and_state

                    self.sess.run(self.recon_err_update, feed_dict={self.truncated_batch_z_placeholder: batch_z,
                                                                    self.batch_state_initial_placeholder: batch_state_initial,
                                                                    self.truncated_batch_obs_placeholder: batch_obs,
                                                                    self.batch_content_placeholder: batch_content_vectors,
                                                                    self.batch_motion_type_placeholder: batch_motion_type_vectors})
                    batch_state_initial = batch_last_state

                    [gen_loss_avg, mse_avg] = self.sess.run([self.dyn_gen_loss_mean, self.recon_err_mean])

                    end_time = time.time()
                    print(
                        'Epoch #%d of #%d, batch #%d of #%d, truncation #%d of #%d, dynamic generator loss: %4.4f, Avg MSE: %4.4f, time: %.2fs' % (
                            epoch + 1, self.num_epochs, iBatch + 1, num_batches, i + 1, self.num_truncation,
                            gen_loss_avg,
                            mse_avg, end_time - start_time))

                content_vectors[indices_batch, :] = batch_content_vectors
                motion_type_vectors[indices_batch, :] = batch_motion_type_vectors

            if epoch % self.log_step == 0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                # saveSampleSequence(sample_videos + img_mean, self.sample_dir, epoch, col_num=10)
                saveSampleSequence(reconstructed_data, self.sample_dir, "%04d" % epoch, col_num=10, scale_method='tanh')
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)
                content_and_state_matrix.dump(os.path.join(self.model_dir, 'content_and_state.dat'))
                motion_type_vectors.dump(os.path.join(self.model_dir, 'motion_type_vectors.dat'))
                state_initial.dump(os.path.join(self.model_dir, 'state_initial.dat'))

            if epoch % 20 == 0:
                saveSampleVideo(reconstructed_data, self.result_dir, original=(train_data), global_step=epoch,
                                scale_method='tanh')

    def test(self, ckpt, info_path,  motion_path, state_initial_path, is_random_content=False,
             is_random_motion_type=False, is_random_state_initial=False):

        assert ckpt is not None, 'no checkpoint provided.'

        sample_dir_testing = os.path.join(self.output_dir, 'synthesis_sequence_testing')
        result_dir_testing = os.path.join(self.output_dir, 'final_result_testing')

        if os.path.exists(sample_dir_testing):
            shutil.rmtree(sample_dir_testing)
        if os.path.exists(result_dir_testing):
            shutil.rmtree(result_dir_testing)


        truncated_batch_z_placeholder_testing = tf.placeholder(
            shape=[None, self.truncated_backprop_length, self.z_size],
            dtype=tf.float32, name='z_testing')
        batch_state_initial_placeholder_testing = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32,
                                                                 name='state_initial_testing')

        batch_content_placeholder = tf.placeholder(shape=[None, self.content_size], dtype=tf.float32,
                                                   name='content')
        batch_motion_type_placeholder = tf.placeholder(shape=[None, self.motion_type_size], dtype=tf.float32,
                                                       name='motion_type')


        all_states_stacked, next_states = self.state_dynamic_model(truncated_batch_z_placeholder_testing,
                                                                   batch_state_initial_placeholder_testing,
                                                                   batch_content_placeholder,
                                                                   batch_motion_type_placeholder,
                                                                   dynamic_type=self.dynamic_mode, reuse=False)
        # we can manipulate the hidden values here

        all_states = tf.reshape(all_states_stacked, shape=(
            tf.shape(all_states_stacked)[0] * tf.shape(all_states_stacked)[1], self.state_size + self.content_size))

        # generator for video frames
        images_syn = self.frame_generator(all_states, reuse=False)

        images_syn = tf.reshape(images_syn, shape=(
            tf.shape(all_states_stacked)[0], tf.shape(all_states_stacked)[1], self.image_size, self.image_size, 3))



        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        z = np.random.normal(0, 1,
                             size=(self.batch_size * self.num_batches_in_test,
                                   self.truncated_backprop_length * self.num_sections_in_test,
                                   self.z_size)) * self.refsig

        if is_random_state_initial:
            state_initial = np.random.normal(0, 1, size=(self.batch_size * self.num_batches_in_test, self.state_size))
        else:
            state_initial_list_from_observed_data = np.load(state_initial_path)
            state_initial = np.tile(state_initial_list_from_observed_data[0:self.batch_size, :], [self.num_batches_in_test, 1])

        if is_random_content:
            content = np.random.normal(0, 1, size=(self.batch_size * self.num_batches_in_test, self.content_size))
        else:
            content_and_state = np.load(info_path)
            content_list_from_observed_data = content_and_state[:, 0, 0:self.content_size]
            # assign your content here (by default, we only use the content variables in the first batch)
            content = np.tile(content_list_from_observed_data[0:self.batch_size, :], [self.num_batches_in_test, 1])

        if is_random_motion_type:

            motion_type = np.random.normal(0, 1,
                                           size=(self.batch_size * self.num_batches_in_test, self.motion_type_size))

        else:
            motion_list_from_observed_data = np.load(motion_path)
            motion_type = np.tile(motion_list_from_observed_data[0:self.batch_size, :], [self.num_batches_in_test, 1])

        sequence = np.zeros(
            [self.batch_size * self.num_batches_in_test, self.truncated_backprop_length * self.num_sections_in_test,
             self.image_size, self.image_size, 3])

        for iB in xrange(self.num_batches_in_test):
            index_b = slice(iB * self.batch_size, (iB + 1) * self.batch_size)

            state_initial_batch = state_initial[index_b, :]
            content_batch = content[index_b, :]
            motion_type_batch = motion_type[index_b, :]

            for i in xrange(self.num_sections_in_test):
                index_t = slice(i * self.truncated_backprop_length, (i + 1) * self.truncated_backprop_length)
                z_batch = z[index_b, index_t, :]

                recon, next_state, all_states = self.sess.run([images_syn, next_states, all_states_stacked],
                                                              feed_dict={truncated_batch_z_placeholder_testing: z_batch,
                                                                         batch_state_initial_placeholder_testing: state_initial_batch,
                                                                         batch_content_placeholder: content_batch,
                                                                         batch_motion_type_placeholder: motion_type_batch})

                sequence[index_b, index_t, :, :, :] = recon
                state_initial_batch = next_state

        # sequence = np.concatenate(sequence, axis=1)
        saveSampleSequence(sequence, sample_dir_testing, col_num=10, scale_method='tanh')
        saveSampleVideo(sequence, result_dir_testing, scale_method='tanh')