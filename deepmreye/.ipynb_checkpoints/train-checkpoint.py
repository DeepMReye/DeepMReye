import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import r2_score
from deepmreye import architecture
from deepmreye.util import util
from deepmreye.util import data_generator


def train_model(dataset, generators, opts, clear_graph=True, save=False, model_path='./', workers=4, use_multiprocessing=True, models=None, return_untrained=False, verbose=0):
    # Clear session if needed
    if clear_graph:
        K.clear_session()
    if use_multiprocessing:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
        workers = 1

    # Unpack generators
    (training_generator, testing_generator, single_testing_generators, single_testing_names,
     single_training_generators, single_training_names, full_testing_list, full_training_list) = generators

    # Test datagenerator and get representative X and y
    ((X, y), _) = next(training_generator)
    if verbose > 0:
        print('Input shape {}, Output shape {}'.format(X.shape, y.shape))
        print('Subjects in training set: {}, Subjects in test set: {}'.format(
            len(single_training_generators), len(single_testing_generators)))

    # Learning rate scheduler
    lr_sched = util.step_decay_schedule(initial_lr=opts['lr'], decay_factor=0.9, num_epochs=opts['epochs'])

    # Get model
    if models is None:
        model, model_inference = architecture.create_standard_model(X.shape[1::], opts)
    else:
        model, model_inference = models
    if return_untrained:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        return (model, model_inference)

    # Train model
    if verbose > 1:
        print(model.summary(line_length=200))
    model.fit(training_generator, steps_per_epoch=opts['steps_per_epoch'], epochs=opts['epochs'], validation_data=testing_generator,
              validation_steps=opts['validation_steps'], callbacks=[lr_sched], use_multiprocessing=use_multiprocessing, workers=workers)

    # Save model weights
    if save:
        model_inference.save_weights(model_path + 'modelinference_{}.h5'.format(dataset))
    if use_multiprocessing:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    return (model, model_inference)


def evaluate_model(dataset, model, generators, save=False, model_path='./', model_description='', verbose=0, **args):
    (training_generator, testing_generator, single_testing_generators, single_testing_names,
     single_training_generators, single_training_names, full_testing_list, full_training_list) = generators
    evaluation, scores = dict(), dict()
    for idx, subj in enumerate(full_testing_list):
        X, real_y = data_generator.get_all_subject_data(subj)
        (pred_y, euc_pred) = model.predict(X, verbose=verbose-2, batch_size=16)
        evaluation[subj] = {'real_y': real_y, 'pred_y': pred_y, 'euc_pred': euc_pred}

        # Quantify predictions
        df_scores = util.get_model_scores(real_y, pred_y, euc_pred, **args)
        scores[subj] = df_scores

        # Print evaluation
        if verbose > 0:
            print(util.color.BOLD + '{} / {} - Model Performance for {}'.format(idx + 1, len(single_testing_names), subj) + util.color.END)
            if verbose > 1:
                pd.set_option('display.width', 120)
                pd.options.display.float_format = '{:.3f}'.format
                print(df_scores)
            else:
                print('Default: r={:.3f}, subTR: r={:.3f}, Euclidean Error: {:.3f}°'.format(
                    df_scores[('Pearson', 'Mean')]['Default'], df_scores[('Pearson', 'Mean')]['Default subTR'], df_scores[('Eucl. Error', 'Mean')]['Default']))
            print('\n')

    # Save dict
    if save:
        np.save(model_path + 'results{}_{}.npy'.format(model_description, dataset), evaluation)

    return (evaluation, scores)
