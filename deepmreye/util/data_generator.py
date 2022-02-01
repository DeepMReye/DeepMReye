import random
import os
import numpy as np
from pprint import pprint
from . import util


def create_generators(full_training_list, full_testing_list, batch_size=8, withinsubject_split=None, augment_list=[0, 0, 0], mixed_batches=True, inner_timesteps=None):
    # Overall training and testing generator
    training_generator = data_generator(full_training_list, batch_size, training=True, inner_timesteps=inner_timesteps,
                                        augment_list=augment_list, mixed_batches=mixed_batches, withinsubject_split=withinsubject_split)
    testing_generator = data_generator(full_testing_list, batch_size, training=False, inner_timesteps=inner_timesteps,
                                       mixed_batches=mixed_batches, withinsubject_split=withinsubject_split)

    # For each testing subject create a single generator
    single_testing_generators, single_testing_names = get_single_data_generators(full_testing_list, batch_size, string_cut=7, training=False,
                                                                                 mixed_batches=mixed_batches, withinsubject_split=withinsubject_split, inner_timesteps=inner_timesteps)
    single_training_generators, single_training_names = get_single_data_generators(full_training_list, batch_size, string_cut=7, training=True,
                                                                                   augment_list=augment_list, mixed_batches=mixed_batches, withinsubject_split=withinsubject_split, inner_timesteps=inner_timesteps)

    training_subjects_string = [os.path.splitext(os.path.basename(p))[0] for p in full_training_list]
    test_subjects_string = [os.path.splitext(os.path.basename(p))[0] for p in full_testing_list]

    print(util.color.BOLD + 'Training set ({}) contains {} subjects: '.format(
        os.path.dirname(full_training_list[0]), len(full_training_list)) + util.color.END)
    pprint(training_subjects_string, compact=True)
    print(util.color.BOLD + 'Test set ({}) contains {} subjects: '.format(
        os.path.dirname(full_testing_list[0]), len(full_testing_list)) + util.color.END)
    pprint(test_subjects_string, compact=True)
    # Should be a dict.
    return (training_generator, testing_generator, single_testing_generators, single_testing_names, single_training_generators, single_training_names)


def create_holdout_generators(datasets, train_split=0.6, **args):
    full_training_list, full_testing_list = list(), list()
    for fn_data in datasets:
        this_file_list = [fn_data + p for p in os.listdir(fn_data)]
        np.random.shuffle(this_file_list)
        train_test_split = int(train_split * len(this_file_list))
        this_training_list = this_file_list[0:train_test_split]
        this_testing_list = this_file_list[train_test_split::]
        full_training_list.extend(this_training_list)
        full_testing_list.extend(this_testing_list)
    (training_generator, testing_generator, single_testing_generators, single_testing_names, single_training_generators,
     single_training_names) = create_generators(full_training_list, full_testing_list, **args)

    return (training_generator, testing_generator, single_testing_generators, single_testing_names,
            single_training_generators, single_training_names, full_testing_list, full_training_list)


def create_cv_generators(dataset, num_cvs=5, **args):
    # Create lists for each cv
    this_file_list = [dataset + p for p in os.listdir(dataset)]
    np.random.shuffle(this_file_list)
    cv_split = np.array_split(this_file_list, num_cvs)
    cv_return = []
    for idx, cvs in enumerate(cv_split):
        full_testing_list = cvs.tolist()
        full_training_list = np.concatenate([x for i, x in enumerate(cv_split) if i != idx]).tolist()

        (training_generator, testing_generator,
         single_testing_generators, single_testing_names,
         single_all_generators, single_all_names) = create_generators(full_training_list, full_testing_list, **args)
        cv_return.append((training_generator, testing_generator, single_testing_generators, single_testing_names,
                          single_all_generators, single_all_names, full_testing_list, full_training_list))

    return cv_return


def create_leaveoneout_generators(datasets, training_subset=None, **args):
    loo_return = []
    for idx, dataset in enumerate(datasets):
        full_testing_list = [dataset + p for p in os.listdir(dataset)]
        training_datasets = [x for i, x in enumerate(datasets) if i != idx]
        full_training_list = [tds + p for tds in training_datasets for p in os.listdir(tds)]

        if training_subset is not None:
            size_before = len(full_training_list)
            full_training_list = [tds for tds in full_training_list if os.path.basename(
                os.path.dirname(tds)) + '/' + os.path.basename(tds) in training_subset]
            print('Using subset ({} / {}) for training {}'.format(len(full_training_list), size_before, dataset))

        (training_generator, testing_generator,
         single_testing_generators, single_testing_names,
         single_all_generators, single_all_names) = create_generators(full_training_list, full_testing_list, **args)
        loo_return.append((training_generator, testing_generator, single_testing_generators, single_testing_names,
                           single_all_generators, single_all_names, full_testing_list, full_training_list))

    return loo_return


def get_single_data_generators(fn_list, batch_size, string_cut=4, **args):
    generators, names = list(), list()
    for subject in fn_list:
        this_generator = data_generator([subject], batch_size, **args)
        generators.append(this_generator)
        this_name = os.path.basename(subject)[:-4]
        if len(this_name) > string_cut:
            this_name = this_name[-string_cut:]
        names.append(this_name)
    return generators, names


def data_generator(file_list, batch_size, training=False, mixed_batches=True,
                   withinsubject_split=None, augment_list=[0, 0, 0], inner_timesteps=None):
    """
    Take a random subject, load it and return a batched subset    
    """
    all_nonan_indices = get_nonan_indices(file_list)
    start_tr, end_tr = get_start_end_tr(withinsubject_split, training)
    if withinsubject_split is None:
        file_list = [file_list[x] for x in range(0, len(all_nonan_indices)) if all_nonan_indices[x].size > 0]
        all_nonan_indices = [all_nonan_indices[x] for x in range(0, len(all_nonan_indices)) if all_nonan_indices[x].size > 0]
    else:
        new_filelist, new_nonan = [], []
        for x in range(0, len(all_nonan_indices)):
            try:
                get_subject_data(file_list[x], batch_size=1, start_tr=start_tr,
                                 end_tr=end_tr, nonan_indices=all_nonan_indices[x])
                new_filelist.append(file_list[x])
                new_nonan.append(all_nonan_indices[x])
            except ValueError:
                pass
        file_list = new_filelist
        all_nonan_indices = new_nonan
    while True:
        if mixed_batches:
            fn_subjects, nonan_indices = zip(*random.choices(list(zip(file_list, all_nonan_indices)), k=batch_size))
            X, y = [], []
            for subj, nans in zip(fn_subjects, nonan_indices):
                tmp_X, tmp_y = get_subject_data(subj, batch_size=1, start_tr=start_tr,
                                                end_tr=end_tr, nonan_indices=nans)
                X.append(tmp_X[0])
                y.append(tmp_y[0])
        else:
            fn_subject = np.random.choice(file_list)
            X, y = get_subject_data(fn_subject, batch_size=batch_size, start_tr=start_tr, end_tr=end_tr)
        # Downsample to number of inner timesteps (subTR)
        if inner_timesteps is not None:
            y = [y_tmp[np.linspace(0, y[0].shape[0] - 1, inner_timesteps, dtype=int), :] for y_tmp in y]
        X, y = np.array(X)[..., np.newaxis], np.array(y)
        # Augmentation
        if training and any(augment_list):
            X = util.augment_input(X, rotation=augment_list[0], shift=augment_list[1], zoom=augment_list[2])

        yield [[X, y], []]


def get_subject_data(fn_subject, batch_size=None, sample_index=None, start_tr=None, end_tr=None, nonan_indices=None):
    data = np.load(fn_subject, mmap_mode='r')
    if batch_size is not None:
        try:
            _ = data['identifier_0']
            divisor = 3
        except:
            divisor = 2
        num_trs = len(data) // divisor
        start_tr, end_tr = get_tr_indices(num_trs, start_tr, end_tr)
        nonan_indices = nonan_indices[(nonan_indices >= start_tr) & (nonan_indices <= end_tr)]
        batch_indices = np.random.choice(nonan_indices, batch_size)
        X = [data['data_' + str(b)] for b in batch_indices]
        y = [data['label_' + str(b)] for b in batch_indices]
    elif sample_index is not None:
        X = data['data_' + str(sample_index)]
        y = data['label_' + str(sample_index)]
    else:
        return data
    return X, y


def get_tr_indices(num_trs, start_tr, end_tr):
    if start_tr is None:
        start_tr = 0
    else:
        start_tr = int(start_tr * num_trs)
    if end_tr is None:
        end_tr = num_trs
    else:
        end_tr = int(end_tr * num_trs)
    return (start_tr, end_tr)


def get_start_end_tr(withinsubject_split, training):
    if withinsubject_split:
        if training:
            start_tr = withinsubject_split[0]
            end_tr = withinsubject_split[1]
        else:
            if withinsubject_split[0] == 0:
                start_tr = withinsubject_split[1]
                end_tr = None
            else:
                start_tr = None
                end_tr = withinsubject_split[0]
    else:
        start_tr = None
        end_tr = None
    return (start_tr, end_tr)


def get_nonan_indices(file_list):
    nonan_indices = []
    for fn_subject in file_list:
        data = np.load(fn_subject, mmap_mode='r')
        try:
            _ = data['identifier_0']
            divisor = 3
        except:
            divisor = 2
        num_trs = len(data) // divisor
        y = np.array([data['label_' + str(sample_index)] for sample_index in range(0, num_trs)])
        nonan_index = ~np.any(np.isnan(y), axis=(1, 2))
        nonan_indices.append(np.where(nonan_index)[0])
    return nonan_indices


def get_all_subject_data(fn_subject):
    data = np.load(fn_subject, mmap_mode='r')
    try:
        _ = data['identifier_0']
        divisor = 3
    except:
        divisor = 2
    num_trs = len(get_subject_data(fn_subject)) // divisor
    subject_data_X, subject_data_y = [], []
    for sample_index in range(0, num_trs):
        subject_data_X.append(data['data_' + str(sample_index)])
        subject_data_y.append(data['label_' + str(sample_index)])
    subject_data_X, subject_data_y = np.array(subject_data_X), np.array(subject_data_y)

    return (subject_data_X[..., np.newaxis], subject_data_y)
