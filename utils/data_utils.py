import os
import random

import pandas as pd
from autogluon.tabular import TabularPredictor


def df_serializer(data: pd.DataFrame, mode):
    attrs_l = [col for col in data.columns if col.endswith('_l')]
    attrs_r = [col for col in data.columns if col.endswith('_r')]
    attrs = [col[:-2] for col in attrs_l]

    if mode == 'mode1':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        data['text'] = data.apply(lambda x: 'Record A is <p>' + x['text_l'] + '</p>. Record B is <p>' + x[
            'text_r'] + '</p>. Given the attributes of the two records, are they the same?', axis=1)

    elif mode == 'mode2':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        data['text'] = data.apply(lambda x: 'Given the attributes of two records, are they the same? Record A is <p>'
                                            + x['text_l'] + '</p>. Record B is <p>' + x['text_r'] + '</p>.', axis=1)
    elif mode == 'mode3':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        data['text'] = data.apply(lambda x: 'Given the attributes of two records, are they the same? Record A is '
                                            + x['text_l'] + '. Record B is ' + x['text_r'] + '.', axis=1)

    elif mode == 'mode4':
        template_l = '{}: {}, ' * (len(attrs) - 1) + '{}: {}'
        template_r = '{}: {}, ' * (len(attrs) - 1) + '{}: {}'
        attrs = [attr[:-2] for attr in attrs_l]
        data['text_l'] = data.apply(
            lambda x: template_l.format(*[item for pair in zip(attrs, x[attrs_l].fillna('N/A')) for item in pair]),
            axis=1)
        data['text_r'] = data.apply(
            lambda x: template_r.format(*[item for pair in zip(attrs, x[attrs_r].fillna('N/A')) for item in pair]),
            axis=1)
        data['text'] = data.apply(lambda x: 'Given the attributes of two records, are they the same? Record A is '
                                            + x['text_l'] + '. Record B is ' + x['text_r'] + '.', axis=1)
    else:
        raise ValueError('Invalid mode')
    return data[['text', 'label']]


def one_pos_two_neg(train_df, dataset_dir):
    dataset_name = dataset_dir.split('/')[-1]
    if len(train_df) < 1200:
        print(f'The training set size of {dataset_name} is less than 1200, which will all be kept.', flush=True)
        return train_df
    else:
        print(f'The training set size of {dataset_name} is larger than 1200, we will do down-sampling '
              f'with one_pos_two_neg to maximally 1200 pairs.', flush=True)
        train_pos_pairs = train_df[train_df['label'] == 1]
        train_neg_pairs = train_df[train_df['label'] == 0]
        train_neg_pairs_sampled = train_neg_pairs.sample(n=2*len(train_pos_pairs), random_state=42)
        train_df_sampled = pd.concat([train_pos_pairs, train_neg_pairs_sampled])
        train_num = min(1200, len(train_df_sampled))
        train_df_sampled = train_df_sampled.sample(n=train_num, random_state=42).reset_index(drop=True)

        return train_df_sampled


def automl_filter(train_df, dataset_dir):
    dataset_name = dataset_dir.split('/')[-1]
    if len(train_df) < 1200:
        print(f'The training set size of {dataset_name} is less than 1200, which will all be kept.', flush=True)
        return train_df
    else:
        print(f'The training set size of {dataset_name} is larger than 1200, we will do down-sampling '
              f'with automl_filter to maximally 1200 pairs.', flush=True)
        automl_data_dir = '/'.join(dataset_dir.split('/')[:-2] + ['automl'] + [dataset_dir.split('/')[-1]])
        train_preds_df = pd.read_csv(os.path.join(automl_data_dir, 'train_preds.csv'))

        train_pos_wrong_preds = train_df[(train_preds_df['prediction']!=train_df['label']) & (train_df['label']==1)]
        train_pos_num = min(400, train_df['label'].sum())
        if len(train_pos_wrong_preds) < train_pos_num:
            train_pos_supply = train_df[(train_preds_df['prediction']==train_df['label']) & (train_df['label']==1)].sample(n=train_pos_num-len(train_pos_wrong_preds), random_state=42)
            train_pos_df = pd.concat([train_pos_wrong_preds, train_pos_supply])
        else:
            train_pos_df = train_pos_wrong_preds.sample(n=train_pos_num, random_state=42)
        train_neg_df = train_df[train_df['label']==0].sample(n=2*train_pos_num, random_state=42)

        filtered_train_df = pd.concat([train_pos_df, train_neg_df]).reset_index(drop=True)

        return filtered_train_df


def automl_filter_flip(train_df, dataset_dir):
    """An augmentation strategy: permute the training set after filtering by AutoML model."""
    filtered_train_df = automl_filter(train_df, dataset_dir)
    dataset_name = dataset_dir.split('/')[-1]
    print(f'then, the training data of {dataset_name} will be augmented by flipping.', flush=True)
    # swap the left and right records
    left_columns = [col for col in filtered_train_df.columns if col.endswith('_l')]
    right_columns = [col for col in filtered_train_df.columns if col.endswith('_r')]
    attrs_flipped = []
    for i, row in filtered_train_df.iterrows():
        left = row[left_columns].values
        label = row['label']
        right = row[right_columns].values
        attrs = list(right) + [label] + list(left)
        attrs_flipped.append(attrs)

    new_train_df = pd.concat([filtered_train_df, pd.DataFrame(attrs_flipped, columns=filtered_train_df.columns)])
    new_train_df = new_train_df.drop_duplicates().reset_index(drop=True)
    return new_train_df


def automl_filter_permute(train_df, dataset_dir):
    """An augmentation strategy: permute the training set after filtering by AutoML model."""
    filtered_train_df = automl_filter(train_df, dataset_dir)
    dataset_name = dataset_dir.split('/')[-1]
    print(f'then, the training data of {dataset_name} will be augmented by permuting.', flush=True)
    # permute columns
    left_columns = [col for col in filtered_train_df.columns if col.endswith('_l')]
    right_columns = [col for col in filtered_train_df.columns if col.endswith('_r')]
    attrs_permuted = []
    for i, row in filtered_train_df.iterrows():
        left_columns_permuted = random.sample(left_columns, len(left_columns))
        left = row[left_columns_permuted].values
        label = row['label']
        right_columns_permuted = random.sample(right_columns, len(right_columns))
        right = row[right_columns_permuted].values
        attrs = list(right) + [label] + list(left)
        attrs_permuted.append(attrs)

    new_train_df = pd.concat([filtered_train_df, pd.DataFrame(attrs_permuted, columns=filtered_train_df.columns)])
    new_train_df = new_train_df.drop_duplicates().reset_index(drop=True)
    return new_train_df


def automl_filter_flip_permute(train_df, dataset_dir):
    """An augmentation strategy: permute the training set after filtering by AutoML model."""
    new_train_df = pd.concat([automl_filter_flip(train_df, dataset_dir),
                              automl_filter_permute(train_df, dataset_dir)]).\
        drop_duplicates().reset_index(drop=True)
    return new_train_df


def read_single_row_data(dataset_dir, mode, sample_func='', print_info=True):
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(dataset_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

    if sample_func:
        sample_func = eval(sample_func)
        train_df = sample_func(train_df, dataset_dir)

    train_df = df_serializer(train_df, mode)
    valid_df = df_serializer(valid_df, mode)
    test_df = df_serializer(test_df, mode)

    if print_info:
        dataset_name = dataset_dir.split('/')[-1]
        print(f"We will use the {mode} partition of the {dataset_name} dataset.", flush=True)
        print(f"An example(row level) after the serialization is:\n{test_df.iloc[0]['text']}", flush=True)

    return train_df, valid_df, test_df


def read_multi_row_data(dataset_dirs, mode='mode1', sample_func='one_pos_two_neg', print_info=True):
    dfs = [read_single_row_data(dataset_dir, mode, sample_func, print_info=False) for dataset_dir in dataset_dirs]
    train_dfs, valid_dfs, test_dfs = zip(*dfs)

    if print_info:
        sample_texts = [valid_df.iloc[0]['text'] for valid_df in valid_dfs]
        print(f"Examples(row level) after the serialization are:\n", flush=True)
        [print(sample_text, flush=True) for sample_text in sample_texts]

    concat_train_df = pd.concat(train_dfs, ignore_index=True)
    concat_valid_df = pd.concat(valid_dfs, ignore_index=True)
    concat_test_df = pd.concat(test_dfs, ignore_index=True)
    print(f'The size of the row level concatenation for training, validation, and test are: {len(concat_train_df)}, '
          f'{len(concat_valid_df)}, {len(concat_test_df)}', flush=True)

    return concat_train_df, concat_valid_df, concat_test_df


def downsample_attr_pairs(group):
    # Balancing labels
    min_count = group['label'].value_counts().min()
    balanced = group.groupby('label').sample(n=min_count, random_state=1)

    # Further downsampling to 800 if necessary
    if len(balanced) > 800:
        balanced = balanced.sample(n=800, random_state=1)

    return balanced


def read_multi_attr_data(dataset_dirs, mode='mode1'):
    train_dfs = [pd.read_csv(os.path.join(dataset_dir, 'attr_train.csv')) for dataset_dir in dataset_dirs]
    valid_dfs = [pd.read_csv(os.path.join(dataset_dir, 'attr_valid.csv')) for dataset_dir in dataset_dirs]
    test_dfs = [pd.read_csv(os.path.join(dataset_dir, 'attr_test.csv')) for dataset_dir in dataset_dirs]

    concat_train_df = pd.concat(train_dfs, ignore_index=True)
    concat_valid_df = pd.concat(valid_dfs, ignore_index=True)
    concat_test_df = pd.concat(test_dfs, ignore_index=True)
    final_train_df = concat_train_df.groupby('attribute').apply(downsample_attr_pairs).reset_index(drop=True)[
        ['left_value', 'right_value', 'label']]
    final_train_df.columns = ['value_l', 'value_r', 'label']
    final_valid_df = concat_valid_df.groupby('attribute').apply(downsample_attr_pairs).reset_index(drop=True)[
        ['left_value', 'right_value', 'label']]
    final_valid_df.columns = ['value_l', 'value_r', 'label']
    final_test_df = concat_test_df.groupby('attribute').apply(downsample_attr_pairs).reset_index(drop=True)[
        ['left_value', 'right_value', 'label']]
    final_test_df.columns = ['value_l', 'value_r', 'label']
    final_train_df = df_serializer(final_train_df, mode)
    final_valid_df = df_serializer(final_valid_df, mode)
    final_test_df = df_serializer(final_test_df, mode)

    return final_train_df, final_valid_df, final_test_df
