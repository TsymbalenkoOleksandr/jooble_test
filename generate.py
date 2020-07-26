import argparse
import pandas as pd


def main(path, normalization):
    # read values and separate it based on columns
    data = pd.read_csv(path, sep="\t")
    data_features = pd.concat(
        [
            data,
            pd.DataFrame(
                [i for i in data["features"].apply(lambda x: x.split(",")).values]
            ),
        ],
        axis=1,
    )
    data_features.drop(columns=["features"], inplace=True)

    # create list of names for dataframe
    names = []
    for i in range(256):
        names.append(f"feature_2_stand_{i}")
    names.insert(0, "code")
    names.insert(0, "id_job")
    data_features.columns = names

    data_features['id_job'] = data_features['id_job'].astype(int)
    data_features['code'] = data_features['code'].astype(int)

    # add the line below in case other factors are processed differently
    data_features = data_features[data_features['code'] == 2]

    # find max_index feature and convert it to int
    data_features["max_feature_2_index"] = (
        data_features.iloc[:, 2:]
        .astype(int)
        .idxmax(axis=1)
        .apply(lambda x: x.split("_")[-1])
        .astype(int)
    )

    data_features["max_feature_2_abs_mean_diff"] = data_features.apply(
        lambda x: int(x["feature_2_stand_{0}".format(x["max_feature_2_index"])])
        - data_features["feature_2_stand_{0}".format(x["max_feature_2_index"])]
        .astype(int)
        .mean(axis=0),
        axis=1,
    )

    # normalize features
    if normalization == 'z_score':
        data_features = z_score(data_features)

    data_features.drop(columns=['code'], inplace=True)
    data_features.to_csv('test_proc.tsv', sep='\t', index=False)


def z_score(df):
    for i in range(256):
        column = df[f'feature_2_stand_{i}'].astype('float64')
        df[f'feature_2_stand_{i}'] = (column - column.mean()) / column.std(ddof=0)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to data", type=str)
    parser.add_argument("--normalization", help="type of normalization", type=str)
    args = parser.parse_args()

    main(args.path, args.normalization)
