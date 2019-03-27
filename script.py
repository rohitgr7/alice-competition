from pathlib import Path
from scipy.sparse import hstack
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PATH = Path('./data')
AUTHOR = 'Rohit_Gupta'
SEED = 7
TFIDF_PARAMS = {'max_features': 200000, 'ngram_range': (1, 7)}
LOGIT_BEST_PARAMS = {'C': 7., 'solver': 'liblinear', 'random_state': SEED}
SITES = [f'site{i}' for i in range(1, 11)]
TIMES = [f'time{i}' for i in range(1, 11)]


def create_submission(test_preds, session_ids):
    df = pd.DataFrame({'session_id': session_ids, 'target': test_preds})
    df.to_csv(f'submission_alice_{AUTHOR}.csv', header=True, index=False)


def vectorize_sites(train_df, test_df, params):
    train_sessions = train_df[SITES].fillna(0.).astype(np.int32).apply(
        lambda row: ' '.join(row.astype(np.str)), axis=1)
    test_sessions = test_df[SITES].fillna(0.).astype(np.int32).apply(
        lambda row: ' '.join(row.astype(np.str)), axis=1)

    vectorizer = TfidfVectorizer(**params)
    train_sessions = vectorizer.fit_transform(train_sessions)
    test_sessions = vectorizer.transform(test_sessions)

    return vectorizer, train_sessions, test_sessions


def add_time_features(df):
    tmp_df = pd.DataFrame()
    cat_feats = []

    # The original data was skewed so converted to log(1+x)
    tmp_df['duration'] = np.log1p(
        (df[TIMES].max(1) - df[TIMES].min(1)).dt.seconds)

    # Worked better than given in kernels
    tmp_df['year_month'] = 12 * \
        (df['time1'].dt.year - 2013) + df['time1'].dt.month
    cat_feats.append('year_month')

    # Month and Hour
    tmp_df['start_month'] = df['time1'].dt.month
    cat_feats.append('start_month')
    tmp_df['start_hour'] = df['time1'].dt.hour
    cat_feats.append('start_hour')

    # Weeks
    tmp_df['week_of_year'] = df['time1'].dt.weekofyear
    cat_feats.append('week_of_year')
    tmp_df['start_weekday'] = df['time1'].dt.dayofweek
    cat_feats.append('start_weekday')

    # Weekend?
    tmp_df['weekend'] = (tmp_df['start_weekday'] > 4).astype(np.int32)

    # Daytime
    tmp_df['day'] = ((df['time1'].dt.hour >= 6) & (
        df['time1'].dt.hour <= 11)).astype(np.int32)
    tmp_df['evening'] = ((df['time1'].dt.hour >= 12) & (
        df['time1'].dt.hour <= 17)).astype(np.int32)
    tmp_df['night'] = ((df['time1'].dt.hour >= 18) & (
        df['time1'].dt.hour <= 23)).astype(np.int32)

    # Alice is mostly active in these hours
    tmp_df['active1213'] = df['time1'].apply(
        lambda t: 12 <= t.hour <= 13).astype(np.int32)
    tmp_df['active1618'] = df['time1'].apply(
        lambda t: 16 <= t.hour <= 18).astype(np.int32)

    return tmp_df, cat_feats


def main():
    # Get the data
    print('[INFO] Loading data...')
    train_df = pd.read_csv(PATH / 'train_sessions.csv', parse_dates=TIMES)
    test_df = pd.read_csv(PATH / 'test_sessions.csv', parse_dates=TIMES)

    train_df.drop('session_id', axis=1, inplace=True)
    train_df.drop_duplicates(inplace=True)
    y_train = train_df.pop('target')
    test_sessions = test_df.pop('session_id')

    # Tfidf vectorization
    print('[INFO] Tfidf Vectorization...')
    vectorizer, train_sites_vec, test_sites_vec = vectorize_sites(
        train_df, test_df, TFIDF_PARAMS)

    # Time related features
    print('[INFO] Adding time-related features...')
    train_time_data, cat_feats = add_time_features(train_df)
    test_time_data, _ = add_time_features(test_df)

    # Categorical features to one-hot
    print('[INFO] Converting to one-hot...')
    train_split = train_df.shape[0]
    full_time_data = pd.get_dummies(
        pd.concat((train_time_data, test_time_data)), columns=cat_feats)
    train_oh_data = full_time_data[:train_split]
    test_oh_data = full_time_data[train_split:]

    # Stacking up and converting to sparse matrix
    print('[INFO] Stacking up...')
    train_data = hstack([train_sites_vec, train_oh_data]).tocsr()
    test_data = hstack([test_sites_vec, test_oh_data]).tocsr()

    # Tuned hyerparameters by cross-validation
    print('[INFO] Training...')
    model = LogisticRegression(**LOGIT_BEST_PARAMS)
    model.fit(train_data, y_train)

    print('[INFO] Predicting...')
    test_preds = model.predict_proba(test_data)[:, 1]

    print('[INFO] Creating submission...')
    create_submission(test_preds, test_sessions)
    print('[INFO] Done!!!')


if __name__ == '__main__':
    main()
