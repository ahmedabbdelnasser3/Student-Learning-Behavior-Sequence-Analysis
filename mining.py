import pandas as pd
from prefixspan import PrefixSpan

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss


# Load & Preprocess

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path)

    df = df.dropna(subset=['student_id', 'timestamp', 'success_label'])

    df['success_label'] = df['success_label'].astype(int)
    df = df.drop_duplicates()

    df = df.fillna({
        'time_spent_minutes': df['time_spent_minutes'].median() if 'time_spent_minutes' in df.columns else 0,
        'quiz_score': 0,
        'assignment_score': 0,
        'notes_taken': 0
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['student_id', 'timestamp'])

    def create_activity(row):
        activity = []
        activity.append("video" if row.get('video_watched_percent', 0) >= 70 else "skip_video")
        activity.append("notes" if row.get('notes_taken', 0) > 0 else "no_notes")
        activity.append("quiz_pass" if row.get('quiz_score', 0) >= 70 else "quiz_fail")
        return "_".join(activity)

    df['activity'] = df.apply(create_activity, axis=1)

    from sklearn.preprocessing import LabelEncoder
    activity_encoder = LabelEncoder()
    df['activity_encoded'] = activity_encoder.fit_transform(df['activity'])

    sequences = (
        df.groupby(['student_id', 'success_label'])['activity']
        .apply(list)
        .reset_index()
    )

    sequences['sequence_length'] = sequences['activity'].apply(len)

    Q1 = sequences['sequence_length'].quantile(0.25)
    Q3 = sequences['sequence_length'].quantile(0.75)
    IQR = Q3 - Q1
    sequences['sequence_length'] = sequences['sequence_length'].clip(
        Q1 - 1.5 * IQR,
        Q3 + 1.5 * IQR
    )

    return df, sequences

# PrefixSpan
def run_prefixspan(sequences, min_support=5, top_k=5):

    high_sequences = sequences[sequences['success_label'] == 1]['activity'].tolist()
    low_sequences = sequences[sequences['success_label'] == 0]['activity'].tolist()

    ps_high = PrefixSpan(high_sequences)
    ps_low = PrefixSpan(low_sequences)

    ps_high.minlen = 2
    ps_low.minlen = 2

    high_patterns = sorted(
        ps_high.frequent(min_support),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    low_patterns = sorted(
        ps_low.frequent(min_support),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return high_patterns, low_patterns

# Simple GSP (kept)
def run_gsp(sequences, min_support=5):

    def is_subsequence(sub, seq):
        it = iter(seq)
        return all(item in it for item in sub)

    all_sequences = sequences['activity'].tolist()
    items = set(item for seq in all_sequences for item in seq)

    patterns = []

    for item in items:
        support = sum(is_subsequence([item], seq) for seq in all_sequences)
        if support >= min_support:
            patterns.append((support, [item]))

    return sorted(patterns, reverse=True)

# ML Dataset
def build_ml_dataset(df, sequences):

    features = sequences[['student_id', 'sequence_length']].copy()

    if 'time_spent_minutes' in df.columns:
        avg_time = (
            df.groupby('student_id')['time_spent_minutes']
            .mean()
            .reset_index()
            .rename(columns={'time_spent_minutes': 'avg_time_spent'})
        )
        features = features.merge(avg_time, on='student_id', how='left')

    features = features.fillna(0)

    X = features.drop(columns=['student_id'])
    y = sequences['success_label']

    return X, y

# Train & Evaluate
def train_and_evaluate_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, best_rf.predict(X_test)),
        'log_loss': log_loss(y_test, best_rf.predict_proba(X_test)),
        'report_dict': classification_report(y_test, best_rf.predict(X_test), output_dict=True)
    }

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr.predict(X_test)),
        'log_loss': log_loss(y_test, lr.predict_proba(X_test)),
        'report_dict': classification_report(y_test, lr.predict(X_test), output_dict=True)
    }

    return results
