import pandas as pd
import numpy as np
import time
from datetime import datetime
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, roc_auc_score
import joblib

def infer_problem_type(df, target_col):
    if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
        return 'classification'
    if df[target_col].nunique() < 20: 
        return 'classification'
    return 'regression'

def check_data_quality(df, target_col):
    report = {}
    n_rows = len(df)
    
    # Leakage
    leakage_cols = []
    for col in df.columns:
        if col != target_col:
            if df[col].nunique() == n_rows or (df[col].dtype in ['int64', 'object'] and df[col].nunique() > n_rows * 0.95):
                leakage_cols.append(col)
    report['leakage_warnings'] = leakage_cols
    
    # Missing Values
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # Outliers
    num_cols = df.select_dtypes(include=np.number).columns
    outliers = {}
    for col in num_cols:
        if col != target_col:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            count = outlier_mask.sum()
            if count > 0:
                outliers[col] = int(count)
    report['outliers'] = outliers
    
    # Class Imbalance
    problem_type = infer_problem_type(df, target_col)
    report['is_imbalanced'] = False
    if problem_type == 'classification':
        counts = df[target_col].value_counts(normalize=True)
        if counts.min() < 0.2:
            report['is_imbalanced'] = True
            report['imbalance_ratio'] = counts.to_dict()
            
    return report

def build_preprocessing_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Track feature names correctly
    feature_names = numeric_features + categorical_features
    return preprocessor, feature_names

def get_models(problem_type, is_imbalanced):
    cw = 'balanced' if is_imbalanced else None
    if problem_type == 'classification':
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight=cw),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight=cw),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'SVC': SVC(probability=True, class_weight=cw, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
    else:
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42),
            'SVR': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor()
        }

def tune_model_optuna(best_model_name, problem_type, X_train, y_train, preprocessor, is_imbalanced, scoring_metric, n_trials=15):
    cw = 'balanced' if is_imbalanced else None
    step_name = 'classifier' if problem_type == 'classification' else 'regressor'
    
    def objective(trial):
        if problem_type == 'classification':
            if best_model_name == 'Logistic Regression':
                C = trial.suggest_float('C', 0.01, 100.0, log=True)
                solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
                model = LogisticRegression(max_iter=2000, class_weight=cw, C=C, solver=solver)
            elif best_model_name == 'Random Forest':
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
                model = RandomForestClassifier(random_state=42, class_weight=cw, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            elif best_model_name == 'XGBoost':
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
                max_depth = trial.suggest_int('max_depth', 3, 9)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
            elif best_model_name == 'SVC':
                C = trial.suggest_float('C', 0.1, 100.0, log=True)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
                model = SVC(probability=True, class_weight=cw, random_state=42, C=C, kernel=kernel)
            else: # KNN
                n_neighbors = trial.suggest_int('n_neighbors', 3, 11, step=2)
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                p = trial.suggest_int('p', 1, 2)
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        else: # regression
            if best_model_name == 'Linear Regression':
                model = LinearRegression()
            elif best_model_name == 'Random Forest':
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
                model = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            elif best_model_name == 'XGBoost':
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
                max_depth = trial.suggest_int('max_depth', 3, 9)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
                model = XGBRegressor(random_state=42, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
            elif best_model_name == 'SVR':
                C = trial.suggest_float('C', 0.1, 100.0, log=True)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
                model = SVR(C=C, kernel=kernel)
            else: # KNN
                n_neighbors = trial.suggest_int('n_neighbors', 3, 11, step=2)
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                p = trial.suggest_int('p', 1, 2)
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), (step_name, model)])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scoring_metric, n_jobs=-1)
        return cv_scores.mean() if problem_type == 'classification' else -cv_scores.mean()

    # Create Optuna study
    direction = 'maximize' if problem_type == 'classification' else 'minimize'
    # For models with no hyperparams like Linear Regression, skip tuning
    if best_model_name == 'Linear Regression':
        pipe = Pipeline(steps=[('preprocessor', preprocessor), (step_name, LinearRegression())])
        return pipe.fit(X_train, y_train)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    # Rebuild best pipeline from best params
    best_params = study.best_params
    if problem_type == 'classification':
        if best_model_name == 'Logistic Regression':
            best_model = LogisticRegression(max_iter=2000, class_weight=cw, **best_params)
        elif best_model_name == 'Random Forest':
            best_model = RandomForestClassifier(random_state=42, class_weight=cw, **best_params)
        elif best_model_name == 'XGBoost':
            best_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
        elif best_model_name == 'SVC':
            best_model = SVC(probability=True, class_weight=cw, random_state=42, **best_params)
        else:
            best_model = KNeighborsClassifier(**best_params)
    else:
        if best_model_name == 'Random Forest':
            best_model = RandomForestRegressor(random_state=42, **best_params)
        elif best_model_name == 'XGBoost':
            best_model = XGBRegressor(random_state=42, **best_params)
        elif best_model_name == 'SVR':
            best_model = SVR(**best_params)
        else:
            best_model = KNeighborsRegressor(**best_params)
            
    final_best_pipe = Pipeline(steps=[('preprocessor', preprocessor), (step_name, best_model)])
    final_best_pipe.fit(X_train, y_train)
    return final_best_pipe

def get_feature_importances(best_pipe, feature_names, problem_type):
    step_name = 'classifier' if problem_type == 'classification' else 'regressor'
    model = best_pipe.named_steps[step_name]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        return None
    
    # Map to names
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    return fi_df

def run_automl(df, target_col, progress_callback=None):
    quality_report = check_data_quality(df, target_col)
    problem_type = infer_problem_type(df, target_col)
    
    if progress_callback:
        progress_callback("Analyzing Data Quality & Schema...")
        
    schema = list(df.drop(columns=[target_col]).columns)
    
    if problem_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    else:
        le = None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor, feature_names_ordered = build_preprocessing_pipeline(X_train)
    models = get_models(problem_type, quality_report.get('is_imbalanced', False))
    
    results = {}
    zoo_results = {}
    best_score = -np.inf if problem_type == 'classification' else np.inf
    best_model_name = None
    best_model_instance = None
    best_param_grid = None
    feature_importances = None

    scoring_metric = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'

    # 1. Model Zoo Evaluation
    for name, model in models.items():
        if progress_callback:
            progress_callback(f"Baseline Evaluation: {name}...")
        
        step_name = 'classifier' if problem_type == 'classification' else 'regressor'
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            (step_name, model)
        ])
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scoring_metric, n_jobs=-1)
        mean_cv = cv_scores.mean() if problem_type == 'classification' else -cv_scores.mean()
        
        zoo_results[name] = mean_cv
        
        is_best = (mean_cv > best_score) if problem_type == 'classification' else (mean_cv < best_score)
        if is_best:
            best_score = mean_cv
            best_model_name = name
            best_model_instance = model

    # 2. Hyperparameter Tuning for Best Model using Optuna
    if progress_callback:
        progress_callback(f"Optuna Tuning Best Model: {best_model_name}...")
        
    start_time = time.time()
    final_best_pipe = tune_model_optuna(best_model_name, problem_type, X_train, y_train, preprocessor, quality_report.get('is_imbalanced', False), scoring_metric, n_trials=15)
    train_time = time.time() - start_time
    
    # 3. Evaluate Final Model on Test Set
    y_pred = final_best_pipe.predict(X_test)
    
    if problem_type == 'classification':
        y_prob = final_best_pipe.predict_proba(X_test) if hasattr(final_best_pipe.named_steps[step_name], 'predict_proba') else None
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = np.nan
        if y_prob is not None:
            try:
                if len(np.unique(y_test)) > 2:
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                else:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
            except Exception:
                pass
        
        results[best_model_name] = {'Accuracy': acc, 'F1': f1, 'AUC': auc, 'Time (s)': round(train_time, 2)}
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[best_model_name] = {'MSE': mse, 'R2': r2, 'Time (s)': round(train_time, 2)}

    if progress_callback:
        progress_callback("Extracting feature importances and calibrating model...")

    feature_importances = get_feature_importances(final_best_pipe, feature_names_ordered, problem_type)
    
    if problem_type == 'classification':
        try:
            calibrated = CalibratedClassifierCV(final_best_pipe.named_steps['classifier'], cv='prefit', method='isotonic')
            calibrated.fit(X_test, y_test)
            final_best_pipe.steps[-1] = ('classifier', calibrated)
        except Exception as e:
            pass 

    # Keep test-set actuals & predictions for evaluation charts in the UI
    y_prob_out = None
    if problem_type == 'classification':
        try:
            y_prob_out = final_best_pipe.predict_proba(X_test)
        except Exception:
            pass

    return {
        'problem_type': problem_type,
        'results': results,
        'zoo_results': zoo_results,
        'best_model_name': best_model_name,
        'best_pipeline': final_best_pipe,
        'label_encoder': le,
        'feature_importances': feature_importances,
        'quality_report': quality_report,
        'schema': schema,
        'X_train': X_train,           # for SHAP
        'y_test': y_test.tolist(),    # ground-truth test labels (encoded)
        'y_pred': y_pred.tolist(),    # model predictions on test set (encoded)
        'y_prob': y_prob_out,         # predicted probabilities (classification only)
        'class_names': le.classes_.tolist() if le is not None else None,
    }

def save_model(results_dict):
    """Serialize model to bytes in memory (no file written to disk)."""
    import io
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.pkl"
    buf = io.BytesIO()
    joblib.dump({
        'pipeline': results_dict['best_pipeline'],
        'label_encoder': results_dict['label_encoder'],
        'schema': results_dict['schema'],
        'problem_type': results_dict['problem_type']
    }, buf)
    buf.seek(0)
    return buf.read(), filename

