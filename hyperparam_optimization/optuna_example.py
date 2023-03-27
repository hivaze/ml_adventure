from catboost import CatBoostClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold


def catboost_one_objective(trial, train, target, cat_features=None, scoring='f1_macro', cb_threads=1, cross_val=True,
                           do_data_shuffle=False):
    # smote_strategy = trial.suggest_categorical('smote_strategy', ['minority', 'all', 'auto'])
    # smote_k_neib = trial.suggest_int('smote_k_neib', 2, 6)

    auto_class_weights = trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced'])
    # scale_pos_weight = trial.suggest_uniform('scale_pos_weight', 0.2, 2)
    grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
    score_function = trial.suggest_categorical('score_function', ['L2', 'Cosine'])

    if grow_policy != 'Lossguide':
        sampling_frequency = trial.suggest_categorical('sampling_frequency', ['PerTree', 'PerTreeLevel'])
    else:
        sampling_frequency = None
    if grow_policy == 'SymmetricTree':
        boosting_type = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])
    else:
        boosting_type = None
    if bootstrap_type == 'Bayesian':
        bagging_temperature = trial.suggest_uniform('bagging_temperature', 0.1, 3.0)
    else:
        bagging_temperature = None

    max_depth = trial.suggest_int('max_depth', 2, 8)
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    learning_rate = trial.suggest_uniform('learning_rate', 0.01, 0.15)
    # learning_rate = 0.05
    random_strength = trial.suggest_uniform('random_strength', 0.3, 1.7)
    l2_leaf_reg = trial.suggest_uniform('l2_leaf_reg', 0.5, 10.0)
    # l2_leaf_reg = 5

    cat_clf = CatBoostClassifier(auto_class_weights=auto_class_weights,
                                 cat_features=cat_features,
                                 grow_policy=grow_policy,
                                 sampling_frequency=sampling_frequency,
                                 bootstrap_type=bootstrap_type,
                                 boosting_type=boosting_type,
                                 max_depth=max_depth,
                                 score_function=score_function,
                                 n_estimators=n_estimators, learning_rate=learning_rate,
                                 random_strength=random_strength,
                                 l2_leaf_reg=l2_leaf_reg,
                                 bagging_temperature=bagging_temperature,
                                 logging_level='Silent',
                                 thread_count=cb_threads)

    if cross_val:
        score = cross_val_score(cat_clf, train, target, cv=StratifiedKFold(n_splits=5, shuffle=do_data_shuffle),
                                scoring=scoring)
        score = score.mean()
    else:
        op_X_train, op_X_test, op_y_train, op_y_test = train_test_split(train, target, test_size=0.2,
                                                                        shuffle=do_data_shuffle)
        cat_clf.fit(op_X_train, op_y_train)
        score = get_scorer(scoring)(cat_clf, op_X_test, op_y_test)

    # score = 0.7 * score + 0.3 * get_scorer(scoring)(cat_clf, train, target)

    return score


study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: catboost_one_objective(trial, X_train, y_train, cross_val=True), n_trials=1000, n_jobs=4)
