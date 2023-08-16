#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from scipy import signal
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import joblib

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Description: Trains a challenge model to perform on challenge data.
# Inputs: data_folder (str, directory of training data), model_folder (str, directory of where the model will be stored), verbose (int, # how much information is printed out)
# Outputs: None (directly)
def train_challenge_model(data_folder, model_folder, verbose):
    # If verbose >= 1, print out training status
    if verbose >= 1:
        print('Finding the Challenge data...')

    # Store the patient IDs, and store the number of total patients
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    # Return message if no patient data was obtained
    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels. If verbose >= 1, print out data status
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    # Adjust the verbosity
    verbose = 2
    
    # Create empty lists for features (X), outcomes (classifier y), and CPCs (regressor y)
    features = list()
    outcomes = list()
    cpcs = list()

    # For each patient:
    for i in range(num_patients):
        
        # If verbosity >= 2, print out progress of the loop
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Append the features of one patient to list of overall features
        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels. Load in the patient info in .txt file
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        
        # Append the outcomes of one patient to list of overall outcomes
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        
        # Append the outcomes of one patient to list of overall CPCs
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    # Vertically stack the featurs
    features = np.vstack(features)
    
    # Flatten the outcomes
    outcomes = np.vstack(outcomes)
    outcomes = outcomes.ravel()
    
    # Flatten the CPCs
    cpcs = np.vstack(cpcs)
    cpcs = cpcs.ravel()
    
    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    # Nested Cross-Validation:
    # Initialize the classifiers: KNN, DT, RF, GB, and LGB
    clf1 = KNeighborsClassifier()
    clf2 = DecisionTreeClassifier(random_state = 123)
    clf3 = RandomForestClassifier(random_state = 123)
    clf4 = GradientBoostingClassifier(random_state = 123)
    clf5 = lgb.LGBMClassifier(random_state = 123)
    
    # Set up classifier parameter grids: KNN, DT, RF, GB, and LGB
    clf1_params = [{'n_neighbors': [9, 11, 13, 15, 17],
                    'p': [1, 2]}]
    
    clf2_params = [{'criterion': ['gini', 'entropy'], 
                    'max_depth': list(range(1, 5)) + [None]}]
    
    clf3_params = [{'n_estimators': [10, 100, 500, 1000],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': list(range(1, 7)) + [None]}]
    
    clf4_params = [{'learning_rate': [0.001, 0.01, 0.1, 1],
                    'max_depth': list(range(1, 5)) + [None],
                    'n_estimators': [10, 100, 500, 1000, 2000]}]
    
    clf5_params = [{'max_depth': list(range(1, 5)) + [-1],
                    'learning_rate': [0.001 ,0.005, 0.01, 0.1],
                    'n_estimators': [10, 100, 250, 500]}]
    
    # Set up the GridCV objects for each model
    gridcvs = {}
    inner_cv = StratifiedKFold(n_splits = 2, 
                               shuffle = True, 
                               random_state = 123)
    
    for params, estimator, name in zip((clf1_params, clf2_params, clf3_params, clf4_params, clf5_params), 
                                       (clf1, clf2, clf3, clf4, clf5), 
                                       ('KNN_clf', 'DT_clf', 'RF_clf', 'GB_clf', 'LGB_clf')):
        score = 'accuracy'
        gcv = GridSearchCV(estimator = estimator, param_grid = params, scoring = score, n_jobs = -1, cv = inner_cv, refit = True)
        gridcvs[name] = gcv
        
    # Outer Loop Code:
    # Create a directory for the outer scores of each model.
    final_outer_scores_clfs = {}
    
    # For each name and estimator in the grid search...
    for name, gs_est in gridcvs.items():
        # Print model name and template
        print(50 * '-', '\n')
        print('Algorithm:', name)
        print('    Inner Loop:')

        # Initialize outer_scores list and the K fold for the outer CV
        outer_scores = []
        outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123)

        # For training indices and validation indices in the training data...
        for train_idx, valid_idx in outer_cv.split(features, outcomes):
            
            # Train the model with given parameters
            gridcvs[name].fit(features[train_idx], outcomes[train_idx])
            
            # Print the best accuracy on the training fold and the best parameters for this model
            print('\n        Best Accuracy (on training fold) %.2f%%' % (gridcvs[name].best_score_ * 100))
            print('       Best parameters:', gridcvs[name].best_params_)
                
            # Run the models on the validation data and print the validation accuracy
            outer_scores.append(gridcvs[name].best_estimator_.score(features[valid_idx], outcomes[valid_idx]))
            print('        Accuracy (on validation fold) %.2f%%' % (outer_scores[-1] * 100))
            
        # Print the results (mean/std of the validation accuracies)
        print('\n    Outer Loop:')
        print('        Accuracy: %.2f%% +/- %.2f' % (np.mean(outer_scores) * 100, np.std(outer_scores) * 100))
        
        # Store the median validation score in the dictionary
        final_outer_scores_clfs[name] = np.mean(outer_scores)
    
    # Find the maximum median validation score, find the name of the model, and set this model as the best classifier
    best_clf_name = max(final_outer_scores_clfs, key = final_outer_scores_clfs.get)
    best_clf_gcv = gridcvs[best_clf_name]
    best_clf = best_clf_gcv.best_estimator_
    best_clf_params = best_clf_gcv.best_params_
    
    # Print the name of the best classifier
    print(best_clf)

    # If verbose >= 1, print status message
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Train the models
    best_clf_with_params = best_clf.__class__(**best_clf_params)
    bg_clf = BaggingClassifier(base_estimator = best_clf_with_params, n_estimators = 250)
    outcome_model = bg_clf.fit(features, outcomes)
    # outcome_model = best_clf.fit(features, outcomes)
    cpc_model = RandomForestRegressor().fit(features, cpcs)

    # Save the models into the model folder
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    # If verbose >= 1, print status message
    if verbose >= 1:
        print('Done.')

# Do *not* change arguments of function!
# Description: Loads a challenge model in.
# Inputs: model_folder (str, directory of model), verbose (int, # how much information is printed out)
# Outputs: model object
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
# Description: Runs the challenge model to perform on challenge data.
# Inputs: model object, data_folder (int, # how much information is printed out)
# Outputs: model object
def run_challenge_models(models, data_folder, patient_id, verbose):
    # Obtain the imputer, outcome model, and CPC model from the model file
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 99.9]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=-1, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=-1, verbose='error')
    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 * ((data - min_value) / (max_value - min_value)) - 1.0
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    
    # Store the patient metadata (.txt file).
    patient_metadata = load_challenge_data(data_folder, patient_id)
    
    # Store all of the NAMES of the recording files.
    recording_ids = find_recording_files(data_folder, patient_id)
    
    # Store the number of recording files for a given patient.
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    group = 'EEG'

    if num_recordings > 0:
        # Get the recording ID
        recording_id = recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        
        if os.path.exists(recording_location + '.hea'):
            # Get the data, channels, and sampling frequency
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = np.array([data[0, :] - data[1, :], 
                                 data[1, :] - data[2, :], 
                                 data[3, :] - data[4, :], 
                                 data[4, :] - data[5, :], 
                                 data[5, :] - data[6, :], 
                                 data[6, :] - data[7, :], 
                                 data[8, :] - data[9, :], 
                                 data[9, :] - data[10, :], 
                                 data[11, :] - data[12, :], 
                                 data[12, :] - data[13, :], 
                                 data[13, :] - data[14, :], 
                                 data[14, :] - data[15, :], 
                                 data[16, :] - data[17, :], 
                                 data[17, :] - data[18, :]]) #Convert to bipolar montages
                
                eeg_features = get_eeg_features(data, sampling_frequency).flatten()
            else:
                eeg_features = float('nan') * np.ones(70) # 14 bipolar channels * 5 features / channel
        else:
            eeg_features = float('nan') * np.ones(70) # 14 bipolar channels * 5 features / channel
    else:
        eeg_features = float('nan') * np.ones(70) # 14 bipolar channels * 5 features / channel

    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        recording_id = recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            features = get_ecg_features(data)
            ecg_features = expand_channels(features, channels, ecg_channels).flatten()
        else:
            ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
    else:
        ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel

    # Extract features.
    return np.hstack((patient_features, eeg_features, ecg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    #sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, n_fft=128, sfreq=sampling_frequency,  fmin=0.5,  fmax=4.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, n_fft=128, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, n_fft=128, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, n_fft=128, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)
        gamma_psd,  _ = mne.time_frequency.psd_array_welch(data, n_fft=128, sfreq=sampling_frequency, fmin=30.0, fmax=100.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
        gamma_psd_mean = np.nanmean(gamma_psd, axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = gamma_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, gamma_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features


'''elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)'''