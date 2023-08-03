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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import signal
    

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from cnn_multimodal import BinaryClassifier
from multimodal import MultimodalClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve




################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')
        


    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
        

    
    train_patient_ids, val_patient_ids = split_list(patient_ids)
    
    
    train_num_patients = len(train_patient_ids)
    val_num_patients = len(val_patient_ids)


    eeg_requested_channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    ecg_requested_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    desired_length = 30000
    
    regressor_features = list()
         
        
    train_eeg_features = list()
    train_ecg_features = list()
    train_clinical_features = list()
    train_outcomes = list()
    train_cpcs = list()
    regressor_features = list()
    
    for i in range(train_num_patients):
        print('    {}/{}...'.format(i+1, train_num_patients))
        recording_ids = find_recording_files(data_folder, train_patient_ids[i])
        current_features = get_features(data_folder, train_patient_ids[i])
        regressor_features.append(current_features)

        selected_recording_ids = [recording_ids[-1]]

        for recording_id in selected_recording_ids:
            group = "ECG"
            recording_location = os.path.join(data_folder, train_patient_ids[i], '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data, channels = reduce_channels(data, channels, ecg_requested_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = expand_channels(data, channels, ecg_requested_channels)

                if data.shape[1] < desired_length:
                    resampled_data = signal.resample(data, desired_length, axis=1)
                else:
                    downsample_factor = int(data.shape[1] / desired_length)
                    downsampled_data = signal.decimate(data, downsample_factor, axis=1)
                    resampled_data = downsampled_data[:, :desired_length]

                train_ecg_features.append(resampled_data)

            else:
                resampled_data = np.full((5, desired_length), np.nan)
                train_ecg_features.append(resampled_data)


            # Process EEG data
            group = "EEG"
            recording_location = os.path.join(data_folder, train_patient_ids[i], '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                data, channels = reduce_channels(data, channels, eeg_requested_channels)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                if data.shape[1] < desired_length:
                    resampled_data = signal.resample(data, desired_length, axis=1)
                else:
                    downsample_factor = int(data.shape[1] / desired_length)
                    downsampled_data = signal.decimate(data, downsample_factor, axis=1)
                    resampled_data = downsampled_data[:, :desired_length]

                train_eeg_features.append(resampled_data)

            else:
                resampled_data_eeg = np.full((19, desired_length), np.nan)
                train_eeg_features.append(resampled_data_eeg)


            patient_metadata = load_challenge_data(data_folder, train_patient_ids[i])
            patient_features = get_patient_features(patient_metadata)
            train_clinical_features.append(patient_features)

            current_outcome = get_outcome(patient_metadata)
            train_outcomes.append(current_outcome)

            current_cpc = get_cpc(patient_metadata)
            train_cpcs.append(current_cpc)

    train_eeg_features = np.stack(train_eeg_features, axis=0)
    train_ecg_features = np.stack(train_ecg_features, axis=0)
    train_clinical_features = np.vstack(train_clinical_features)
    train_outcomes = np.vstack(train_outcomes)
    train_cpcs = np.vstack(train_cpcs)
    regressor_features=np.vstack(regressor_features)
    
    print(train_eeg_features.shape)
    print(train_ecg_features.shape)
    print(train_clinical_features.shape)
    
    train_eeg_features = np.nan_to_num(train_eeg_features)
    train_ecg_features = np.nan_to_num(train_ecg_features)



    val_eeg_features = list()
    val_ecg_features = list()
    val_clinical_features = list()
    val_outcomes = list()
    val_cpcs = list()
    
    
    for i in range(val_num_patients):
        print('    {}/{}...'.format(i+1, val_num_patients))
        recording_ids = find_recording_files(data_folder, val_patient_ids[i])
     

        selected_recording_ids = [recording_ids[-1]]

        for recording_id in selected_recording_ids:
 
            group = "ECG"
            recording_location = os.path.join(data_folder, val_patient_ids[i], '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data, channels = reduce_channels(data, channels, ecg_requested_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = expand_channels(data, channels, ecg_requested_channels)

                if data.shape[1] < desired_length:
                    resampled_data = signal.resample(data, desired_length, axis=1)
                else:
                    downsample_factor = int(data.shape[1] / desired_length)
                    downsampled_data = signal.decimate(data, downsample_factor, axis=1)
                    resampled_data = downsampled_data[:, :desired_length]

                val_ecg_features.append(resampled_data)

            else:
                resampled_data = np.full((5, desired_length), np.nan)
                val_ecg_features.append(resampled_data)



            group = "EEG"
            recording_location = os.path.join(data_folder, val_patient_ids[i], '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                data, channels = reduce_channels(data, channels, eeg_requested_channels)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)

                if data.shape[1] < desired_length:
                    resampled_data = signal.resample(data, desired_length, axis=1)
                else:
                    downsample_factor = int(data.shape[1] / desired_length)
                    downsampled_data = signal.decimate(data, downsample_factor, axis=1)
                    resampled_data = downsampled_data[:, :desired_length]

                val_eeg_features.append(resampled_data)
               
            else:
                resampled_data_eeg = np.full((19, desired_length), np.nan)
                val_eeg_features.append(resampled_data_eeg)
              

            patient_metadata = load_challenge_data(data_folder, val_patient_ids[i])
            patient_features = get_patient_features(patient_metadata)
            val_clinical_features.append(patient_features)


            current_outcome = get_outcome(patient_metadata)
            val_outcomes.append(current_outcome)

            current_cpc = get_cpc(patient_metadata)
            val_cpcs.append(current_cpc)





    val_eeg_features = np.stack(val_eeg_features, axis=0)
    val_ecg_features = np.stack(val_ecg_features, axis=0)
    val_clinical_features = np.vstack(val_clinical_features)
    val_outcomes = np.vstack(val_outcomes)
    val_cpcs = np.vstack(val_cpcs)

    print(val_eeg_features.shape)
    print(val_ecg_features.shape)
    print(val_clinical_features.shape)

    val_eeg_features = np.nan_to_num(val_eeg_features)
    val_ecg_features = np.nan_to_num(val_ecg_features)
    


    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    clinical_imputer = SimpleImputer().fit(train_clinical_features)
    train_clinical_features = clinical_imputer.transform(train_clinical_features)
    val_clinical_features = clinical_imputer.transform(val_clinical_features)

    
    scaler_model = MinMaxScaler()
    scaler_model.fit(train_clinical_features)
    train_clinical_features = scaler_model.transform(train_clinical_features)
    val_clinical_features = scaler_model.transform(val_clinical_features)
    
    num_eeg_channels = 19
    num_ecg_channels = 5
    num_clinical_features = 8

    model = MultimodalClassifier(num_eeg_channels, num_ecg_channels, num_clinical_features)
    
    train_eeg_data=torch.from_numpy(train_eeg_features).float()
    train_ecg_data=torch.from_numpy(train_ecg_features).float()
    train_clinical_data=torch.from_numpy(train_clinical_features).float()

    train_target=torch.from_numpy(train_outcomes)


    val_eeg_data=torch.from_numpy(val_eeg_features).float()
    val_ecg_data=torch.from_numpy(val_ecg_features).float()
    val_clinical_data=torch.from_numpy(val_clinical_features).float()

    val_target=torch.from_numpy(val_outcomes)
    
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = TensorDataset(train_eeg_data, train_ecg_data, train_clinical_data, train_target)
    val_data = TensorDataset(val_eeg_data, val_ecg_data, val_clinical_data, val_target)

    batch_size =  32 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    
    model.train()
    num_epochs = 300
 

    train_losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_tprs = []
    train_fprs = []
    train_aurocs = []
    train_aurpcs = []
    train_score = []

    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_tprs = []
    val_fprs = []
    val_aurocs = []
    val_aurpcs = []
    val_score = []

    best_val_accuracy=0

    for epoch in range(num_epochs):
            train_loss = 0.0
   

            train_pred_labels = []
            train_true_labels = []
            train_pred_outcomes = []
            val_pred_labels = []
            val_true_labels = []
            val_pred_outcomes = []


            # Iterate over the batches of training data
            for i, (train_eeg_input,train_ecg_input,train_clinical_input, train_target_input) in enumerate(train_loader):



                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                train_output = model(train_eeg_input,train_ecg_input,train_clinical_input)


                # Calculate the loss
                loss = criterion(train_output, train_target_input.float())

                # Backward pass
                loss.backward()


                # Update the model parameters
                optimizer.step()

                #scheduler.step()

                # Accumulate the training loss
                train_loss += loss.item()


                train_pred_outcomes.extend(train_output.detach().numpy())
                #train_pred_outcomes = numpy.array(train_pred_outcomes)

                # Calculate accuracy (training)
                train_predicted_labels = (train_output > 0.5).float()

                train_pred_labels.extend(train_predicted_labels.cpu().numpy())
                train_true_labels.extend(train_target_input.cpu().numpy())
                #train_accuracy += (train_predicted_labels == train_target).float().mean()



            train_loss /= len(train_loader)
            #train_accuracy /= len(train_loader)

            # Validation
            val_loss = 0.0
            #val_accuracy = 0.0
            with torch.no_grad():
                model.eval()  # Set the model to evaluation mode
                for val_eeg_input,val_ecg_input,val_clinical_input, val_target_input in val_loader:
                    val_output = model(val_eeg_input,val_ecg_input,val_clinical_input)
                    val_loss += criterion(val_output, val_target_input.float()).item()
                    val_predicted_labels = (val_output > 0.5).float()

                    val_pred_labels.extend(val_predicted_labels.cpu().numpy())
                    val_true_labels.extend(val_target_input.cpu().numpy())
                    val_pred_outcomes.extend(val_output.detach().numpy())
                    #val_pred_outcomes = numpy.array(val_pred_outcomes)

                    #val_accuracy += (val_predicted_labels == val_target).float().mean()

                val_loss /= len(val_loader)
                #val_accuracy /= len(val_loader)



            train_pred_labels = np.array(train_pred_labels)
            train_true_labels = np.array(train_true_labels)
            train_outcomes = np.array(train_outcomes)

            val_pred_labels = np.array(val_pred_labels)
            val_true_labels = np.array(val_true_labels)
            val_outcomes = np.array(val_outcomes)




            train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
            train_precision = precision_score(train_true_labels, train_pred_labels)
            train_recall = recall_score(train_true_labels, train_pred_labels)
            train_tpr = train_recall
            train_fpr = 1.0 - train_tpr
      
            train_aurpc = average_precision_score(train_true_labels, train_pred_labels)
            train_score = compute_score(train_true_labels,train_pred_outcomes)

            if len(np.unique(train_pred_labels)) > 1:
                train_auroc = roc_auc_score(train_true_labels, train_pred_labels)
            else:
                train_auroc=0



            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            val_precision = precision_score(val_true_labels, val_pred_labels)
            val_recall = recall_score(val_true_labels, val_pred_labels)
            val_tpr = val_recall
            val_fpr = 1.0 - val_tpr
            #val_auroc = roc_auc_score(val_true_labels, val_pred_labels)
            val_aurpc = average_precision_score(val_true_labels, val_pred_labels)
            val_score = compute_score(val_true_labels,val_pred_outcomes)

            if len(np.unique(val_pred_labels)) > 1:
                val_auroc = roc_auc_score(val_true_labels, val_pred_labels)
            else:
                val_auroc=0       

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)
            train_tprs.append(train_tpr)
            train_fprs.append(train_fpr)
            train_aurocs.append(train_auroc)
            train_aurpcs.append(train_aurpc)


            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_tprs.append(val_tpr)
            val_fprs.append(val_fpr)
            val_aurocs.append(val_auroc)
            val_aurpcs.append(val_aurpc)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_accuracy},train score: {train_score}")
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}, Val Acc: {val_accuracy},val_score: {val_score} ")
           
            if val_accuracy >= best_val_accuracy: 
                best_val_accuracy = val_accuracy
                print('------------------Model Saved------------------------')
                
                multimodal_filename = os.path.join(model_folder, 'multimodal_outcome_model.pth')
             
                torch.save(model.state_dict(), multimodal_filename)


    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.


    regressor_imputer = SimpleImputer().fit(regressor_features)


    regressor_features = regressor_imputer.transform(regressor_features)
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(regressor_features,train_cpcs.ravel())

    save_challenge_model(model_folder, clinical_imputer, regressor_imputer,scaler_model,cpc_model)

    if verbose >= 1:
        print('Done.')


def load_challenge_models(model_folder, verbose):
    clinical_imputer_filename = os.path.join(model_folder, 'clinical_imputer_model.pkl')
    regressor_imputer_filename = os.path.join(model_folder, 'regressor_imputer_model.pkl')
    outcome_filename = os.path.join(model_folder, 'multimodal_outcome_model.pth')
    scaler_filename = os.path.join(model_folder, 'scaler_model.pkl')
    cpc_filename = os.path.join(model_folder, 'cpc_model.pkl')
        
    clinical_imputer = joblib.load(clinical_imputer_filename)
    regressor_imputer = joblib.load(regressor_imputer_filename)
    scaler_imputer = joblib.load(scaler_filename)
    outcome_model = torch.load(outcome_filename)
    cpc_model = joblib.load(cpc_filename)
  
    models = {'clinical_imputer': clinical_imputer,'regressor_imputer': regressor_imputer, 'scaler_imputer': scaler_imputer,'outcome_model': outcome_model, 'cpc_model': cpc_model}

    return models


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    clinical_imputer = models['clinical_imputer']
    regressor_imputer = models['regressor_imputer']
    scaler_imputer = models['scaler_imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    desired_length = 30000
    
    regressor_features = get_features(data_folder, patient_id)
    regressor_features = regressor_features.reshape(1, -1)


    regressor_features = regressor_imputer.transform(regressor_features)

    # Apply models to features.
    
    
    eeg_requested_channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    ecg_requested_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)
 
    
    recording_id = recording_ids[-1]
    group="ECG"
    recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))

    if os.path.exists(recording_location + '.hea'):
                    data, channels, sampling_frequency = load_recording_data(recording_location)
                    utility_frequency = get_utility_frequency(recording_location + '.hea')
                    data, channels = reduce_channels(data, channels, ecg_requested_channels)
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                    data=expand_channels(data, channels, ecg_requested_channels)
                    
                            
                    if data.shape[1] < 30000:
                        ecg_data = signal.resample(data, 30000, axis=1)
                    else:
                        downsample_factor = int(data.shape[1] / desired_length)
                        downsampled_data = signal.decimate(data, downsample_factor, axis=1)
                        ecg_data = downsampled_data[:, :desired_length]



    else:
        ecg_data=np.full((5, 30000), np.nan)



    recording_id = recording_ids[0]
    group="EEG"
    recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))

    if os.path.exists(recording_location + '.hea'):
                    data, channels, sampling_frequency = load_recording_data(recording_location)
                    data,channels=reduce_channels(data,channels,eeg_requested_channels)
                    utility_frequency = get_utility_frequency(recording_location + '.hea')
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)

                    if data.shape[1] < 30000:
                        eeg_data = signal.resample(data, 30000, axis=1)
                    else:
                        downsample_factor = int(data.shape[1] / desired_length)
                        downsampled_data = signal.decimate(data, downsample_factor, axis=1)
                        eeg_data = downsampled_data[:, :desired_length]

    else:
        eeg_data=np.full((19, 30000), np.nan)

    patient_metadata = load_challenge_data(data_folder, patient_id)
    clinical_data = get_patient_features(patient_metadata)
    
    
    eeg_data = np.nan_to_num(eeg_data)
    ecg_data = np.nan_to_num(ecg_data)

    clinical_data = clinical_data.reshape(1, -1)
    clinical_data = clinical_imputer.transform(clinical_data)
  
    clinical_data = scaler_imputer.transform(clinical_data)
    
    num_eeg_channels = 19
    num_ecg_channels = 5
    num_clinical_features = 8

    model = MultimodalClassifier(num_eeg_channels, num_ecg_channels, num_clinical_features)
   
    model.load_state_dict(outcome_model)
    model.eval()
    print(eeg_data.shape)
    print(ecg_data.shape)
    print(clinical_data.shape)
    eeg_test_data = torch.from_numpy(eeg_data).unsqueeze(0).float()
    ecg_test_data = torch.from_numpy(ecg_data).unsqueeze(0).float()
    clinical_test_data = torch.from_numpy(clinical_data).float()



    with torch.no_grad():
        test_predictions = model(eeg_test_data,ecg_test_data,clinical_test_data)
        print(test_predictions)
        outcome = np.where(test_predictions >= 0.5, 1, 0)[0]
        print(outcome)
        outcome_probability = test_predictions.squeeze().detach().cpu().numpy()


    cpc = cpc_model.predict(regressor_features)[0]


    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, clinical_imputer, regressor_imputer,scaler_model,cpc_model):
    joblib.dump(clinical_imputer, f'{model_folder}/clinical_imputer_model.pkl')
    joblib.dump(regressor_imputer, f'{model_folder}/regressor_imputer_model.pkl')
    joblib.dump(scaler_model, f'{model_folder}/scaler_model.pkl')
    joblib.dump(cpc_model, f'{model_folder}/cpc_model.pkl')

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 90.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

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
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4']
    group = 'EEG'

    if num_recordings > 0:
        recording_id = recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]]) # Convert to bipolar montage: F3-P3 and F4-P4
                eeg_features = get_eeg_features(data, sampling_frequency).flatten()
            else:
                eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
        else:
            eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
    else:
        eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel

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
    return np.hstack((patient_features,eeg_features, ecg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
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
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features

def split_list(original_list):

    shuffled_list = random.sample(original_list, len(original_list))
    

    split_point = int(len(shuffled_list) * 4/5)
    

    larger_list = []
    smaller_list = []
    

    for index, element in enumerate(shuffled_list):
        if index < split_point:
            larger_list.append(element)
        else:
            smaller_list.append(element)
    
    return larger_list, smaller_list

def compute_score(labels, outputs):
    assert len(labels) == len(outputs)
    num_instances = len(labels)

    # Use the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1]+1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs,axis=0)[::-1]

    

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j-1]
        fp[j] = fp[j-1]
        fn[j] = fn[j-1]
        tn[j] = tn[j-1]

        while i < num_instances and outputs[idx[i].item()] >= thresholds[j]:
            if labels[idx[i].item()]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs and FPRs.
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j] > 0:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            fpr[j] = float(fp[j]) / float(fp[j] + tn[j])
        else:
            tpr[j] = float('nan')
            fpr[j] = float('nan')

    # Find the largest TPR such that FPR <= 0.05.
    max_fpr = 0.05
    max_tpr = float('nan')
    if np.any(fpr <= max_fpr):
        indices = np.where(fpr <= max_fpr)
        max_tpr = np.max(tpr[indices])

    return max_tpr
