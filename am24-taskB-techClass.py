#!/usr/bin/env python
# coding: utf-8

# # Task B: Pitched/Percussive Binary classification
# ---

# In[ ]:


import dataset.aGPTset.ExpressiveGuitarTechniquesDataset as agptset
import os
import librosa
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

import am24utils
from am24utils import Run

dataset = agptset.import_db()

DOTEST = False
VERBOSE = False
DB_PATH = 'dataset/aGPTset'
printVerbose = lambda x: print(x) if VERBOSE else None
MULTIPROCESSING = False

if MULTIPROCESSING:
    import multiprocessing as mp


# In[ ]:


print("Filtering the Dataset...")
filtered_notes_db, filtered_files_db = am24utils.filter_files_db(dataset)
#make sure that no audio_file_path contains "impro"
assert filtered_notes_db.index.get_level_values(1).str.contains('impro').sum() == 0, "Some audio_file_path contain 'impro' (%i)"%(filtered_notes_db.index.get_level_values(1).str.contains('impro').sum())
print("Done (%i notes in the filtered db)."%len(filtered_notes_db))


# In[ ]:


def get_onsetlist_filename_ispercussive(filtered_notes_db:pd.DataFrame, filtered_files_db:pd.DataFrame):
    onsetlist = []
    filenames = []
    players = []
    ispercussive = []
    for file in filtered_notes_db.index.get_level_values(1).unique():
        if file in filtered_files_db.index:
            afp = filtered_files_db[filtered_files_db.index == file].full_audiofile_path.values
            assert len(afp) == 1, "More than one audio file path for file %s"%file
            filenames.append(afp[0])
            cur_onset_list = filtered_notes_db.loc[filtered_notes_db.index.get_level_values(1) == file].onset_label_samples.values
            cur_isPercussive = filtered_notes_db.loc[filtered_notes_db.index.get_level_values(1) == file].isPercussive.values
            assert len(cur_onset_list) == len(cur_isPercussive), "Onset list and isPercussive have different lengths"
            # print('%s has %i onsets and %i isPercussive'%(file,len(cur_onset_list),len(cur_isPercussive)))
            cur_onset_list = [int(x) for x in cur_onset_list]
            onsetlist.append(cur_onset_list)
            cur_isPercussive = [True if 'true' in x.lower() else False for x in cur_isPercussive]
            ispercussive.append(cur_isPercussive)
            cur_player = filtered_files_db[filtered_files_db.index == file].player_id.values
            assert len(cur_player) == 1, "More than one player for file %s"%file
            cur_player = int(cur_player[0])
            players.append(cur_player)
        else:
            raise ValueError("File %s not found in the files db"%file)
        
    return onsetlist, filenames, ispercussive, players
        



onsetlist,filenames,ispercussive,playerlist  = get_onsetlist_filename_ispercussive(filtered_notes_db,filtered_files_db)
assert len(onsetlist) == len(filenames) == len(ispercussive) == len(playerlist), "Different lengths for onsetlist, filenames, ispercussive and playerlist"

packedData = (onsetlist,filenames,ispercussive,playerlist)
# if DOTEST:
#     packedData = (onsetlist[:10],filenames[:10],ispercussive[:10],playerlist[:10])


# In[ ]:


#from importlib import reload
#reload(am24utils)


# In[ ]:


def load_and_compute_features_for_file(cur_filename, 
                                       cur_onsetlist, 
                                       cur_isOnsetPercussivelist, 
                                       cur_player,
                                       window_size_samples,
                                       onset_perturbation_distribution,
                                       onset_perturbation_max_samples, 
                                       onset_perturbation_min_samples):
    # print('Processing file %s'%cur_filename)
    assert len(cur_onsetlist) == len(cur_isOnsetPercussivelist), "onsetlist and isOnsetPercussivelist have different lengths (%i != %i)"%(len(cur_onsetlist), len(cur_isOnsetPercussivelist))
    if onset_perturbation_distribution is not None:
        # print('Applying onset perturbation to file %s'%cur_filename)
        cur_onsetlist = am24utils.apply_onset_perturbation(cur_onsetlist, onset_perturbation_distribution, onset_perturbation_max_samples, onset_perturbation_min_samples)

    # print('Computing features for file %s'%cur_filename)
    Xfn, yfn = am24utils.get_Xy(cur_filename, cur_onsetlist, cur_isOnsetPercussivelist, window_size_samples)
    
    assert len(Xfn) == len(yfn), "Xfn and yfn have different lengths (%i != %i)"%(len(Xfn), len(yfn))

    print('.',end='', flush=True)
    playerlist = [cur_player]*len(Xfn)
    return Xfn, yfn,playerlist

def run_taskB(runs, packedData, classifier='KNN'):
    onsetlist_list,filenames_list,isOnsetPercussive_list, player_list = packedData
    assert len(onsetlist_list) == len(filenames_list) == len(isOnsetPercussive_list) == len(player_list), "Different lengths for onsetlist_list, filenames_list, isOnsetPercussive_list and player_list"
    for ridx,run in enumerate(runs):
        # Add color and bold to the print
        print('Running task B for Run:%s [%i,%i]'%(run.name,ridx+1,len(runs)), end='\r')
        print('+--%s--Arguments--------------+'%(run.name))
        print('| Window size: %i'%run.window_size_samples)
        print('| Onset perturbation distribution: %s'%run.onset_perturbation_distribution)
        print('| Onset perturbation max samples: %i'%run.onset_perturbation_max_samples)
        print('| Onset perturbation min samples: %i'%run.onset_perturbation_min_samples)
        print('+-------------------------------------+')


        X,y,group = [],[],[]
        if MULTIPROCESSING:
            # replace previous commented block with parallel processing
            pool = mp.Pool(mp.cpu_count()//2)
            # results = [pool.apply_async(load_and_compute_features_for_file, args=(os.path.join(DB_PATH,filenames_list[i]), onsetlist_list[i], isOnsetPercussive_list[i], player_list[i])) for i in range(len(onsetlist_list))]
            results = [pool.apply_async(load_and_compute_features_for_file, 
                        args=(os.path.join(DB_PATH,filenames_list[i]), 
                            onsetlist_list[i], 
                            isOnsetPercussive_list[i], 
                            player_list[i], 
                            run.window_size_samples, 
                            run.onset_perturbation_distribution, 
                            run.onset_perturbation_max_samples, 
                            run.onset_perturbation_min_samples)) for i in range(len(onsetlist_list))]
            pool.close()
            pool.join()
            print('All files processed.')
            for r in results:
                Xfnret, yfnret, playervecret = r.get()
                X.extend(Xfnret)
                y.extend(yfnret)
                group.extend(playervecret)
        else:
            for i in range(len(onsetlist_list)):
                Xfn, yfn, playervec = load_and_compute_features_for_file(os.path.join(DB_PATH,filenames_list[i]),
                                                                        onsetlist_list[i], 
                                                                        isOnsetPercussive_list[i], 
                                                                        player_list[i], 
                                                                        run.window_size_samples, 
                                                                        run.onset_perturbation_distribution, 
                                                                        run.onset_perturbation_max_samples, 
                                                                        run.onset_perturbation_min_samples)
                X.extend(Xfn)
                y.extend(yfn)
                group.extend(playervec) # Add the player id to the group list, repeated for each onset
        



        assert len(X) == len(y), "X and y have different lengths (%i != %i)"%(len(X), len(y))
        assert len(X) == len(group), "X and group have different lengths (%i != %i)"%(len(X), len(group))
        # print('X',X)
        # print('y',y)
        # print(group)
        

        from sklearn.model_selection import StratifiedGroupKFold
        N_FOLDS = 3
        skf = StratifiedGroupKFold(n_splits=N_FOLDS)

        skf.get_n_splits(X, y, group)
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y, group)):
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            group_train, group_test = np.array(group)[train_index], np.array(group)[test_index]
            print('Fold %i/%i'%(fold_idx+1,N_FOLDS))
            print('Train:', len(X_train), len(y_train),'Groups: ', sorted(list(set(group_train))))
            print('Test:', len(X_test), len(y_test),'Groups: ', sorted(list(set(group_test))))
            
            # For each test set, print the percentage of percussive and pitched notes
            percussive_perc = sum(y_test)/len(y_test)
            pitched_perc = 1 - percussive_perc
            print('Test Percussive: %.2f%%'%(percussive_perc*100))
            print('Test Pitched: %.2f%%'%(pitched_perc*100))



            # Use SMOTE to balance the classes
            # Print the percentage of percussive and pitched notes before SMOTE
            percussive_perc = sum(y_train)/len(y_train)
            pitched_perc = 1 - percussive_perc
            print('Train Percussive (before SMOTE): %.2f%%'%(percussive_perc*100))
            print('Train Pitched (before SMOTE): %.2f%%'%(pitched_perc*100))
            
            from imblearn.over_sampling import SMOTE
            smote = SMOTE()
            print('Balancing classes with SMOTE ...')
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Print the percentage of percussive and pitched notes after SMOTE
            percussive_perc = sum(y_train)/len(y_train)
            pitched_perc = 1 - percussive_perc
            print('Train Percussive (after SMOTE): %.2f%%'%(percussive_perc*100))
            print('Train Pitched (after SMOTE): %.2f%%'%(pitched_perc*100))
        
            if classifier.upper()=='KNN':
                from sklearn.neighbors import KNeighborsClassifier as KNN
                # Train KNN
                print('Training', classifier, '...')
                knn = KNN(n_neighbors=3)
                knn.fit(X_train, y_train)
                
                # Test
                y_pred = knn.predict(X_test)
            elif classifier.upper()=='SVM':
                from sklearn.svm import SVC
                # Train SVM
                svm = SVC()
                print('Training', classifier, '...')
                svm.fit(X_train, y_train)
                
                # Test
                y_pred = svm.predict(X_test)
            
            # # Metrics
            classification_report = sk.metrics.classification_report(y_test, y_pred, labels=[True, False],target_names=['Percussive', 'Pitched'])
            print('-Classification FOLD %i/%i-'%(fold_idx+1,N_FOLDS))
            print('\t\t\t'+classification_report.replace('\n','\n\t\t\t'))
            print('\n\n')

            confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred, labels=[True, False], )
            print('-Confusion Matrix FOLD %i/%i-'%(fold_idx+1,N_FOLDS))
            print('\t\t\t'+str(confusion_matrix).replace('\n','\n\t\t\t'))
            print('\n\n')
            print('\n\n')
            
        
            run.foldresults.append({
                'classification_report': classification_report,
                'confusion_matrix': confusion_matrix,
                'accuracy': sk.metrics.accuracy_score(y_test, y_pred),
                'dictclassifiction_report' : sk.metrics.classification_report(y_test, y_pred, output_dict=True)
            })
        print('Run %s done.'%run.name)

        # TODO: CHECK WHY THIS PRINTED "Average classification report over %i folds: 9" WHILE WE WERE RUNNING 3 FOLDS ??????????
        print('Average classification report over %i folds:', len(run.foldresults))
        # print("'Average classification report over %i folds:"%N_FOLDS)
        mean_dict = am24utils.report_average([r['dictclassifiction_report'] for r in run.foldresults])
        mean_accuracy = sum([r['accuracy'] for r in run.foldresults])/len(run.foldresults)
        run.results = {'mean_classification_report_dict': mean_dict,'mean_classification_report_str': am24utils.classification_report_dict2print(mean_dict), 'mean_accuracy': mean_accuracy, 'number_folds': N_FOLDS}
        print(run.results['mean_classification_report_str'])
        
to_run = am24utils.get_run_list()

for ridx,run in enumerate(to_run):
    print(run.name)

           
run_taskB(to_run, packedData=packedData)


# In[ ]:


# Save Results
import pickle, datetime

resdir_path = os.path.join('results','task-B','date_%s'%(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
os.makedirs(resdir_path)

bakfilename  = 'taskB_KNN_results.pickle'

with open(os.path.join(resdir_path,bakfilename), 'wb') as f:
    pickle.dump(to_run, f)
    

am24utils.plot_runs(to_run, arg_metric='mean_f1')
plt.savefig(os.path.join(resdir_path,'accuracy_KNN.png'))
plt.savefig(os.path.join(resdir_path,'accuracy_KNN.pdf'))


# In[ ]:


# previous_run = to_run[0]
# for run in to_run[1:]:
#     print('Run %s'%run.name)
#     print("Len(Fold_Results): ",len(run.foldresults))
#     print("Type: ",type(run.foldresults))
#     print("Run and Previous Run Fold Results are equal: ",run.foldresults == previous_run.foldresults)
#     previos_run = run


# In[ ]:


# TEST WITH RESNET AND SQUEEZENET

def load_and_compute_features_for_file(cur_filename, 
                                       cur_onsetlist, 
                                       cur_isOnsetPercussivelist, 
                                       cur_player,
                                       window_size_samples,
                                       onset_perturbation_distribution,
                                       onset_perturbation_max_samples, 
                                       onset_perturbation_min_samples):
    # print('Processing file %s'%cur_filename)
    assert len(cur_onsetlist) == len(cur_isOnsetPercussivelist), "onsetlist and cur_dynamicslist have different lengths (%i != %i)"%(len(cur_onsetlist), len(cur_isOnsetPercussivelist))
    if onset_perturbation_distribution is not None:
        # print('Applying onset perturbation to file %s'%cur_filename)
        cur_onsetlist = am24utils.apply_onset_perturbation(cur_onsetlist, onset_perturbation_distribution, onset_perturbation_max_samples, onset_perturbation_min_samples)

    # print('Computing features for file %s'%cur_filename)
    Xfn, yfn = am24utils.get_Xy(cur_filename, cur_onsetlist, cur_isOnsetPercussivelist, window_size_samples, features=["log-mel","mfcc"])
    
    assert len(Xfn) == len(yfn), "Xfn and yfn have different lengths (%i != %i)"%(len(Xfn), len(yfn))

    print('.',end='',flush=True)
    playerlist = [cur_player]*len(Xfn)
    return Xfn, yfn,playerlist

def run_taskB(runs, packedData, classifier='RESNET'):
    onsetlist_list,filenames_list,isOnsetPercussive_list, player_list = packedData
    assert len(onsetlist_list) == len(filenames_list) == len(isOnsetPercussive_list) == len(player_list), "Different lengths for onsetlist_list, filenames_list, dynamics_list and player_list"
    for ridx,run in enumerate(runs):
        print('Running task B for Run:%s [%i,%i]'%(run.name,ridx+1,len(runs)), end='\r')
        print('+--%s--Arguments--------------+'%(run.name))
        print('| Window size: %i'%run.window_size_samples)
        print('| Onset perturbation distribution: %s'%run.onset_perturbation_distribution)
        print('| Onset perturbation max samples: %i'%run.onset_perturbation_max_samples)
        print('| Onset perturbation min samples: %i'%run.onset_perturbation_min_samples)
        print('+-------------------------------------+')

        '''
        # for i in range(len(onsetlist_list)):
        #     Xfn, yfn, playervec = load_and_compute_features_for_file(os.path.join(DB_PATH,filenames_list[i]),
        #                                                              onsetlist_list[i], 
        #                                                              dynamics_list[i], 
        #                                                              player_list[i])
        #     X.extend(Xfn)
        #     y.extend(yfn)
        #     group.extend(playervec) # Add the player id to the group list, repeated for each onset
        '''

        X,y,group = [],[],[]
        if MULTIPROCESSING:
            # replace previous commented block with parallel processing
            pool = mp.Pool(mp.cpu_count())
            # results = [pool.apply_async(load_and_compute_features_for_file, args=(os.path.join(DB_PATH,filenames_list[i]), onsetlist_list[i], dynamics_list[i], player_list[i])) for i in range(len(onsetlist_list))]
            results = [pool.apply_async(load_and_compute_features_for_file, 
                        args=(os.path.join(DB_PATH,filenames_list[i]), 
                            onsetlist_list[i], 
                            isOnsetPercussive_list[i], 
                            player_list[i], 
                            run.window_size_samples, 
                            run.onset_perturbation_distribution, 
                            run.onset_perturbation_max_samples, 
                            run.onset_perturbation_min_samples)) for i in range(len(onsetlist_list))]
            pool.close()
            pool.join()

            print('All files processed.')
            
            X,y,group = [],[],[]
            for r in results:
                Xfnret, yfnret, playervecret = r.get()
                X.extend(Xfnret)
                y.extend(yfnret)
                group.extend(playervecret)
        else:
            for i in range(len(onsetlist_list)):
                Xfn, yfn, playervec = load_and_compute_features_for_file(os.path.join(DB_PATH,filenames_list[i]),
                                                                        onsetlist_list[i], 
                                                                        isOnsetPercussive_list[i], 
                                                                        player_list[i],
                                                                        run.window_size_samples, 
                                                                        run.onset_perturbation_distribution, 
                                                                        run.onset_perturbation_max_samples, 
                                                                        run.onset_perturbation_min_samples)
                X.extend(Xfn)
                y.extend(yfn)
                group.extend(playervec) # Add the player id to the group list, repeated for each onset

        







        
        assert len(X) == len(y), "X and y have different lengths (%i != %i)"%(len(X), len(y))
        assert len(X) == len(group), "X and group have different lengths (%i != %i)"%(len(X), len(group))
        # print('X',X)
        # print('y',y)
        # print(group)
        

        from sklearn.model_selection import StratifiedGroupKFold, KFold
        N_FOLDS = 3
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        skf.get_n_splits(X, y, group)
        run.results = []
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y, group)):
        # for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y, group)):
            X_features = [x[1:] for x in X]
            X_specs = [x[0] for x in X]
            X_train, X_test = np.array(X_features)[train_index], np.array(X_features)[test_index]
            S_train, S_test = np.array(X_specs)[train_index], np.array(X_specs)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            group_train, group_test = np.array(group)[train_index], np.array(group)[test_index]
            print('Fold %i/%i'%(fold_idx+1,N_FOLDS))
            print('Train:', len(X_train), len(y_train),'Groups: ', sorted(list(set(group_train))))
            print('Test:', len(X_test), len(y_test),'Groups: ', sorted(list(set(group_test))))
            
            
            # Use SMOTE to balance the classes
            # Print the percentage of percussive and pitched notes before SMOTE
            percussive_perc = sum(y_train)/len(y_train)
            pitched_perc = 1 - percussive_perc
            print('Train Percussive (before SMOTE): %.2f%%'%(percussive_perc*100))
            print('Train Pitched (before SMOTE): %.2f%%'%(pitched_perc*100))
          
            # Use SMOTE to balance the classes
            from imblearn.over_sampling import SMOTE
            smote = SMOTE()
            print('Balancing classes with SMOTE ...')
            shape_train = S_train.shape[1:]
            X_resampled, y_train = smote.fit_resample(S_train.reshape(-1, shape_train[0]*shape_train[1]*shape_train[2]), y_train)
            # Reshape the data
            S_train = X_resampled.reshape(-1, shape_train[0], shape_train[1], shape_train[2])
            
            # Print the percentage of percussive and pitched notes after SMOTE
            percussive_perc = sum(y_train)/len(y_train)
            pitched_perc = 1 - percussive_perc
            print('Train Percussive (after SMOTE): %.2f%%'%(percussive_perc*100))
            print('Train Pitched (after SMOTE): %.2f%%'%(pitched_perc*100))


            if classifier.upper()=='KNN':
                from sklearn.neighbors import KNeighborsClassifier as KNN
                # Train KNN
                print('Training', classifier, '...')
                knn = KNN(n_neighbors=3)
                knn.fit(X_train, y_train)
                
                # Test
                y_pred = knn.predict(X_test)
            elif classifier.upper()=='SVM':
                from sklearn.svm import SVC
                # Train SVM
                svm = SVC()
                print('Training', classifier, '...')
                svm.fit(X_train, y_train)
                
                # Test
                y_pred = svm.predict(X_test)
                
            # TODO: uncomment NN classifier and test it    
            elif classifier.upper()=='RESNET':
                import torch
                from am24utils import AudioResNet
                # Train ResNet for audio classification
                device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
                resnet = AudioResNet(num_classes=2, fine_tuning=True).to(device)
                #am24utils.save_or_reset_weights(resnet, 'resnet_weights_starting_weights.pth')
                print('Training', classifier, '...')
                resnet.fit(S_train, y_train)
                
                # Save the weights
                #am24utils.save_or_reset_weights(resnet, f'squeezenet_weights_f{fold_idx+1}/{N_FOLDS}.pth')
                
                # Test
                _, y_pred = resnet.test(S_test, y_test)
                
                resnet.cpu()
                resnet = None
                
                del resnet
                # Free memory
                torch.cuda.empty_cache()
                # Use garbage collector
                import gc
                gc.collect()
                
            elif classifier.upper()=='SQUEEZENET':
                import torch
                from am24utils import AudioSqueezeNet
                # Train ResNet for audio classification
                device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
                squeezenet = AudioSqueezeNet(num_classes=2).to(device)
                #am24utils.save_or_reset_weights(squeezenet, 'squeezenet_starting_weights.pth')
                print('Training', classifier, '...')
                squeezenet.fit(S_train, y_train)
                
                # Save the weights
                #am24utils.save_or_reset_weights(squeezenet, f'squeezenet_weights_f{fold_idx+1}/{N_FOLDS}.pth')
                
                # Test
                _, y_pred = squeezenet.test(S_test, y_test)
                
                squeezenet.cpu()
                squeezenet = None
                
                del squeezenet
                # Free memory
                torch.cuda.empty_cache()
                # Use garbage collector
                import gc
                gc.collect()
                
            
            # # Metrics
            classification_report = sk.metrics.classification_report(y_test, y_pred, labels=[True, False],target_names=['Percussive', 'Pitched'])
            print('-Classification FOLD %i/%i-'%(fold_idx+1,N_FOLDS))
            print('\t\t\t'+classification_report.replace('\n','\n\t\t\t'))
            print('\n\n')

            confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred, labels=[True, False], )
            print('-Confusion Matrix FOLD %i/%i-'%(fold_idx+1,N_FOLDS))
            print('\t\t\t'+str(confusion_matrix).replace('\n','\n\t\t\t'))
            print('\n\n')
            print('\n\n')
            
        
            run.foldresults.append({
                'classification_report': classification_report,
                'confusion_matrix': confusion_matrix,
                'accuracy': sk.metrics.accuracy_score(y_test, y_pred),
                'dictclassifiction_report' : sk.metrics.classification_report(y_test, y_pred, output_dict=True)
            })
        print('Run %s done.'%run.name)

        print('Average classification report over %i folds:',len(run.foldresults))
        mean_dict = am24utils.report_average([r['dictclassifiction_report'] for r in run.foldresults])
        mean_accuracy = sum([r['accuracy'] for r in run.foldresults])/len(run.foldresults)
        run.results = {'mean_classification_report_dict': mean_dict,'mean_classification_report_str': am24utils.classification_report_dict2print(mean_dict), 'mean_accuracy': mean_accuracy, 'number_folds': N_FOLDS}
        print(run.results['mean_classification_report_str'])


# TEST_WINDOWSIZES_VALUES = [4800]
# TEST_PERTURBATION_DISTRIBUTIONS = ['normal']
# TEST_PERTURBATION_MAXSAMPLES = [0]

# to_run = am24utils.get_run_list(winsizes = TEST_WINDOWSIZES_VALUES,
#                  pert_distributions = TEST_PERTURBATION_DISTRIBUTIONS,
#                  pert_maxsamples  = TEST_PERTURBATION_MAXSAMPLES)       
       
       
# Create the runs        
to_run_NN = am24utils.get_run_list()

for ridx,run in enumerate(to_run_NN):
    print(run.name)


run_taskB(to_run_NN, packedData=packedData)


# In[ ]:


import pickle, datetime

# resdir_path = os.path.join('results','task-B','date_%s'%(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# os.makedirs(resdir_path)

# resdir_path = os.path.join('results','task-B','date_2024-05-07_13-07-18')

bakfilename  = 'taskB_ResNet_results.pickle'

with open(os.path.join(resdir_path,bakfilename), 'wb') as f:
    pickle.dump(to_run_NN, f)
    

am24utils.plot_runs(to_run_NN, arg_metric='mean_f1')
plt.savefig(os.path.join(resdir_path,'accuracy_ResNet.png'))
plt.savefig(os.path.join(resdir_path,'accuracy_ResNet.pdf'))


# In[ ]:


# import pickle, datetime

# resdir_path = os.path.join('results','task-B','date_%s'%(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# os.makedirs(resdir_path)

# bakfilename  = 'taskB_results.pickle'

# with open(os.path.join(resdir_path,bakfilename), 'wb') as f:
#     pickle.dump(to_run, f)


# In[ ]:


# accuracies_toplot = [run.results['mean_accuracy'] for run in to_run]
# runnames_labels = [run.name for run in to_run]


# plt.figure(figsize=(1.5*len(accuracies_toplot),5))
# plt.bar(runnames_labels, accuracies_toplot)
# plt.xticks(rotation=45)
# plt.title('Accuracy')
# # plt.show()

# dirname = os.path.join('results','task-B','date_%s'%(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# os.makedirs(dirname)

# am24utils.plot_runs(to_run_NN)
# plt.savefig(os.path.join(resdir_path,'accuracy.png'))
# plt.savefig(os.path.join(resdir_path,'accuracy.pdf'))

