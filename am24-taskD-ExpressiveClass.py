#!/usr/bin/env python
# coding: utf-8

# # Task D: Expressive technique classification
# ---

# In[ ]:


import dataset.aGPTset.ExpressiveGuitarTechniquesDataset as agptset
import os
import librosa
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import sklearn as sk
import imblearn

import am24utils
from am24utils import Run


dataset = agptset.import_db()

DOTEST = False
VERBOSE = False
DB_PATH = 'dataset/aGPTset'
printVerbose = lambda x: print(x) if VERBOSE else None
printVerboseLevel1 = lambda x,vl: print(x) if vl > 1 else None
printVerboseLevel2 = lambda x,vl: print(x) if vl > 2 else None
printVerboseLevel3 = lambda x,vl: print(x) if vl > 3 else None
MULTIPROCESSING = False

if MULTIPROCESSING:
    import multiprocessing as mp


# In[ ]:


Percussive_TECHNIQUES=['Kick', 'Snare-A', 'Tom', 'Snare-B']
Pitched_TECHNIQUES=['Natural Harmonics', 'Palm Mute', 'Pick Near Bridge', 'Pick Over the Soundhole', 'Bending', 'Hammer-on', 'Staccato', 'Vibrato']
TECHNIQUES = Percussive_TECHNIQUES + Pitched_TECHNIQUES
print(len(TECHNIQUES))
techniques_str_to_int = lambda x: {TECHNIQUES[0]:0, TECHNIQUES[1]:1, TECHNIQUES[2]:2, TECHNIQUES[3]:3, TECHNIQUES[4]:4, TECHNIQUES[5]:5, TECHNIQUES[6]:6, TECHNIQUES[7]:7, TECHNIQUES[8]:8, TECHNIQUES[9]:9,TECHNIQUES[10]:10, TECHNIQUES[11]:11}[x]
techniques_int_to_str = lambda x: TECHNIQUES[x] if (x < 3 and x>=0 and type(x)==int) else None


# In[ ]:


print("Filtering the Dataset...")

# Filter the db to keep only pitched notes
filtered_notes_db, filtered_files_db = am24utils.filter_files_db(dataset)
#make sure that no audio_file_path contains "impro"
assert filtered_notes_db.index.get_level_values(1).str.contains('impro').sum() == 0, "Some audio_file_path contain 'impro' (%i)"%(filtered_notes_db.index.get_level_values(1).str.contains('impro').sum())
print("Done (%i notes in the filtered db)."%len(filtered_notes_db))


# In[ ]:


def get_onsetlist_filename_techniques(filtered_notes_db:pd.DataFrame, filtered_files_db:pd.DataFrame):
    onsetlist = []
    filenames = []
    players = []
    techniques = []
    for file in filtered_notes_db.index.get_level_values(1).unique():
        if file in filtered_files_db.index:
            afp = filtered_files_db[filtered_files_db.index == file].full_audiofile_path.values
            assert len(afp) == 1, "More than one audio file path for file %s"%file
            filenames.append(afp[0])
            cur_onset_list = filtered_notes_db.loc[filtered_notes_db.index.get_level_values(1) == file].onset_label_samples.values
            cur_technique = filtered_notes_db.loc[filtered_notes_db.index.get_level_values(1) == file].expressive_technique_id.values
            assert len(cur_onset_list) == len(cur_technique), "Onset list and cur_technique have different lengths"
            # print('%s has %i onsets and %i cur_techniques'%(file,len(cur_onset_list),len(cur_techique)))
            cur_onset_list = [int(x) for x in cur_onset_list]
            onsetlist.append(cur_onset_list)
            for x in cur_technique:
                assert int(x) in range(12), "Technique %i not in range 0-11"%int(x)
            cur_technique = [int(x) for x in cur_technique]
            techniques.append(cur_technique)
            cur_player = filtered_files_db[filtered_files_db.index == file].player_id.values
            assert len(cur_player) == 1, "More than one player for file %s"%file
            cur_player = int(cur_player[0])
            players.append(cur_player)
        else:
            raise ValueError("File %s not found in the files db"%file)
        
    return onsetlist, filenames, techniques, players
        

onsetlist,filenames,techniques,playerlist  = get_onsetlist_filename_techniques(filtered_notes_db,filtered_files_db)
assert len(onsetlist) == len(filenames) == len(techniques) == len(playerlist), "Different lengths for onsetlist, filenames, techniques and playerlist"
packedData = (onsetlist,filenames,techniques,playerlist)


# In[ ]:


def load_and_compute_features_for_file(cur_filename, 
                                       cur_onsetlist, 
                                       cur_techniqueslist, 
                                       cur_player,
                                       window_size_samples,
                                       onset_perturbation_distribution,
                                       onset_perturbation_max_samples, 
                                       onset_perturbation_min_samples):
    # print('Processing file %s'%cur_filename)
    assert len(cur_onsetlist) == len(cur_techniqueslist), "onsetlist and cur_techniqueslist have different lengths (%i != %i)"%(len(cur_onsetlist), len(cur_techniqueslist))
    if onset_perturbation_distribution is not None:
        # print('Applying onset perturbation to file %s'%cur_filename)
        cur_onsetlist = am24utils.apply_onset_perturbation(cur_onsetlist, onset_perturbation_distribution, onset_perturbation_max_samples, onset_perturbation_min_samples)

    # print('Computing features for file %s'%cur_filename)
    Xfn, yfn = am24utils.get_Xy(cur_filename, cur_onsetlist, cur_techniqueslist, window_size_samples)
    
    assert len(Xfn) == len(yfn), "Xfn and yfn have different lengths (%i != %i)"%(len(Xfn), len(yfn))

    print('.',end='',flush=True)
    playerlist = [cur_player]*len(Xfn)
    return Xfn, yfn,playerlist

def run_taskD(runs, packedData, classifier='KNN'):
    onsetlist_list,filenames_list,techniques_list, player_list = packedData
    assert len(onsetlist_list) == len(filenames_list) == len(techniques_list) == len(player_list), "Different lengths for onsetlist_list, filenames_list, techinques_list and player_list"
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
        #                                                              techniques_list[i], 
        #                                                              player_list[i])
        #     X.extend(Xfn)
        #     y.extend(yfn)
        #     group.extend(playervec) # Add the player id to the group list, repeated for each onset
        '''

        X,y,group = [],[],[]
        if MULTIPROCESSING:
            # replace previous commented block with parallel processing
            pool = mp.Pool(mp.cpu_count())
            # results = [pool.apply_async(load_and_compute_features_for_file, args=(os.path.join(DB_PATH,filenames_list[i]), onsetlist_list[i], techniques_list[i], player_list[i])) for i in range(len(onsetlist_list))]
            results = [pool.apply_async(load_and_compute_features_for_file, 
                        args=(os.path.join(DB_PATH,filenames_list[i]), 
                            onsetlist_list[i], 
                            techniques_list[i], 
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
                                                                        techniques_list[i], 
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
        run.results = []
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y, group)):
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            group_train, group_test = np.array(group)[train_index], np.array(group)[test_index]
            print('Fold %i/%i'%(fold_idx+1,N_FOLDS))
            print('Train:', len(X_train), len(y_train),'Groups: ', sorted(list(set(group_train))))
            print('Test:', len(X_test), len(y_test),'Groups: ', sorted(list(set(group_test))))
            
            # For each test set, print the percentage of each technique (0,1,2,3,4,5,6,7,8,9,10,11)
            eachdyn = []
            for technique in range(len(TECHNIQUES[:8])):
                eachdyn.append(100*sum(y_test==technique)/len(y_test))
                print('Techique %s in test set: %.2f%%'%(TECHNIQUES[technique], eachdyn[-1]))
                

            # If any of the techniques percentage is more than 10%than the equal split, enable smote
            if any([abs(x-100/len(TECHNIQUES[:8]))>10 for x in eachdyn]):
                # Use SMOTE to balance the classes
                from imblearn.over_sampling import SMOTE
                smote = SMOTE()
                print('Balancing classes with SMOTE ...')
                X_train, y_train = smote.fit_resample(X_train, y_train)
        
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
            # elif classifier.upper()=='RESNET':
            #     import torch
            #     from am24utils import AudioResNet
            #     # Train ResNet for audio classification
            #     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            #     resnet = AudioResNet(num_classes=3).to(device)
            #     print('Training', classifier, '...')
            #     resnet.fit(X_train, y_train)
                
            #     # Test
            #     y_pred = resnet.predict(X_test)
                
            # elif classifier.upper()=='SQUEEZENET':
            #     import torch
            #     from am24utils import AudioSqueezeNet
            #     # Train ResNet for audio classification
            #     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            #     squeezenet = AudioSqueezeNet(num_classes=3).to(device)
            #     print('Training', classifier, '...')
            #     resnet.fit(X_train, y_train)
                
            #     # Test
            #     y_pred = resnet.predict(X_test)
                
            
            # # Metrics
            classification_report = sk.metrics.classification_report(y_test, y_pred, labels=[techniques_str_to_int(x) for x in TECHNIQUES[:8]],target_names=TECHNIQUES[:8])
            print('-Classification FOLD %i/%i-'%(fold_idx+1,N_FOLDS))
            print('\t\t\t'+classification_report.replace('\n','\n\t\t\t'))
            print('\n\n')

            confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred, labels=[techniques_str_to_int(x) for x in TECHNIQUES[:8]] )
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
       


# TEST_WINDOWSIZES_VALUES = [2400]
# TEST_PERTURBATION_DISTRIBUTIONS = ['normal']
# TEST_PERTURBATION_MAXSAMPLES = [2400, 0]

# to_run = am24utils.get_run_list(winsizes = TEST_WINDOWSIZES_VALUES,
#                  pert_distributions = TEST_PERTURBATION_DISTRIBUTIONS,
#                  pert_maxsamples  = TEST_PERTURBATION_MAXSAMPLES)       
       
       
# Create the runs        
to_run = am24utils.get_run_list()

for ridx,run in enumerate(to_run):
    print(run.name)

run_taskD(to_run, packedData=packedData)


# In[ ]:


import pickle, datetime


resdir_path = os.path.join('results','task-D','date_%s'%(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
os.makedirs(resdir_path)

bakfilename  = 'taskD_KNN_results.pickle'

with open(os.path.join(resdir_path,bakfilename), 'wb') as f:
    pickle.dump(to_run, f)
    
am24utils.plot_runs(to_run, 'mean_f1')
plt.savefig(os.path.join(resdir_path,'accuracy_KNN.png'))
plt.savefig(os.path.join(resdir_path,'accuracy_KNN.pdf'))


# In[ ]:


# TEST WITH RESNET AND SQUEEZENET

def load_and_compute_features_for_file(cur_filename, 
                                       cur_onsetlist, 
                                       cur_techniqueslist, 
                                       cur_player,
                                       window_size_samples,
                                       onset_perturbation_distribution,
                                       onset_perturbation_max_samples, 
                                       onset_perturbation_min_samples):
    # print('Processing file %s'%cur_filename)
    assert len(cur_onsetlist) == len(cur_techniqueslist), "onsetlist and cur_techniqueslist have different lengths (%i != %i)"%(len(cur_onsetlist), len(cur_techniqueslist))
    if onset_perturbation_distribution is not None:
        # print('Applying onset perturbation to file %s'%cur_filename)
        cur_onsetlist = am24utils.apply_onset_perturbation(cur_onsetlist, onset_perturbation_distribution, onset_perturbation_max_samples, onset_perturbation_min_samples)

    # print('Computing features for file %s'%cur_filename)
    Xfn, yfn = am24utils.get_Xy(cur_filename, cur_onsetlist, cur_techniqueslist, window_size_samples, features=["log-mel","mfcc"])
    
    assert len(Xfn) == len(yfn), "Xfn and yfn have different lengths (%i != %i)"%(len(Xfn), len(yfn))

    print('.',end='',flush=True)
    playerlist = [cur_player]*len(Xfn)
    return Xfn, yfn,playerlist

def run_taskD(runs, packedData, classifier='RESNET'):
    onsetlist_list,filenames_list,techniques_list, player_list = packedData
    assert len(onsetlist_list) == len(filenames_list) == len(techniques_list) == len(player_list), "Different lengths for onsetlist_list, filenames_list, techniques_list and player_list"
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
        #                                                              techniques_list[i], 
        #                                                              player_list[i])
        #     X.extend(Xfn)
        #     y.extend(yfn)
        #     group.extend(playervec) # Add the player id to the group list, repeated for each onset
        '''

        X,y,group = [],[],[]
        if MULTIPROCESSING:
            # replace previous commented block with parallel processing
            pool = mp.Pool(mp.cpu_count())
            # results = [pool.apply_async(load_and_compute_features_for_file, args=(os.path.join(DB_PATH,filenames_list[i]), onsetlist_list[i], techniques_list[i], player_list[i])) for i in range(len(onsetlist_list))]
            results = [pool.apply_async(load_and_compute_features_for_file, 
                        args=(os.path.join(DB_PATH,filenames_list[i]), 
                            onsetlist_list[i], 
                            techniques_list[i], 
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
                                                                        techniques_list[i], 
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
        # for fold_idx, (ti+1,' -rain_index, test_index) in enumerate(skf.split(X, y, group)):
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y, group)):
            X_features = [x[1:] for x in X]
            X_specs = [x[0] for x in X]
            X_train, X_test = np.array(X_features)[train_index], np.array(X_features)[test_index]
            S_train, S_test = np.array(X_specs)[train_index], np.array(X_specs)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            group_train, group_test = np.array(group)[train_index], np.array(group)[test_index]
            print('Fold %i/%i'%(fold_idx+1,N_FOLDS))
            print('Train:', len(X_train), len(y_train),'Groups: ', sorted(list(set(group_train))))
            print('Test:', len(X_test), len(y_test),'Groups: ', sorted(list(set(group_test))))
            
            # For each test set, print the percentage of each technique (0,1,2,3,4,5,6,7,8,9,10,11)
            eachdyn = []
            for technique in range(len(TECHNIQUES[:8])):
                test_dyn = 100*sum(y_test==technique)/len(y_test)
                train_dyn = 100*sum(y_train==technique)/len(y_train)
                eachdyn.append(train_dyn)
                print('Technique %s in test set: %.2f%%'%(TECHNIQUES[technique], test_dyn))
                

            # If any of the techniques percentage is more than 10% than the equal split, enable smote
            if any([abs(x-100/8)>10 for x in eachdyn]):
                
                for i in range(len(eachdyn)):
                    print(i+1,' - Technique %s in train set (Before SMOTE): %.2f%%'%(TECHNIQUES[i], eachdyn[i]))
                # Use SMOTE to balance the classes
                from imblearn.over_sampling import SMOTE
                smote = SMOTE()
                print('Balancing classes with SMOTE ...')
                shape_train = S_train.shape[1:]
                X_resampled, y_train = smote.fit_resample(S_train.reshape(-1, shape_train[0]*shape_train[1]*shape_train[2]), y_train)
                # Reshape the data
                S_train = X_resampled.reshape(-1, shape_train[0], shape_train[1], shape_train[2])
                
                eachdyn = []
                for technique in range(len(TECHNIQUES[:8])):
                    train_dyn = 100*sum(y_train==technique)/len(y_train)
                    eachdyn.append(train_dyn)
                    print('Technique %s in train set (AFTER SMOTE): %.2f%%'%(TECHNIQUES[i], train_dyn))


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
                resnet = AudioResNet(num_classes=8, fine_tuning=True).to(device)
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
                squeezenet = AudioSqueezeNet(num_classes=8).to(device)
                #am24utils.save_or_reset_weights(squeezenet, 'squeezenet_weights_starting_weights.pth')
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
            classification_report = sk.metrics.classification_report(y_test, y_pred, labels=[techniques_str_to_int(x) for x in TECHNIQUES[:8]],target_names=TECHNIQUES[:8])
            print('-Classification FOLD %i/%i-'%(fold_idx+1,N_FOLDS))
            print('\t\t\t'+classification_report.replace('\n','\n\t\t\t'))
            print('\n\n')

            confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred, labels=[techniques_str_to_int(x) for x in TECHNIQUES[:8]] )
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

run_taskD(to_run_NN, packedData=packedData)


# In[ ]:


# import pickle, datetime


# resdir_path = os.path.join('results','task-D','date_%s'%(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# os.makedirs(resdir_path)

bakfilename  = 'taskD_ResNet_results.pickle'

with open(os.path.join(resdir_path,bakfilename), 'wb') as f:
    pickle.dump(to_run_NN, f)
    
am24utils.plot_runs(to_run_NN, 'mean_f1')
plt.savefig(os.path.join(resdir_path,'accuracy_ResNet.png'))
plt.savefig(os.path.join(resdir_path,'accuracy_ResNet.pdf'))


# In[ ]:


# # Check memory usage
# import os
# import psutil
# process = psutil.Process(os.getpid())
# print(process.memory_info().rss / 1024**2)  # in bytes

# # Check GPU memory usage
# import torch
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())

# # Clear GPU memory
# torch.cuda.empty_cache()

# # Check GPU memory usage
# print(torch.cuda.memory_allocated())


# In[ ]:


# get_window_size_from_run = lambda run: run.window_size_samples
# get_onset_perturbation_distribution_from_run = lambda run: run.onset_perturbation_distribution
# get_onset_perturbation_max_samples_from_run = lambda run: run.onset_perturbation_max_samples


# all_sizes = sorted(list(set([get_window_size_from_run(run) for run in to_run])))
# runs_grouped_by_size = [[run for run in to_run if get_window_size_from_run(run) == size] for size in all_sizes]
# # print('Runs grouped by size:',[[ga.name for ga in g] for g in runs_grouped_by_size])

# runs_grouped_size_x_maxsamp = []
# for rungroup in runs_grouped_by_size:
#     all_maxsamp = sorted(list(set([get_onset_perturbation_max_samples_from_run(run) for run in rungroup])))
#     runs_grouped_size_x_maxsamp.append([[run for run in rungroup if get_onset_perturbation_max_samples_from_run(run) == maxsamp] for maxsamp in all_maxsamp])

# # print('Runs grouped by size and maxsamp:',[[[(ga.name,ga.window_size_samples,ga.onset_perturbation_max_samples) for ga in g] for g in gg] for gg in runs_grouped_size_x_maxsamp])

# # window_size_samples
# # onset_perturbation_max_samples


# fig, ax = plt.subplots(figsize=(20,6))
# barWidth = 0.25
# group1_spacing = 0.1
# group2_spacing = 0.05
# # runs_grouped_size_x_maxsamp

# xticks_positions = []
# xticks_text = []
# max_group2_width = max([max([len(e)*barWidth+(group2_spacing*(len(e)-1)) for e in ee]) for ee in runs_grouped_size_x_maxsamp])

# max_window_group_width = max([len(e)*barWidth+(group1_spacing*(len(e)-1)) for e in runs_grouped_size_x_maxsamp])
# all_rects = []
# for idx,run_wsize_group in enumerate(runs_grouped_size_x_maxsamp):
#     for idx2,run_maxsamp_group in enumerate(run_wsize_group):
#         accuracies_toplot = [run.results['mean_accuracy'] for run in run_maxsamp_group]
#         runnames_labels = [run.name for run in run_maxsamp_group]
#         r1 = np.arange(len(accuracies_toplot))
#         x_bars_pos = r1*barWidth + idx2*max_group2_width+ idx*max_window_group_width
#         xticklabs = [run.onset_perturbation_distribution if not run.onset_perturbation_distribution == None else '' for run in run_maxsamp_group]

#         color_from_probdist = lambda probdist: {'':'blue','normal':'orange','normal':'red'}[probdist]
#         label_from_probdist = lambda probdist: {'':'Dataset labels','normal':'Normal Pert.','normal':'Normal'}[probdist]

#         cur_rects = ax.bar(x_bars_pos, 
#                accuracies_toplot, 
#                width = barWidth,
#                label = [label_from_probdist(e) for e in xticklabs],
#                color = [color_from_probdist(e) for e in xticklabs])
#         all_rects.append(cur_rects)
#         # ax.bar(r1 + idx*barWidth, accuracies_toplot, width = barWidth, label = 'w%i_p%s'%(run_maxsamp_group[0].window_size_samples, run_maxsamp_group[0].onset_perturbation_distribution))
#         # xticks_positions.extend(x_bars_pos)
#         # xticks_text.extend(xticklabs)
#         # print(xticks_text,end='\n\n')

# ax.set_xticks([r + barWidth for r in range(len(r1))])
# # ax.set_xticks(xticks_positions)
# # ax.set_xticklabels(xticks_text)

# plt.xticks(rotation=45)
# plt.ylabel('Accuracy')

# ###Curstom legend

# handles = []
# labels = []
# for rects in all_rects:
#     handle, label = rects[0], rects[0].get_label()
#     handles.append(handle)
#     labels.append(label)

# # Creating the legend with combined entries for each group of bars
# ax.legend(list(set(handles)), list(set(labels)))



# plt.savefig(os.path.join(resdir_path,'accuracy.png'))
# plt.savefig(os.path.join(resdir_path,'accuracy.pdf'))


