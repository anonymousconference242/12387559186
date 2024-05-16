DISABLE_TORCH = False

import os
import librosa
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
if not DISABLE_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    from torchvision import models
    import tqdm


# Parameters for mel spectrogram (DEFAULTS)
HOP_SIZE = 32
WIN_SIZE = 1024
N_FFT = 1024
N_MELS = 128

# Parameters for Run 
WINDOWSIZES_DEFAULT_VALUES = [4800, 2400, 1200, 600, 300] # [4800, 2400, 1200] #
PERTURBATION_DEFAULT_DISTRIBUTIONS = ['normal']
PERTURBATION_DEFAULT_MAXSAMPLES = [2400, 1200, 240, 0]

from_window_to_hop = lambda window_size: {4800: 256, 2400: 128, 1200: 64, 600: 32, 300:16}[window_size]
from_window_to_fft = lambda window_size: {4800: 1024, 2400: 512, 1200: 256, 600: 128, 300:128}[window_size]


class Run():
    name = ""
    window_size_samples = 0
    onset_perturbation_distribution = None
    onset_perturbation_max_samples = 0
    onset_perturbation_min_samples = 0
    #results = []
    #foldresults = []

    def __init__(self, 
                 name:str, 
                 window_size_samples:int, 
                 onset_perturbation_distribution:str, 
                 onset_perturbation_max_samples:int,
                 onset_perturbation_min_samples:int):
        self.name = str(name)
        self.window_size_samples = int(window_size_samples)
        self.onset_perturbation_max_samples = int(onset_perturbation_max_samples)
        self.onset_perturbation_distribution = str(onset_perturbation_distribution) if onset_perturbation_distribution is not None else None
        self.onset_perturbation_distribution = None if onset_perturbation_distribution == "None" else onset_perturbation_distribution
        self.onset_perturbation_min_samples = int(onset_perturbation_min_samples)
        self.results = []
        self.foldresults = []

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return '<'+self.name+'>'
    
    # Write method to sort, first according to window_size_samples, then perturbation distribution, then max samples
    def __lt__(self, other):
        if self.window_size_samples < other.window_size_samples:
            return True
        if self.window_size_samples > other.window_size_samples:
            return False
        
        if self.onset_perturbation_distribution == None and other.onset_perturbation_distribution == None:
            pass
        elif self.onset_perturbation_distribution == None:
            return True
        elif other.onset_perturbation_distribution == None:
            return False
        elif self.onset_perturbation_distribution == other.onset_perturbation_distribution:
            pass
        else:
            return self.onset_perturbation_distribution < other.onset_perturbation_distribution
        
        if self.onset_perturbation_max_samples < other.onset_perturbation_max_samples:
            return True
        return False

def compute_mean_f1(run: Run):
    fold_f1_mavgs = []
    for fold in run.foldresults:
        keys = list(fold['dictclassifiction_report'].keys())
        
        # Remove  'accuracy', 'macro avg', 'weighted avg'
        keys = [k for k in keys if k not in ['accuracy', 'macro avg', 'weighted avg']]

        class_f1s = [] 
        for curclass in keys:
            class_f1s.append(fold['dictclassifiction_report'][curclass]['f1-score'])
        fold_f1_mavgs.append(sum(class_f1s) / len(class_f1s))
    run_f1_mavg = sum(fold_f1_mavgs) / len(fold_f1_mavgs)
    return run_f1_mavg


def get_run_list(winsizes:list           = WINDOWSIZES_DEFAULT_VALUES,
                 pert_distributions:list = PERTURBATION_DEFAULT_DISTRIBUTIONS,
                 pert_maxsamples:list    = PERTURBATION_DEFAULT_MAXSAMPLES):
    runs = []
    for w in winsizes:
        w = int(w)
        for p in pert_distributions:
            for m in pert_maxsamples:
                m = int(m)
                if m==0:
                    p = 'None'
                # print('Appending ',w,p,m)
                runs.append(Run(name="Run_w%i_p%s"%(w,p) + ("_m%i"%m if m>0 else ""), 
                                 window_size_samples=w,
                                 onset_perturbation_distribution=p,
                                 onset_perturbation_max_samples=m,
                                 onset_perturbation_min_samples=0))
    return sorted(runs)


def plot_runs(to_run, arg_metric = 'mean_accuracy', arg_plottype = 'bar', color='tab:blue', title=None, groundTruthBar_color = 'w',boldline=1.1, groundtruth_linestyle='-'):
    if arg_metric == 'mean_accuracy':
        arg_metric_LABEL = 'Mean Accuracy'
    elif arg_metric == 'mean_f1':
        arg_metric_LABEL = 'Mean F1-Score'
    elif arg_metric == 'errors':
        arg_metric_LABEL = 'Absolute Pitch Error (MIDI)'


    if arg_metric == 'mean_f1':
        # In this case we explicitely compute the mean F1 score if not already a key of the results dict
        for r in to_run:
            if 'mean_f1' not in r.results:
                r.results['mean_f1'] = compute_mean_f1(r)

    bar_width = 0.7
    bar_space = 0.4
    onset_perturbation_distribution_space = 0.3
    window_space = 0.5

    x_positions = [0.]
    prev = to_run[0]
    cur_pos = 0.0
    windowgroup_braket_startend = [0.]
    windowgroup_sizes = [to_run[0].window_size_samples]
    for i, r in enumerate(to_run[1:]):
        if r.window_size_samples != prev.window_size_samples:
            windowgroup_braket_startend.append(cur_pos)
        cur_pos += bar_width + bar_space
        if r.onset_perturbation_distribution != prev.onset_perturbation_distribution:
            cur_pos += onset_perturbation_distribution_space
        if r.window_size_samples != prev.window_size_samples:
            cur_pos += window_space
            windowgroup_braket_startend.append(cur_pos)
            windowgroup_sizes.append(r.window_size_samples)
        x_positions.append(cur_pos)
        prev = r
    windowgroup_braket_startend.append(x_positions[-1])

    assert len(x_positions) == len(to_run)

    matplotlibpalette = {
    'tab:blue' : '#1f77b4',
    'tab:orange' : '#ff7f0e',
    'tab:green' : '#2ca02c',
    'tab:red' : '#d62728',
    'tab:purple' : '#9467bd',
    'tab:brown' : '#8c564b',
    'tab:pink' : '#e377c2',
    'tab:gray' : '#7f7f7f',
    'tab:olive' : '#bcbd22',
    'tab:cyan' : '#17becf'}

    # Now we plot the bars
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(to_run):
        curlinewidth = boldline if r.onset_perturbation_max_samples == 0 else 1
        curlinestyle = groundtruth_linestyle if r.onset_perturbation_max_samples == 0 else '-'
        if arg_plottype == 'bar':
            curcolor = matplotlibpalette[color] if r.onset_perturbation_max_samples != 0 else groundTruthBar_color
            ax.bar(x_positions[i], r.results[arg_metric], bar_width, label=r.name, color=curcolor, edgecolor='k', linewidth=curlinewidth)
        elif arg_plottype == 'box':
            ax.boxplot(r.results[arg_metric], positions=[x_positions[i]], widths=bar_width, labels=[r.name],whiskerprops = dict(linestyle=curlinestyle,linewidth=curlinewidth, color='black'),boxprops= dict(linestyle=curlinestyle,linewidth=curlinewidth, color='black'),capprops= dict(linewidth=curlinewidth, color='black'))
        else:
            raise ValueError('Unknown plot type')
        
    # If box, add enlargeXlim% of range to the max xlim and remove 10% of range from the min xlim
    enlargeXlim = 0.03
    if arg_plottype == 'box':
        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        ax.set_xlim(ax.get_xlim()[0] - enlargeXlim*xrange, ax.get_xlim()[1] + enlargeXlim*xrange)

    ax.set_xticks(x_positions)
    getRunMaxMs = lambda r: r.onset_perturbation_max_samples//48

    pltBold = lambda y: r"$\bf{{{x}}}$".format(x=y)
    getBoldRunMaxMs = lambda r: pltBold(getRunMaxMs(r)) if getRunMaxMs(r) == 0 else getRunMaxMs(r)

    ax.set_xticklabels([getBoldRunMaxMs(r) for r in to_run]) #, rotation=45) #, ha='right')

    if arg_plottype == 'box':
        polishres = lambda x: max(x)
        min_ylim = -1
    else:
        polishres = lambda x: x
        min_ylim = 0


    ax.set_ylim(min_ylim, max([polishres(r.results[arg_metric])*1.1 for r in to_run]) + 0.1)

    # draw vertical red lines to separate different window groups
    grouped_bracketends = []
    for i in range(0, len(windowgroup_braket_startend), 2):
        grouped_bracketends.append((windowgroup_braket_startend[i],windowgroup_braket_startend[i+1]))
    # print(grouped_bracketends)
        
    # Print horizontal lines just below xtick lables, outside the plot area, each starting and ending at the vertical red lines
    def draw_line_at_y(y,ax,color='gray',linewidth=1):
        lineout = ax.plot(list(ax.get_xlim()),[y,y],c=color,linewidth=linewidth)
        lineout[0].set_clip_on(False)

    def draw_short_hor_line(xstart,xend,y,ax,color='gray',linewidth=1):
        lineout = ax.plot([xstart,xend],[y,y],c=color,linewidth=linewidth)
        lineout[0].set_clip_on(False)

    def draw_short_hor_bracket(xstart,xend,y,ax,color='gray',linewidth=1, ends_height = 0.1):
        lineout = ax.plot([xstart,xend],[y,y],c=color,linewidth=linewidth)
        lineout[0].set_clip_on(False)
        # Draw the ends
        lineout = ax.plot([xstart,xstart],[y,y+ends_height],c=color,linewidth=linewidth)
        lineout[0].set_clip_on(False)
        lineout = ax.plot([xend,xend],[y,y+ends_height],c=color,linewidth=linewidth)
        lineout[0].set_clip_on(False)

    horline_pos = -0.075
    textpos_pos = -0.085

    if ax.get_ylim()[0] < 0:
        horline_pos -= 0.02
        textpos_pos -= 0.02

    # print('horline_pos before',horline_pos)
    ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    horline_pos *= ylim_range
    textpos_pos *= ylim_range
    # print('horline_pos after',horline_pos)
    for se_idx, (start, end) in enumerate(grouped_bracketends):
        astart = start - bar_width/2
        aend = end + bar_width/2
        draw_short_hor_bracket(astart, aend,horline_pos,ax, color='k', ends_height=0.01*ylim_range)
        # Print Window size in MS below
        
        curwsms = windowgroup_sizes[se_idx]/48
        curwsms = int(curwsms) if curwsms == int(curwsms) else curwsms
        curtext = f'W={curwsms}ms'
        ax.text((astart+aend)/2, textpos_pos, curtext, ha='center', va='top')

    ax.set_xlabel('Onset Perturbation Max [ms] and Analysis Window Size (W) [ms]')
    # move xlabel below
    ax.xaxis.set_label_coords(0.5, -0.16)

    if arg_plottype == 'box':
        # put y ticks in 12 intervals e.g. 0,12,24,36,48,60,72,84,96,108,120
        next12multiple = lambda x: 12 * (x//12 + 1)
        yticks = np.arange(0,next12multiple(ax.get_ylim()[1]),12).astype(int)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks])


    ax.set_ylabel(arg_metric_LABEL)
    if title is not None:
        ax.set_title(title)

    return fig, ax

def filter_files_db(dataset):
    files_to_keep = dataset['files_df'].index.tolist()
    files_to_keep = [file for file in files_to_keep if 'impro' not in file]        # Remove those that contain impro
    
    #Take from the dataset['noteLabels_df] only the rows that have the filenames in the second value of the multiindex
    filtered_notes_df = dataset['noteLabels_df'].loc[dataset['noteLabels_df'].index.get_level_values(1).isin(files_to_keep)]

    filtered_filenames = filtered_notes_df.index.get_level_values(1).tolist()
    filtered_filenames = list(sorted(set(filtered_filenames)))
    filtered_files_df = dataset['files_df'].loc[filtered_filenames]

    return filtered_notes_df,filtered_files_df

# Utils to average classification report over folds
def report_average(reports:list):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict

def classification_report_dict2print(report):
    ret = ""
    classes = list(report.keys())[0:-3]
    summary_metrics = list(report.keys())[-3:]
    longest_1st_column_name = max([len(key) for key in report.keys()])
    ret = ' ' * longest_1st_column_name
    ret += '  precision    recall  f1-score   support\n\n'

    METRIC_DECIMAL_DIGITS = 4
    metric_digits = METRIC_DECIMAL_DIGITS + 2 # add 0 and dot

    header_spacing = 1
    metrics = list(report[classes[0]].keys())
    longest_1st_row_name = max([len(key) for key in report[classes[0]].keys()]) + header_spacing

    for classname in classes:
        ret += (' '*(longest_1st_column_name-len(classname))) + classname + ' '
        for metric in metrics:
            if metric != "support":
                ret += (' '*(longest_1st_row_name-metric_digits))
                ret += "%.4f" % round(report[classname][metric],METRIC_DECIMAL_DIGITS)
            else:
                current_support_digits = len(str(int(report[classname][metric])))
                ret += (' '*(longest_1st_row_name-current_support_digits))
                ret += "%d" % round(report[classname][metric],0)
        ret += '\n'
    ret += '\n'

    # Accuracy
    ret += (' '*(longest_1st_column_name-len(summary_metrics[0]))) + summary_metrics[0] + ' '
    ret += 2* (' '*longest_1st_row_name)
    ret += (' '*(longest_1st_row_name-metric_digits))
    ret += "%.4f" % round(report["accuracy"],METRIC_DECIMAL_DIGITS)
    current_support_digits = len(str(int(report[summary_metrics[-1]]['support'])))
    ret += (' '*(longest_1st_row_name-current_support_digits))
    ret += "%d" % round(report[summary_metrics[-1]]['support'],0)
    ret += '\n'
  
  
    for classname in summary_metrics[1:]:
        ret += (' '*(longest_1st_column_name-len(classname))) + classname + ' '
        for metric in metrics:
            if metric != "support":
                ret += (' '*(longest_1st_row_name-metric_digits))
                ret += "%.4f" % round(report[classname][metric],METRIC_DECIMAL_DIGITS)
            else:
                current_support_digits = len(str(int(report[classname][metric])))
                ret += (' '*(longest_1st_row_name-current_support_digits))
                ret += "%d" % round(report[classname][metric],0)
        ret += '\n'
    ret += '\n'

    return ret


def apply_onset_perturbation(onset_list, onset_perturbation_distribution, onset_perturbation_max_samples, onset_perturbation_min_samples):
    if onset_perturbation_distribution is None:
        return onset_list
    
    if onset_perturbation_distribution == "uniform":
        perturbation = np.random.randint(onset_perturbation_min_samples, onset_perturbation_max_samples, len(onset_list))
        perturbation_sign = np.random.choice([-1,1], len(onset_list))
        res =  onset_list + (perturbation * perturbation_sign)
        return [int(r) for r in res]
    
    if onset_perturbation_distribution == "ODpdf":
        # Apply a normal between onset_perturbation_min_samples and onset_perturbation_max_samples
        # TODO: fix with correct distribution
        perturbation = np.random.normal(loc=0, scale=0.3, size=len(onset_list))
        perturbation = np.clip(perturbation, -1, 1)
        perturbation*=onset_perturbation_max_samples
        res = onset_list + perturbation
        return [int(r) for r in res]    
    
    if onset_perturbation_distribution == "normal":
        # Normal distribution, scaled with a third of the max perturbation, so that 99.7% of the perturbations are within the min and max
        perturbation = np.random.normal(loc=0, scale=onset_perturbation_max_samples/3, size=len(onset_list))
        res = onset_list + perturbation
        return [int(r) for r in res] 

    
    raise ValueError("Unknown onset perturbation distribution %s"%onset_perturbation_distribution)

# Utils for getting features from audio files
def get_Xy(filename, onsetlist, y, window_size_samples, features=["mfcc", "spectral_centroid", "spectral_bandwidth", "rms"]): # ["log-mel","mfcc", "spectral_centroid", "spectral_bandwidth", "rms"]
    assert os.path.exists(filename), "File %s does not exist"%filename
    sig, sr = librosa.load(filename, sr=None)
    Xloc = []
    yloc = []
    for i, onset in enumerate(onsetlist):
        if onset < 0:
            continue # Negative onset
        if onset+window_size_samples > len(sig):
            continue
        assert type(onset) == int, "Onset is not an integer (%s)"%str(onset)
        assert onset >= 0, "Onset is negative (%i)"%onset
        assert onset+window_size_samples < len(sig), "Onset is too large (%i > %i)"%(onset+window_size_samples, len(sig))
        assert type(window_size_samples) == int, "Window size is not an integer (%s)"%str(window_size_samples)

        sigSlice = sig[onset:onset+window_size_samples]


        cur_features = []
        
        
        # Compute log-mel Spectrogram
        if "log-mel" in features:
            ## Compute log-mel
            mel = librosa.feature.melspectrogram(y = sigSlice, sr = sr, n_fft = from_window_to_fft(window_size_samples), hop_length = from_window_to_hop(window_size_samples), n_mels = N_MELS, fmax = sr//2)
            logmel = np.log(mel + 1e-9)
            # logmel = np.log(spec + 1e-9) 
            delta = librosa.feature.delta(logmel)
            delta_delta = librosa.feature.delta(logmel, order=2)
            
            spec = np.concatenate([logmel[np.newaxis,:], delta[np.newaxis,:], delta_delta[np.newaxis,:]], axis=0)
            
            cur_features.append(spec)
            
        ## Compute MFCC
        if "mfcc" in features:
            mfcc = librosa.feature.mfcc(y=sigSlice, sr=sr, n_mfcc=13, center=False, win_length=window_size_samples//2, n_fft=window_size_samples)
            # mfcc = np.mean(mfcc, axis=1)
            mfcc = mfcc.flatten()
            cur_features.extend(mfcc) # Old version
            # cur_features.append(mfcc)
            
        ## Compute Spectral Centroid
        if "spectral_centroid" in features:
            spectral_centroid = librosa.feature.spectral_centroid(y=sigSlice, sr=sr, n_fft=window_size_samples)
            spectral_centroid = np.mean(spectral_centroid)
            cur_features.append(spectral_centroid)

        ## Compute Spectral Bandwidth
        if "spectral_bandwidth" in features:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=sigSlice, sr=sr, n_fft=window_size_samples)
            spectral_bandwidth = np.mean(spectral_bandwidth)
            cur_features.append(spectral_bandwidth)

        ## Compute RMS
        if "rms" in features:
            rms = librosa.feature.rms(y=sigSlice, frame_length=window_size_samples)
            rms = np.mean(rms)
            cur_features.append(rms)

        Xloc.append(cur_features)
        yloc.append(y[i])
    assert len(Xloc) == len(yloc), "Xloc and yloc have different lengths (%i != %i)"%(len(Xloc), len(yloc))
    return Xloc, yloc

if not DISABLE_TORCH:
    # Utils for loading and     saving models during KFold training
    def save_or_reset_weights(model, save_path):
        if os.path.exists(save_path):
            state_dict = torch.load(save_path)
            model.load_state_dict(state_dict)
        else:
            torch.save(model.state_dict(), save_path)
        return model

    # SpecAugment class for data augmentation
    from torchaudio.transforms import TimeMasking, FrequencyMasking, TimeStretch
    class SpecAugment:
        def __init__(self, time_stretch_n_fft=N_FFT,time_stretch_hop_length=HOP_SIZE, freq_masking=27, time_masking=100):
            super(SpecAugment, self).__init__()
            self.time_stretch = TimeStretch(n_freq=time_stretch_n_fft, hop_length=time_stretch_hop_length)
            self.freq_masking = FrequencyMasking(freq_mask_param=freq_masking)
            self.time_masking = TimeMasking(time_mask_param=time_masking, p=0.7)
            
        def __call__(self, spec):
            spec = self.time_warp(spec)
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
            return spec

        def time_warp(self, spec):
            overriding_rate = np.random.uniform(0.9, 1.2)
            # Time stretch
            spec = self.time_stretch(spec, overriding_rate)
            return spec

        def freq_mask(self, spec):
            # Frequency masking
            spec = self.freq_masking(spec)
            return spec

        def time_mask(self, spec):
            # Time masking
            spec = self.time_masking(spec)
            return spec

    # AudioConvNet as a base class for AudioResNet, AudioSqueezeNet, and other audio classification models
    class AudioConvNet(nn.Module):
        def __init__(self):
            super(AudioConvNet, self).__init__()
            
        def fit(self, X, y, epochs=5, batch_size=128, lr=0.001, device=None):
            
            if device is None:
                device = next(self.parameters()).device 
                
            self.train()
            optimizer = optim.Adam(self.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            assert X.shape[1] == 3, f"Dataset should be in the format (N, channels, freqs, time_steps) with 3 channels (mel, delta, delta-delta). Obtained shape: {X.shape}"
            dataset = data.TensorDataset(torch.tensor(X).float().to(device), torch.tensor(y).long().to(device))
            dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                # For each epoch print loss and accuracy
                running_loss = 0.0
                correct = 0
                total = 0
                for i, (inputs, labels) in tqdm.tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{epochs}", leave=True, total=len(dataloader)):
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print('Epoch: [%d] --- loss: %.3f accuracy: %.3f' % (epoch + 1, running_loss / total, correct / total))
                
            # Free memory
            del dataset, dataloader
            torch.cuda.empty_cache()
            # Garbage collection
            import gc
            gc.collect()
            
            return self
        
        
        def test(self, X, y, batch_size=128, device=None):
                
                if device is None:
                    device = next(self.parameters()).device 
                    
                self.eval()
                criterion = nn.CrossEntropyLoss()
                
                assert X.shape[1] == 3, f"Dataset should be in the format (N, channels, freqs, time_steps) with 3 channels (mel, delta, delta-delta). Obtained shape: {X.shape}"
                dataset = data.TensorDataset(torch.tensor(X).float().to(device), torch.tensor(y).long().to(device))
                dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                y_pred = []
                running_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for i, (inputs, labels) in tqdm.tqdm(enumerate(dataloader), desc="Testing", leave=True, total=len(dataloader)):
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        y_pred.extend(predicted.cpu().numpy())
                        
                print('Test --- loss: %.3f accuracy: %.3f' % (running_loss / total, correct / total))
                
                # Free memory
                del dataset, dataloader
                torch.cuda.empty_cache()
                # Garbage collection
                import gc
                gc.collect()
                
                return correct / total, y_pred
                
        def predict(self, X):
            self.eval()
            with torch.no_grad():
                y_pred = self(torch.tensor(X).float().to(next(self.parameters()).device))
                return torch.argmax(y_pred, dim=1).cpu().numpy()
            
        def score(self, X, y):
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred) # sklearn.metrics.accuracy_score

    # ResNet model for audio classification
    class AudioResNet(AudioConvNet):
        def __init__(self, num_classes, resnet_version=18, pretrained=True, fine_tuning=True):
            super(AudioResNet, self).__init__()
            if resnet_version == 18:
                resnet = models.resnet18(pretrained=pretrained)
            elif resnet_version == 34:
                resnet = models.resnet34(pretrained=pretrained)
            elif resnet_version == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif resnet_version == 101:
                resnet = models.resnet101(pretrained=pretrained)
            elif resnet_version == 152:
                resnet = models.resnet152(pretrained=pretrained)
            else:
                raise ValueError("Unknown ResNet version %i. Available versions are: 18, 34, 50, 101, 152"%resnet_version)
            
            if not fine_tuning:
                for param in resnet.parameters():
                    param.requires_grad = False
                    
            self.features = nn.Sequential(*list(resnet.children())[1:-1])  # Remove the first and last layer
            self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Modify first layer for 1-channel input
            self.fc = nn.Linear(resnet.fc.in_features, num_classes) # Add a new fully connected layer for the new number of classes

        def forward(self, x):
            x = self.conv1(x)  # Apply the modified first layer
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        
    # SqueezeNet model for audio classification
    class AudioSqueezeNet(AudioConvNet):
        def __init__(self, num_classes, version=1.0, pretrained=True, fine_tuning=True):
            super(AudioSqueezeNet, self).__init__()
            if version == 1.0:
                squeezenet = models.squeezenet1_0(pretrained=pretrained)
            elif version == 1.1:
                squeezenet = models.squeezenet1_1(pretrained=pretrained)
            else:
                raise ValueError("Unknown SqueezeNet version %f. Version must be 1.0 or 1.1"%version)
            
            if not fine_tuning:
                for param in squeezenet.parameters():
                    param.requires_grad = False
                    
            self.features = squeezenet.features
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return torch.flatten(x, 1)
        


    if __name__ == "__main__":
        # from torchsummary import summary
        
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # # Create dummy data
        # X = np.random.rand(10, 3, 128, 21)
        # y = np.random.randint(0, 2, 10)
        
        # # Test ResNet18 model
        # resnet = AudioResNet(2).to(device)
        # resnet.fit(X, y, device=device)
        
        # print("RESNET")
        # print("Labels:\n", y)
        # print("Predictions:\n", resnet.predict(X))
        # print("Accuracy Score:", resnet.score(X, y))
        
        # # Test SqueezeNet model
        # squeezenet = AudioSqueezeNet(2).to(device)
        # squeezenet.fit(X, y, device=device)
        # print("SqueezeNet")
        # print("Labels:\n", y)
        # print("Predictions:\n", resnet.predict(X))
        # print("Accuracy Score:", resnet.score(X, y))

        # # ResNet18
        # resnet = AudioResNet(2).to("cuda")
        # summary(resnet, (3, 128, 21))
        
        # # SqueezeNet
        # squeezenet = AudioSqueezeNet(2).to("cuda")
        # summary(squeezenet, (3, 128, 21))
        
        

        print('\n'.join(str(e) for e in sorted(get_run_list())))
