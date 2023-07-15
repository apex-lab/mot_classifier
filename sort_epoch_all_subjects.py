import mne, json, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import math
import mne_connectivity

# dictionary for the events (maps tags to codes)
def event_dictionary(raw):
    event_dict = {}
    i = 1
    for event in raw.event_id:
        event_dict[event] = i
        i += 1
    return event_dict

# reverse dictionary (maps codes to tags)
def rev_event_dictionary(game_dict):
    rev_game_dict = {}
    for key in game_dict:    
        rev_game_dict[str(game_dict[key])] = key
    return rev_game_dict

# filter our data to keep only filters of interest
def filter(raw):
    # Apply a band-pass filter to keep frequencies of interest low = 0.1, high = 50 
    filtered_raw = raw.filter(l_freq=0.1, h_freq=50)
    return filtered_raw

# returns true if event name is either 'FLSH' or 'MVE0'
def valid_tag(event_name):
    if event_name == 'FLSH' or event_name == 'MVE0':
        return True
    return False

# finds the level associated with the din and returns it as a string with length = 2
def find_level(events_raw, rev_dict, index):
    level_found = False
    j = 0
    
    # find the level in which the tag occured
    while level_found == False:
        try:
            if 'FIX' in rev_dict[str(events_raw[index-j][2])]:
                level = '99'
                level_found = True
            # if it is a fixation tag then extract the level
            elif 'FX' in rev_dict[str(events_raw[index-j][2])]: 
                level = rev_dict[str(events_raw[index-j][2])]
                
                # remove F and X from the tag, leaving us with just the numbers as a string
                level = level.replace("F","")
                level = level.replace("X", "")
                
                # indicate that we found the level
                level_found = True
                
                # pad with zeroes so that it will be a string of length = 2
                while len(level) < 2:
                    level = '0' + level
                    
            # index by one so that we can look back one further spot in the specified window
            j += 1 

        # if we reach the beginning of the list and cannot find a fixation tag
        except:
            # if error raised then print this
            print("No tags found! Fixation")
            print("tag: " + rev_dict[str(events_raw[index][2])])
            level_found == True
            level = -1
    return level    
       
# find the result of the trial associated with the din and returns it as a string of length = 2 
def find_result(events_raw, rev_dict, index):
    j = 0
    result_found = False
        
    # find the result of the level in which the tag occured
    while result_found == False: 
        try:
            # if we find a miss
            if 'MS' in rev_dict[str(events_raw[index+j][2])]:
                result = rev_dict[str(events_raw[index+j][2])] 
                result = result[2:]
                if result[0] == '0':
                    result[0] = '8'
                result_found = True
            elif 'MIS' in rev_dict[str(events_raw[index+j][2])]:
                result = '88'
                result_found = True
            # if we find a correct trial
            elif 'CRCT' == rev_dict[str(events_raw[index+j][2])]: 
                result = str(99) # 99 will indicate a correct trial
                result_found = True
            elif 'COR' in rev_dict[str(events_raw[index+j][2])]:
                result = str(99)
                result_found = True            
            j += 1
        except:
            result_found = True 
            print("No tags found! Result") 
            print("tag: " + rev_dict[str(events_raw[index][2])])
            result = -1
            result_found = True
    return result 

# function to find the index of the event where the real trials begin
def find_start_index(events_raw, rev_dict):
    previous_events = []
    level_three_found = False
    din_max_found = False
    din_max_present = False
    
    # checks to see if we preloaded the tags. 
    # No user should get a perfect 3.00 d-prime, so if it is present then we present tags
    for event in events_raw:
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
        if event_name == '3.00':
            din_max_present = True
            break
    
    # find when the real trials began        
    for index, event in enumerate(events_raw):
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
           
        if event_name == '3.00':
            din_max_found = True
        
        # if we present the tags and we haven't reached the 3.00 tag, then we
        # certainly have not found the beginning of the real trials
        if din_max_found == False and din_max_present == True:
            continue
        
        # update the counter to find the start of the real trials
        j = 1
        
        # once we find the first instance of level 3, we know the real trials have started
        # because it is not possible to reach level 3 in the guide/practice rounds.
        # Once we find level 3, we will reverse iterate through the list of events
        # until we find level 1. We will store the index of this level 1 tag, and we 
        # will call that the start of the real trials
        if event_name == 'FX3X':
            level_three_found = True
            cur_tag = 'FX3X' 
            while cur_tag != 'FX1X':
                # current tag in the reverse iterative process
                cur_tag = rev_dict[str(events_raw[index - j][2])]
                previous_events.append([index - j, cur_tag])
                j += 1
        if level_three_found == True:
            break

    # extract the index in the the events_raw file of the BEGN/STRT tag 
    # and return this as the index we begin with
    start_index = previous_events[-1][0]
    return start_index
            
# transforms each din into a new tag reflecting the level, trial result, and the tag the din is associated with
def fix_events(data, events_raw, rev_dict):
    used_codes = []
    updated_events = []
    valid_count = 0
    
    # find the start index
    start_index = find_start_index(events_raw, rev_dict)
    
    # go through every event in the event list after the real trials start
    for index, event in enumerate(events_raw[start_index - 1:]): 
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
               
        # if not 'FLSH' or 'MVE0' then skip this event
        if not valid_tag(event_name):
            continue
        valid_count += 1
        
        # make it clear what the event code is by adding leading zeroes (will be length = 3)
        while len(str(event_code)) < 3: 
            event_code = '0'+ str(event_code)     
        
        # at this point, we are only analyzing tags that occur in the real trials and are not dins
        
        # time of an event in samples / 1000 (roughly correlates to seconds)
        time = event[0] / 1000 

        # look in window of tag to see if there is a din
        window = data.copy().crop(tmin= time - 0.5, tmax= time + 0.5)
        
        # find events within the specified window
        window_events = mne.find_events(raw=window, stim_channel='STI 014', shortest_event= 1, initial_event=True)
        
        # variable to see if we have found a din within the window
        din_found = False 
        
        # iterate over all of the events in the window
        for window_event in window_events:
            if din_found == True:
                # we take the first DIN we find as the DIN associated with the tag
                continue
            window_event_name = rev_dict[str(window_event[2])]
            
            # if we find a din in the window and it is the first din we find in that window
            # then we create our new tag from it
            if window_event_name == 'DIN1':
                din_found = True 
                # time of the din in samples
                din_time = window_event[0]

                # find the level and the result associated with the din
                level = find_level(events_raw, rev_dict, index)
                result = find_result(events_raw, rev_dict, index)
                if level == -1:
                    print('level error. Tag: ' + event_name)
                if result == -1:
                    print('result error. Tag: ' + event_name)        
                # if we have no problems then construct the new tag
                if level != -1 and result != -1:
                    # result = 99 implies correct, anything else indicates how many correct out of how many total
                    # initial_code is the original tag that we started with
                    # level is the level that the user is on
                    new_code = int(str(result) + str(event_code) + str(level) + '00')  # need to come up with interesting code for these tags
                    while new_code in used_codes:
                        new_code += 1
                    used_codes.append(new_code)
                    updated_events.append(np.array([din_time, 0, new_code]))
        if din_found == False:
           print('no dins in the window. Tag: ' + event_name)
    print("valid tags: " + str(valid_count))
    print("length of new event list: " + str(len(updated_events)))
                    
    return updated_events

# given one of the new tags, this function extracts the level, the original event code, and the result of that trial
def extract_level_code_and_result(event):
    event_code = str(event[2])
    
    event_code = event_code[:-2]    
    # level that the user was on (last two numbers of the new code)
    if event_code[-2] == '0': # if the level was padded with a leading zero then do not include it
        level = event_code[-1]
    else: # double digit level
        level = event_code[-2] + event_code[-1]
    # remove the last two digits of the event code because we already extracted the level
    event_code = event_code[:-2]   
       
    # extract the original code that the new code was created from (middle three numbers of the new code)
    og_code = event_code[-3] + event_code[-2] + event_code[-1]
    og_code = str(int(og_code)) # perform int operation to remove leading zeroes
        
    # remove the digits relating to the original event code
    event_code = event_code[:-3]
        
    # the remaining values in the event code are the digits (as strings) representing the result
    # (99 = Correct), (else = incorrect)
    if len(event_code) != 2:
        print('length: ' + str(len(event_code)) + ' event code: ' +  str(event[2]))
        print('above event code does not equal two')
        event_code = '0' + event_code
    result = event_code
    if result == '99':
        result = 'correct'
    else:
        result = 'incorrect' # + str(result)
    return level, og_code, result

# creates a dict for the new, updated tags
def updated_dict(events, rev_dict):
    dict = {}
    for event in events:
        event_timestamp = str(event[0])
        # find the level, og event code, and result of that trial
        level, og_code, result = extract_level_code_and_result(event)
        og_tag = rev_dict[str(og_code)]
        # create the key accd to MNE documentation where we add slashes
        # see https://mne.tools/dev/auto_tutorials/raw/20_event_arrays.html
        # and find the "Mapping Event IDs to trial descriptors" portion for details
        # about what I am doing
        key = level + '/' + og_tag + '/' + result + '/' + event_timestamp
        value = event[2]
        dict[key] = value
    return dict

# sorts a list of events by timestamp
def sort_chronologically(array):
    time_list = []
    new_events = []
    for event in array:
        time_list.append([event[0], event])
    time_list.sort()
    for event in time_list:
        new_events.append(event[1])
    return new_events.copy()

# creates an evoked data structure from ??
# last big step!!
def average(epochs):
    channels = list(range(15,16))
    #Now that we have our epochs we use the "pick" method to specify which
    # channel (or subset of channels accd to the documentation) that we want to examine
    # Finally we set "avg" to the average over that channel/subset of channels and return this value
    signal = epochs.pick(channels)
    avg = signal.average()

    return avg

# creates a list of correct events vs incorrect events.
# For each of those, creates a list for flash and move start
def separate_events(events, rev_dict):

    # lists for each relevant tag and result
    correct_flash_list = []
    correct_move_start_list = []
    incorrect_flash_list = []
    incorrect_move_start_list = []
    
   # go over all of the events, sort them into their proper list.
   # e.g. a correct flash tag goes into "correct_flash_list"
    for event in events:
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
        
        # add each event corresponding to an incorrect trial to its proper list (FLSH or MVE0)
        if 'incorrect' in event_name:    
            if 'FLSH' in event_name:
                incorrect_flash_list.append(event)
            else:
                incorrect_move_start_list.append(event)
                
        # add each event corresponding to an correct trial to its proper list (FLSH or MVE0)
        else:
            if 'FLSH' in event_name:
                correct_flash_list.append(event)
            else:
                correct_move_start_list.append(event)
    
    # put the sublists for FLSH and MVE0 into a helpful container list
    incorrect_events = [incorrect_flash_list, incorrect_move_start_list] 
    correct_events = [correct_flash_list, correct_move_start_list]     
    return correct_events, incorrect_events

# epochs the events by result (correct vs incorrect) and tag (FLSH vs MVE0)
def epoch_events(raw, correct_events, incorrect_events):
    epoch_dict = {}
    
    # filter for frequencies we are interested in (0.1, 50)
    filtered_raw = filter(raw.copy()) 
    
    # correct/flash epochs
    correct_flash = mne.Epochs(raw= filtered_raw,events=correct_events[0], tmin= -0.5, tmax = 1, baseline=(None,0), picks= ['eeg'], on_missing= 'ignore', preload=True)
    epoch_dict['correct/FLSH'] = correct_flash
    print('correct/flash length: ' + str(len(epoch_dict['correct/FLSH'])))
    
    # incorrect/flash epochs
    incorrect_flash = mne.Epochs(raw= filtered_raw,events=correct_events[1], tmin= -0.5, tmax = 1, baseline=(None,0), picks= ['eeg'], on_missing= 'ignore', preload=True)
    epoch_dict['incorrect/FLSH'] = incorrect_flash
    print('incorrect/flash length: ' + str(len(epoch_dict['incorrect/FLSH'])))
    
    # correct/move_start epochs
    correct_move = mne.Epochs(raw= filtered_raw,events=incorrect_events[0], tmin= -0.5, tmax = 4.5, baseline=(None,0), picks= ['eeg'], on_missing= 'ignore', preload=True)
    epoch_dict['correct/MVE0'] = correct_move
    print('correct/move length: ' + str(len(epoch_dict['correct/MVE0'])))

    # incorrect/move_start epochs
    incorrect_move = mne.Epochs(raw= filtered_raw,events=incorrect_events[1], tmin= -0.5, tmax = 4.5, baseline=(None,0), picks= ['eeg'], on_missing= 'ignore',preload=True)
    epoch_dict['incorrect/MVE0'] = incorrect_move
    print('incorrect/move length: ' + str(len(epoch_dict['incorrect/MVE0'])))
    
    return epoch_dict

# creates a dictionary of epochs for a given subject (soon to be evokeds)
def subject_dict(file_path):
    # print the name of the subject file 
    mne.set_log_level(False)
    print('\n\n' + str(file_path))
    
    # load the raw file
    raw = mne.io.read_raw_egi(file_path, preload=True)
    print('loaded the raw')
    
    # create forward and reverse lookup dictionary
    dict = event_dictionary(raw)
    rev_dict = rev_event_dictionary(dict)
    
    # get the list of events
    events_raw = mne.find_events(raw= raw, stim_channel= 'STI 014', shortest_event=1)
    print('found the events')
    
    # attach each tag to the timestamp of its DIN and update the codes
    events = fix_events(raw, events_raw, rev_dict)
    print('fixed the events')

    # create a new dict/ reverse dict for ease of access
    dict = updated_dict(events, rev_dict)
    rev_dict = rev_event_dictionary(dict)
    print('updated the dictionaries')
    
    # separate the events into sublists based on type of tag ('FLSH', 'MVE0')
    # and result ('correct', 'incorrect')
    correct_events, incorrect_events = separate_events(events, rev_dict)
    print('categorized the events')
    
    # epoch the events into a dictionary with 'correct' or 'incorrect' paired with
    # 'FLSH' or 'MVE0' (e.g. 'incorrect/FLSH' or 'correct/FLSH'). 
    # Total of four entries in the dictionary.
    # Each entry will be a list of epochs corresponding to those conditions
    epoch_dict = epoch_events(raw, correct_events, incorrect_events)
    print('epoched the events')

    #evoked_dict = 
    print('evoked the events')
    return epoch_dict

# iterates over subjects (just one at the moment)
def main():
    mff_path = 'C:\\Users\\Administrator\\Documents\\eeg code\\Aptima Data complete 06.22.23\\Subject 051523\\051523_20230515_122244.mff'
    epoch_dict = subject_dict(mff_path)
    '''
    SCRATCH = os.getenv('SCRATCH')
    data_path = os.path.join(SCRATCH, 'data')
    overall_epoch_dict = {}
    overall_avg_dict = {}
    successes = 0
    failures = 0
    i = 1
    for subject_folder in os.listdir(data_path):
        path = os.path.join(data_path, subject_folder)
        for thing in os.listdir(path):
            if '.mff' in thing:
                try:
                    file_path = os.path.join(path, thing)
                    overall_epoch_dict[str(i)], overall_avg_dict[str(i)]  = subject_dicts(file_path)
                    i += 1
                    successes += 1
                except:
                    failures += 1
                    continue
    print('successes: ' + str(successes))
    print('failures: ' + str(failures))
    '''   
    
if __name__ == "__main__":
    main()
