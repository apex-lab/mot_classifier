import mne, json, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import math
import mne_connectivity

power_freq = 60

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

# filter/clean up our data (ripped from Felix's code)
# need more info from Shannon about which frequencies we want to remove and which we want to keep
def filter(data, low, high):
    # notch filters are filters that remove 
    notch_filtered_raw = data.copy().notch_filter(freqs=[power_freq])

    # Apply a band-pass filter to keep frequencies of interest low = 1, high = 50 
    filtered_raw = notch_filtered_raw.copy().filter(l_freq=low, h_freq=high)
    return filtered_raw

# returns the average over each channel as a dictionary
# NEEEDS FIXING 
def average(data, events, event_dict):
    avg_dict = {}
        
    # Define the epoch parameters (each epoch is 1.5 seconds)
    minimum = -0.5  # start of each epoch (seconds)
    maximum = 1  # end of each epoch (seconds)

    # Compute Epochs
    epochs_sep = []
    epochs_sep_avg = []
    channel_names = []

    # Given a bit of data we extract (min - max) length epochs of that data.
    # do we want to use projection vectors? How about baseline period? how about preload?
    epochs_grouped = mne.Epochs(raw= data, event_id= event_dict, tmin= minimum, tmax = maximum, baseline=(None,0), picks = ['eeg','eog'], on_missing= 'ignore')


    #NEW Now that we have our epochs we use the "pick" method to specify which
    # channel (or subset of channels accd to the documentation) that we want to examine
    # Finally we set "avg" to the average over that channel/subset of channels and return this value
    #signal = epochs_grouped.pick(channel)
    #avg = signal.average()

    #return avg

# returns true if event name is either 'FLSH', 'FXNM', 'MVE0' or 'MVE1'
def valid_tag(event_name):
    if event_name == 'FLSH':
        return True
    if event_name == 'MVE0':
        return True
    if event_name == 'MVE1':
        return True
    if event_name[:2] == 'FX':
        return True
    return False

# finds the level associated with the din and returns it as a string with length = 2
def find_level(events_raw, rev_dict, index):
    level = ''
    level_found = False
    j = 0
    
    # find the level in which the tag occured
    while level_found == False:
        try:
            
            # if it is a fixation tag then extract the level
            if 'FX' in rev_dict[str(events_raw[index-j][2])]: 
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
                result_found = True
                
            # if we find a correct trial
            elif 'CRCT' == rev_dict[str(events_raw[index+j][2])]: 
                result = str(99) # 99 will indicate a correct trial
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
    for index, event in enumerate(events_raw):
        # store the netstation id and name of the event
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
        # update the counter to find the start of the real trials
        j = 1
        
        # once we find the first instance of level 3, we know the real trials have started
        # and we will reverse iterate through the list of events using j until we find
        # the moment when the subject moved from the practice trials to the real trials.
        # This is denoted by either 'BEGN' or 'STRT'
        if event_name == 'FX3X':
            level_three_found = True
            cur_tag = 'FX3X' 
            while cur_tag != 'BEGN' and cur_tag != 'STRT':
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
            
# transforms each din into a new tag reflecting the level, performance on the level, and the tag the din is associated with
# also want to keep track of the OG place where we found each relevant tag as an element/entry in the new dictionary we create
def fix_events(data, events_raw, rev_dict):
    updated_events = []
    valid_count = 0
    
    start_index = find_start_index(events_raw, rev_dict)
    # go thru every event in the event list
    for index, event in enumerate(events_raw): 
        # store the netstation id and name of the event
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
               
        # if not 'FLSH', 'FXNM', 'MVE0', or 'MVE1' or not real trial then skip this event
        if (not valid_tag(event_name)) or (index < start_index):
            continue
        valid_count += 1
        
        # make it clear what the event code is by adding leading zeroes (will be length = 3)
        while len(str(event_code)) < 3: 
            event_code = '0'+ str(event_code)     
        
        # at this point, we are only analyzing tags that occur in the real trials and are not dins
        
        # time of an event in samples * 1000 (roughly correlates to seconds)
        time = event[0] / 1000 

        # look in window of tag to see if there is a din
        window = data.copy().crop(tmin= time - 0.3, tmax= time + 0.3) 
        
        try: 
            # find events within the specified window
            window_events = mne.find_events(raw=window, stim_channel='STI 014', shortest_event= 1, initial_event=True)
        except:
            continue
        #din_found = False # variable to see if we have found a din within the window
        
        # iterate over all of the events in the window
        for window_event in window_events:
            # find the name of the event
            window_event_name = rev_dict[str(window_event[2])]
            
            # if we find a din in the window and it is the first din we find in that window
            if window_event_name == 'DIN1': 
                # time of the din in samples
                din_time = window_event[0]

                # find the level and the result associated with the din
                level = find_level(events_raw, rev_dict, index)
                result = find_result(events_raw, rev_dict, index)
                        
                # if we have no problems then construct the new tag
                if level != -1 and result != -1:
                    # result = 99 implies correct, anything else indicates how many correct out of how many total
                    # initial_code is the original tag that we started with
                    # level is the level that the user is on
                    new_code = int(str(result) + str(event_code) + str(level))  # need to come up with interesting code for these tags
                    updated_events.append(np.array([din_time, 0, new_code]))
    print("valid tags: " + str(valid_count))
    print("length of new event list: " + str(len(updated_events)))
                    
    return updated_events

# given one of the new tags, this function extracts the level, the original event code, and the result of that trial
def extract_level_code_and_result(event):
    event_code = str(event[2])
        
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
        print('remaining event code does not equal two')
    result = event_code
    if result == '99':
        result = 'correct'
    else:
        result = 'incorrect'
    return level, og_code, result

# creates a dict for the new, updated tags
def updated_dict(events, rev_dict):
    dict = {}
    for event in events:
        event_timestamp = event[0]
        # find the level, og event code, and result of that trial
        level, og_code, result = extract_level_code_and_result(event)
        og_tag = rev_dict[str(og_code)]
        # create the key accd to MNE documentation where we add slashes
        # see https://mne.tools/dev/auto_tutorials/raw/20_event_arrays.html
        # and find the "Mapping Event IDs to trial descriptors" portion for details
        # about what I am doing
        key = level + '/' + og_tag + '/' + result
        value = event[2]
        
        # the word (the key) is defined to be the new, transformed code
        # So each unique combination of level, original code, and result
        # will have a unique code. And every code will correspond to a unique
        # tag consisting of level, original code, and result (A bijection!!!!!!)
        dict[key] = value
    return dict

def sort_chronologically(array):
    i = 0
    time_list = []
    new_events = []
    for event in array:
        time_list.append([event[0], event])
    time_list.sort()
    for event in time_list:
        new_events.append(event[1])
    return new_events.copy()
        
# create a 2-tuple of epoch lists. The first always being 
def epoch_dictionary(raw, subsetted_lists):
    epoch_dict = {}
    list_names = ['correct', 'incorrect']
    i = 0
            
    # iterate over both the correct/incorrect ARRAYS
    for list in subsetted_lists:
        key = list_names[i] # either 'correct' or 'incorrect'
        epoch_dict[key] = {}

        print('\n\n START OF NEW SET OF EPOCHS \n\n')
        epochs = mne.Epochs(raw= raw,events=list, tmin= -0.5, tmax = 1, baseline=(None,0), picks= ['eeg','eog'], on_missing= 'ignore', preload = True)
        print('\n\n END OF NEW SET OF EPOCHS \n\n')
        epoch_dict[key] = epochs 
        i += 1
    
    return epoch_dict
            
# given the epoch dict and a list of the relevant keywords, this allows us to use the keywords to extract
# which of the sublists we want. The return value will be a 2-tuple, with the first entry being the epochs
# formed from the events that were a part of trials where the user was correct. The second entry will be 
# symmetrical for incorrect. The keywords allow us to extract the groups of epochs we are interested in.
# For example, keywords = ['FX', 'MVE1'] will return a 2-tuple of epochs objects, with the first entry being
# the epochs of 'FX' tags and 'MVE1' tags as one grand epoch. The second object will be the symmetrical version
# for the incorrect ones.       
def specify_lists(evt_cat_lists, keywords):
    result_dict = {'0': 'correct', '1': 'incorrect'}
    tag_dict = {'0': 'FLSH', '1': 'MVE0', '2': 'MVE1', '3': 'FX'}
    correct_list = []
    incorrect_list = []
    overall_list = [correct_list, incorrect_list]
    
    # iterate over correct/incorrect lists
    i = 0
    for result in evt_cat_lists:
        # iterate over the given keywords (a subset of the tags)
        for keyword in keywords:
            j = 0
            # iterate over the tags to see which keywords are there
            for tag in result:
                # if the keyword is in the name of the list, then add each element of the list
                if keyword in tag_dict[str(j)]:
                    for element in tag:
                        overall_list[i].append(element)
                j += 1
        # sort these lists according to timestamp 
        overall_list[i] = np.asarray(sort_chronologically(overall_list[i]).copy())
        
        i += 1
    return overall_list
                    

# creates a list of correct events vs incorrect events.
# For each of those, creates a list for flash, fixation, move start and move end.
# For the fixation list, creates a list for each level
def seperate_events(events, rev_dict):
    correct_events = []
    incorrect_events = []
    
    # iterate over all of the relevant events
    # sort them into lists of correct/incorrect
    for event in events:
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
        if 'incorrect' in event_name:
            #print('incorrect event name: ' + str(event_name))
            incorrect_events.append(event)
        else:
            correct_events.append(event)
    #print('length of correct events: ' + str(len(correct_events)))
    #print('length of incorrect events: ' + str(len(incorrect_events)))
    #-----------------------------------------------------------------------------
    # at this point the events are properly sorted into correct vs incorrect lists
    #-----------------------------------------------------------------------------
    tag_list = ['FLSH', 'MVE0', 'MVE1', 'FX'] # list of relevant tags
    
    # sublists for each relevant tag
    flash_list = []
    move_start_list = []
    move_end_list = []
    fixation_list = []
    list_of_lists = [flash_list, move_start_list, move_end_list, fixation_list]
    
    # create sublists from the correct events
    for event in correct_events:
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
        
        # go thru the tags in the list of relevant tags.
        # find which of these relevant tags the event is.
        i = 0
        for tag in tag_list:
            #print('event_name: ' + event_name)
            if tag in event_name:
                break
            i += 1
        
        # add this event to the proper sublist
        list_of_lists[i].append(event)
    
    # add each sublist to the correct_events list
    # reset sublists to an empty lists
    correct_events = []
    for list in list_of_lists:
        list = sort_chronologically(list).copy()
        correct_events.append(list)
     
     
    flash_list_inc = []
    move_start_list_inc = []
    move_end_list_inc = []
    fixation_list_inc = []
    list_of_lists_inc = [flash_list_inc, move_start_list_inc, move_end_list_inc, fixation_list_inc]   
    # create sublists from the incorrect events     
    for event in incorrect_events:
        event_code = event[2]
        event_name = rev_dict[str(event_code)]
        
        # go thru the tags in the list of relevant tags.
        # find which of these relevant tags the event is.
        i = 0
        for tag in tag_list:
            if tag in event_name:
                break
            i += 1
        
        # add this event to the proper sublist
        list_of_lists_inc[i].append(event)
    
    # add each sublist to the incorrect events list
    incorrect_events = []
    for list in list_of_lists_inc:
        list = sort_chronologically(list).copy()
        incorrect_events.append(list)
    for list in correct_events:
        print('length of correct events: ' + str(len(list)))
    for list in incorrect_events:
        print('length of incorrect events: ' + str(len(list)))
    # list of the level numbers
    # we want to find the maximum level obtained
    #levels_list = []
    
    # iterate over all of the correct fixation events and extract the maximum level attained
    #max_level = 0
    #for event in correct_events[3]:
    #    level, og_code, result = extract_level_code_and_result(event)
    #    if int(level) > max_level:
    #        max_level = int(level)
    
    # create a list of empty lists with length of the maximum level
    #for i in range(0, max_level):
    #    levels_list.append([])
    
    # for each event in the fixation list, find its level and then append that event
    # to the entry in the list that corresponds to that level (enty = level - 1)
    #for event in correct_events[3]:
    #    level, og_code, result = extract_level_code_and_result(event)
    #    levels_list[level - 1].append(event)
    
    #correct_events[3] = []
    #correct_events[3] = levels_list
    #levels_list = []
    
        # iterate over all of the correct fixation events and extract the maximum level attained
    #max_level = 0
    #for event in incorrect_events[3]:
    #    level, og_code, result = extract_level_code_and_result(event)
    #    if int(level) > max_level:
    #        max_level = int(level)
    
    # create a list of empty lists with length of the maximum level
    #for i in range(0, max_level):
    #    levels_list.append([])
    
    # for each event in the fixation list, find its level and then append that event
    # to the entry in the list that corresponds to that level (enty = level - 1)
    #for event in incorrect_events[3]:
    #    level, og_code, result = extract_level_code_and_result(event)
    #    levels_list[level - 1].append(event)
    
    #incorrect_events[3] = []
    #incorrect_events[3] = levels_list
    
    return [correct_events, incorrect_events]

# creates a powerset of elements in the list
# fuck it we're just going to hard code it for now
# but this would be a fun coding project
def powset():
    powerset = [['FLSH'], ['MVE0'], ['MVE1'], ['FX'],\
        ['FLSH', 'MVE0'], ['FLSH', 'MVE1'], ['FLSH', 'FX'], ['MVE0', 'MVE1'], ['MVE0', 'FX'], ['MVE1', 'FX'],\
            ['FLSH', 'MVE0', 'MVE1'], ['FLSH', 'MVE0', 'FX'], ['FLSH', 'MVE1', 'FX'], ['MVE0', 'MVE1', 'FX'],\
                ['FLSH','MVE0','MVE1','FX']]
    return powerset

def main():
    # still need to create a way to iterate over all of the mff files
    # mne.set_log_level(False)
    file_path = 'C:\\Users\\Administrator\\Documents\\eeg code\\Aptima Data complete 06.22.23\\Subject 051523\\051523_20230515_122244.mff'
    raw = mne.io.read_raw_egi(file_path, preload= True)
    dict = event_dictionary(raw)
    rev_dict = rev_event_dictionary(dict)
    
    # extract the total list of events and then create a new events list
    # with the tags/codes of interest/updated codes
    events_raw = mne.find_events(raw= raw, stim_channel= 'STI 014', shortest_event=1)
    events = fix_events(raw, events_raw, rev_dict)
    
    # create a new dict/ reverse dict for ease of access
    dict = updated_dict(events, rev_dict)
    rev_dict = rev_event_dictionary(dict)
    
    # 2 level nested dictionary of all of the epochs.
    # level 1 is correct epochs vs incorrect epochs.
    # level 2 is lists: 4 lists within each level 1 dictionary
    # The 4 lists are: 'FLSH', 'FXNM', 'MVE0', 'MVE1'
    print('\n\n starting event categorization \n\n')
    event_categorization_list = seperate_events(events, rev_dict)
    for list in event_categorization_list:
        for sublist in list:
            print(str(len(sublist)))
    
    # returns the powerset of ['FLSH', 'MVE0', 'MVE1', 'FX']
    powerset = powset()
    subj_epoch_dict = {}
    # subselect the lists that we are interested in for the subject (Subsets of 'FLSH', 'MVE0', 'MVE1', 'FX')
    print('\n\n start subselecting lists and epoching \n\n')
    # iterate over each element in the powerset
    for element in powerset:
        # for each element set the key to empty and i = 0
        key = ''
        i = 0
        # iterate over each tag within that element of the powerset
        for object in element:
            # set the first part of the key to the first tag
            if i == 0:
                key = object
            # for the rest of the tags, add a slash and the tag to the end of the key
            else:
                key = key + '/' + object
            i += 1
        # based on that list of tags, create a master list of correct vs incorrect trials for those keys
        subsetted_lists = specify_lists(event_categorization_list.copy(), element).copy()
        
        # epochs will be a dictionary with two keys: correct and incorrect for the specified tags
        epochs = epoch_dictionary(raw, subsetted_lists)
        
        # each key in our subject specific dictionary will be the element of the powerset that our
        # epochs are created from. So to index into the epochs for correct trials for 'FX' and 'FLSH'
        # for this subject, you would look under subj_epoch_dict['FLSH/FX']['correct']
        # MAKE SURE TO LIST THE TAGS IN ORDER OF ['FLSH', 'MVE0', 'MVE1', 'FX']
        subj_epoch_dict[key] = epochs
    
    print(str(len(subj_epoch_dict['FX']['correct'].events)))
    print(str(len(subj_epoch_dict['FLSH/MVE0']['incorrect'].events)))
    print(str(len(subj_epoch_dict['MVE0/MVE1/FX']['correct'].events)))
    print(str(len(subj_epoch_dict['FLSH/MVE0/MVE1/FX']['incorrect'].events)))

    # at this point, i have a method for taking a user's data, extracting the relevant tags, tying those
    # tags to their associated DIN, separating those tags into correct vs incorrect lists, subsetting those
    # correct/incorrect lists into FLSH, MVE0, MVE1, and FX, and then creating all possible combinations 
    # thereof (all combinations within the correct list and all possible combinations within the incorrect list),
    # and thus creating all possible sets of epochs.
    
    # NEED: to create level by level epochs for the user (probably each user's epochs are in a dict with two keys,
    # one being 'tags' and the other being 'levels' and every possible combination within the levels of the four tags)
    
    # ALSO NEED: generalize this to do this for n > 1 number of users.

    # create the level by level data for the user and all subsets of tags
    
    # for each combination of tags, there is a dict with two entries ('correct', 'incorrect') 
    # combination is the string representing the combination of tags that we are looking at.
    # value is the dictionary corresponding to either 'correct' or 'incorrect'
    
    # creates a dictionary with the highest level being correct/incorrect dicts
   # subj_level_dict = {'correct': {}, 'incorrect': {}}
    #for combination, value in subj_epoch_dict.items():
        # for each of either 'correct'/'incorrect', the epoch is the set of epochs associated with
        # that result and the specific combination of tags that we are looking at
    #    for result, epoch in value.items():
            # creates an empty list for each potential level within the result category (cast to ndarray later)
    #        for i in range(1, 101):
    #            subj_level_dict['level ' + str(i)] = {}
            # iterate over all events in the epoch, find its level, and then add it to its proper level dict entry
    #        for event in epoch.events():
                # extract the level, code and the result
    #            level, code, result = extract_level_code_and_result(event)
                
                # add the event to its proper dictionary entry
    #            level_dict['level ' + str(level)].append(event)
if __name__ == "__main__":
    main()