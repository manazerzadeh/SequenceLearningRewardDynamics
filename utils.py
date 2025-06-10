import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from scipy import stats
import matplotlib.cm as cm
import seaborn as sns
from typing import List
import pingouin as pg

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM


from natsort import index_natsorted

path = "./SLR1/SequenceLearningReward"
path_misc = "./SLR1_miscs/"

seq_length = 7

fingers = ['1', '2', '3', '4', '5'] #mapping of fingers to numbers


iti = 3000   #Inter trial interval
hand = 2 #left or right hand

def read_dat_file(path : str):
    column_names = pd.read_csv(path, delimiter='\t', usecols=lambda column: not column.startswith("Unnamed")).columns
    dtype_dict = {col: int for col in column_names}
    dtype_dict['timeThreshold'] = float
    dtype_dict['timeThresholdSuper'] = float
    dtype_dict['globalStartTime'] = str

    data = pd.read_csv(path, delimiter= '\t', dtype = dtype_dict, usecols=lambda column: not column.startswith("Unnamed"))
    data['cue'] = data['cue'].astype('str')
    return data


def read_dat_files_subjs_list(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    return [read_dat_file(path + "_" + str(sub) + ".dat") for sub in subjs_list]



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['isError'] == 0) & (subj['timingError'] == 0)]


def remove_error_trials_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isTrialError'] == 0) & (subj_press['timingError'] == 0)]


def remove_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isPressError']) == 0]



def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'pressTime'+str(i+1)
        col2 = 'pressTime'+str(i+2)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    # subj['IPI0'] = subj['RT']



def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'hand', 'isTrain', 'symbol', 'cue', 'isMasked', 
                             'windowSize', 'isError', 'timingError', 'isCross', 'crossTime'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)

    

    
    return subj_melted


def finger_melt_presses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'hand', 'isTrain', 'symbol', 'cue', 'isMasked', 
                             'windowSize', 'isError', 'timingError', 'isCross', 'crossTime'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('press') and not _.startswith('pressTime')],
                    var_name='Press_Number', 
                    value_name='Press_Value')
    

    subj_melted['N'] = subj_melted['Press_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'hand', 'isTrain', 'symbol', 'cue', 'isMasked', 
                             'windowSize', 'isError', 'timingError', 'isCross', 'crossTime'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('response')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = subj_melted['Response_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_presses = finger_melt_presses(subj)
    melt_responses = finger_melt_responses(subj)
    merged_df = melt_IPIs.merge(melt_presses, on = ['BN', 'TN', 'SubNum', 'hand', 'isTrain', 'symbol', 'cue', 'isMasked', 
                             'windowSize', 'isError', 'timingError', 'isCross', 'crossTime','N'])\
                                               .merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'hand', 'isTrain', 'symbol', 'cue', 'isMasked', 
                             'windowSize', 'isError', 'timingError', 'isCross', 'crossTime', 'N'] )

    return add_press_error(merged_df)


def add_press_error(merged_df):
    merged_df['isPressError'] = ~(merged_df['Press_Value'] == merged_df['Response_Value'])
    return merged_df



def finger_melt_Forces(subjs_force: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each Finger Force in the whole experiment adding two columns, "Force_Number" determining the order of Force
    and "Force_Value" determining the time of Force
    """

    
    subj_force_melted = pd.melt(subjs_force, 
                    id_vars=['state', 'time', 'BN', 'TN', 'SubNum', 'isTrain', 'isCross', 'crossTime',
                             'points', 'zone', 'isError', 'RT', 'ET'], 
                    value_vars =  [_ for _ in subjs_force.columns if _.startswith('force')],
                    var_name='Force_Number', 
                    value_name='Force_Value')
    
    return subj_force_melted


def cut_force(subjs_force: pd.DataFrame, side_padding) -> pd.DataFrame:
    """
    Cuts the force data to the same length as the IPI data
    """
    subjs_force = subjs_force[(subjs_force['RT'] <= subjs_force['time'] + side_padding) & (subjs_force['time'] <= subjs_force['ET'] + side_padding)]
    return subjs_force



def cut_force_left(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[(subjs_force['RT'] >= subjs_force['time'])]
    return subjs_force


def cut_force_right(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[(subjs_force['ET'] <= subjs_force['time'])]
    return subjs_force

