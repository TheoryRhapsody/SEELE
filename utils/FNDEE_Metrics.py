# -*- coding: utf-8 -*-
from functools import reduce
from itertools import chain
import numpy as np
import pandas as pd
import math
from collections import Counter


def duplicate_removal(datas, condition, model="key"):
    """
    :param datas: data for deduplication, format: [{},{}...]
    :param condition: the key values are for reference when removing duplicate data 
    :param model: deduplication mode, when model="key", param condition is referred; when model="notkey", param condition is not referred.
    :return: data after deduplication, format: [{},{}...]
    """

    def flags(keys, data):
        tmp_dic = {}
        for key in keys:
            tmp_dic.update({key: data.get(key)})
        return tmp_dic

    removal_data = []
    values = []
    if datas:
        if model == "key":
            keys = condition
        elif model == "notkey":
            keys = [key for key in datas[0].keys() if key not in condition]
        else:
            raise ValueError("The model value passed in is incorrect and cannot be matched!")
        for data in datas:
            if flags(keys, data) not in values:
                removal_data.append(data)
                values.append(flags(keys, data))

    return removal_data


def run_function(x, y):
    return x if y in x else x + [y]


def custome_str_cat(sep="|"):
    def str_cat(column_name):
        return sep.join(column_name.unique())

    if sep:
        return str_cat


def read_file_stand(file):
    df = pd.read_json(file)
    df["event_list"] = df["event_list"].apply(lambda x: "|".join([str(xx) for xx in x]))
    if "text" in df.columns:
        df.drop("text", axis=1, inplace=True)
    # remove the labels when event_list==[]
    return df


def read_file_user(file):
    df = pd.read_json(file)
    df["event_list"] = df["event_list"].apply(lambda x: "|".join([str(xx) for xx in x]))
    if "text" in df.columns:
        df.drop("text", axis=1, inplace=True)
    # remove the predictions when event_list==[]
    # df=df[df['event_list']!=''].reset_index(drop=True)
    return df


# preprocess for arguments
def process_arguments(event_list_pred):
    """
    Step 1: Delete any empty role, text, or offset in arguments
    Step 2: Determine whether there is a subset under the same event_type and trigger
    :param event_list_pred:
    :return:
    """

    label_dict = {
        "Experiment": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Manoeuvre": {"Subject": 0, "Date": 0, "Area": 0, "Content": 0},
        "Deploy": {"Subject": 0, "Militaryforce": 0, "Date": 0, "Location": 0},
        "Support": {"Subject": 0, "Object": 0, "Materials": 0, "Date": 0},
        "Accident": {"Subject": 0, "Date": 0, "Location": 0, "Result": 0},
        "Exhibit": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Conflict": {"Subject": 0, "Object": 0, "Date": 0, "Location": 0},
        "Injure": {"Subject": 0, "Quantity": 0, "Date": 0, "Location": 0},
    }
    # Step 1
    for elp in event_list_pred:
        arguments_temp = []
        for argu in elp["arguments"]:
            if argu != {}:
                if argu["text"] != "" or argu["offset"] != [] or argu["role"] != "":
                    if elp['event_type'] in list(label_dict.keys()):
                        if argu['role'] in list(label_dict[elp['event_type']].keys()):
                            arguments_temp.append(argu)
        elp["arguments"] = arguments_temp
    # Step 2
    for i in range(len(event_list_pred)):
        for j in range(i + 1, len(event_list_pred)):
            if (
                event_list_pred[i]["event_type"] == event_list_pred[j]["event_type"]
            ) & (event_list_pred[i]["trigger"] == event_list_pred[j]["trigger"]):

                aa_before = event_list_pred[i]["arguments"]
                aa_before_list = [
                    [aa_b["role"], aa_b["text"], aa_b["offset"]] for aa_b in aa_before
                ]

                aa_after = event_list_pred[j]["arguments"]
                aa_after_list = [
                    [aa_a["role"], aa_a["text"], aa_a["offset"]] for aa_a in aa_after
                ]

                aa_before_judge = [
                    1 if aab in aa_after_list else 0 for aab in aa_before_list
                ]

                aa_after_judge = [
                    1 if aaa in aa_before_list else 0 for aaa in aa_after_list
                ]
                if 0 not in aa_before_judge:
                    event_list_pred[i]["arguments"] = []
                elif 0 not in aa_after_judge:

                    event_list_pred[j]["arguments"] = []
    # Step 3: Remove the prediction when arguments = []
    for elp in event_list_pred:
        if elp["arguments"] == []:
            event_list_pred.remove(elp)
    event_list_pred = duplicate_removal(
        event_list_pred, ["event_type", "trigger", 'arguments']
    )
    return event_list_pred



# evaluate trigger extraction
def calculate_trg(standardResultFile, userCommitFile):
    pred_df = read_file_user(userCommitFile)
    label_df = read_file_stand(standardResultFile)
    chk_df = pd.merge(label_df, pred_df, on=["id"], suffixes=("_label", "_pred"))
    correct_count = {
        "Experiment": 0,
        "Manoeuvre": 0,
        "Deploy": 0,
        "Support": 0,
        "Accident": 0,
        "Exhibit": 0,
        "Conflict": 0,
        "Injure": 0,
    }
    pred_count = {
        "Experiment": 0,
        "Manoeuvre": 0,
        "Deploy": 0,
        "Support": 0,
        "Accident": 0,
        "Exhibit": 0,
        "Conflict": 0,
        "Injure": 0,
    }
    label_count = {
        "Experiment": 0,
        "Manoeuvre": 0,
        "Deploy": 0,
        "Support": 0,
        "Accident": 0,
        "Exhibit": 0,
        "Conflict": 0,
        "Injure": 0,
    }
    for i in range(len(chk_df)):
        event_list_pred = chk_df.loc[i, "event_list_pred"].split("|")
        event_list_label = chk_df.loc[i, "event_list_label"].split("|")
        if len(event_list_pred) > 0:
            event_list_pred = [eval(elp) for elp in event_list_pred if elp != ""]
            for elp in event_list_pred:
                pred_count[elp["event_type"]] += 1
        if len(event_list_label) > 0:
            event_list_label = [eval(ell) for ell in event_list_label if ell != ""]
            for ell in event_list_label:
                if ell["event_type"] != "":
                    label_count[ell["event_type"]] += 1

        event_list_pred = process_arguments(event_list_pred)
        event_list_pred_arg = []
        for elp in event_list_pred:
            del elp["arguments"]
            event_list_pred_arg.append(elp)

        event_list_label_arg = []
        for ell in event_list_label:
            del ell["arguments"]
            event_list_label_arg.append(ell)
        if len(event_list_pred_arg) > 0 and len(event_list_label_arg) > 0:
            for elp in event_list_pred_arg:
                if elp in event_list_label_arg:
                    correct_count[elp["event_type"]] += 1
    P_UP = 0  
    P_DOWN = 0  
    R_UP = 0  #
    R_DOWN = 0  
    for et in [
        "Experiment",
        "Manoeuvre",
        "Deploy",
        "Support",
        "Accident",
        "Exhibit",
        "Conflict",
        "Injure",
    ]:
        P_UP += correct_count[et]
        P_DOWN += pred_count[et]
        R_UP += correct_count[et]
        R_DOWN += label_count[et]

    P = (P_UP+ 1e-16) / (P_DOWN+ 1e-16)
    R = (R_UP+ 1e-16) / (R_DOWN+ 1e-16)
    return P, R, (2 * P * R) / (P + R)


def calculate_arg3_step(match_index, ellarguments, event_list_pred,coref_arguments):
    count = []
    if len(match_index) >= 1:
        for mi in match_index:
            temp = 0
            for ellargu in ellarguments:
                pred_arguments = event_list_pred[mi]
                if ellargu in pred_arguments["arguments"]:
                    temp += 1
                else:
                    elparguments=pred_arguments['arguments']
                    for elpargu in elparguments:
                        # if elpargu not in ellarguments:
                        if elpargu != ellargu:
                            # for argu in ellarguments:
                                if elpargu != {}:
                                    if elpargu['role']!='' or elpargu['text']!='' or elpargu['offset']!=[]:
                                        if ellargu["role"] == elpargu["role"]: #change if argu["role"] == elpargu["role"]
                                            match_coref = [
                                                {
                                                    "text": ellargu["text"],
                                                    "offset": ellargu["offset"],
                                                },
                                                {
                                                    "text": elpargu["text"],
                                                    "offset": elpargu["offset"],
                                                },
                                            ]
                                            rrresult = [
                                                1
                                                if match_coref[0] in ca
                                                   and match_coref[1] in ca
                                                else 0
                                                for ca in coref_arguments
                                            ]
                                            if 1 in rrresult:
                                                temp += 1
                                                break
            count.append(temp)
        return match_index,count
    
# evaluate argument extraction
def calculate_arg3(standardResultFile,userCommitFile):
    label_dict = {
        "Experiment": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Manoeuvre": {"Subject": 0, "Date": 0, "Area": 0, "Content": 0},
        "Deploy": {"Subject": 0, "Militaryforce": 0, "Date": 0, "Location": 0},
        "Support": {"Subject": 0, "Object": 0, "Materials": 0, "Date": 0},
        "Accident": {"Subject": 0, "Date": 0, "Location": 0, "Result": 0},
        "Exhibit": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Conflict": {"Subject": 0, "Object": 0, "Date": 0, "Location": 0},
        "Injure": {"Subject": 0, "Quantity": 0, "Date": 0, "Location": 0},
    }
    pred_df = read_file_user(userCommitFile)
    label_df = read_file_stand(standardResultFile)
    pred_count = {
        "Experiment": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Manoeuvre": {"Subject": 0, "Date": 0, "Area": 0, "Content": 0},
        "Deploy": {"Subject": 0, "Militaryforce": 0, "Date": 0, "Location": 0},
        "Support": {"Subject": 0, "Object": 0, "Materials": 0, "Date": 0},
        "Accident": {"Subject": 0, "Date": 0, "Location": 0, "Result": 0},
        "Exhibit": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Conflict": {"Subject": 0, "Object": 0, "Date": 0, "Location": 0},
        "Injure": {"Subject": 0, "Quantity": 0, "Date": 0, "Location": 0},
    }
    label_count = {
        "Experiment": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Manoeuvre": {"Subject": 0, "Date": 0, "Area": 0, "Content": 0},
        "Deploy": {"Subject": 0, "Militaryforce": 0, "Date": 0, "Location": 0},
        "Support": {"Subject": 0, "Object": 0, "Materials": 0, "Date": 0},
        "Accident": {"Subject": 0, "Date": 0, "Location": 0, "Result": 0},
        "Exhibit": {"Subject": 0, "Equipment": 0, "Date": 0, "Location": 0},
        "Conflict": {"Subject": 0, "Object": 0, "Date": 0, "Location": 0},
        "Injure": {"Subject": 0, "Quantity": 0, "Date": 0, "Location": 0},
    }
    true_per_type_count = {
        "Experiment": 0,
        "Manoeuvre": 0,
        "Deploy": 0,
        "Support": 0,
        "Accident": 0,
        "Exhibit": 0,
        "Conflict": 0,
        "Injure": 0,
    }
    F1_per_type_count = {
        "Experiment": 0,
        "Manoeuvre": 0,
        "Deploy": 0,
        "Support": 0,
        "Accident": 0,
        "Exhibit": 0,
        "Conflict": 0,
        "Injure": 0,
    }
    correct_num=0
    chk_df = pd.merge(label_df, pred_df, on=["id"], suffixes=("_label", "_pred"))
    noMeanCount = 0
    for i in range(len(chk_df)):
        coref_arguments = chk_df.loc[i, "coref_arguments"]
        event_list_pred = chk_df.loc[i, "event_list_pred"].split("|")
        event_list_label = chk_df.loc[i, "event_list_label"].split("|")
        if len(event_list_pred) > 0:
            event_list_pred = [eval(elp) for elp in event_list_pred if elp != ""]
            event_list_pred = duplicate_removal(
                event_list_pred, ["event_type", "trigger", "arguments"]
            )
            for elp in event_list_pred:
                for ea in elp['arguments']:
                    if ea != {}:
                        if ea['role'] != '' or ea['text'] != '' or ea['offset'] != []:
                            if elp['event_type'] in list(label_dict.keys()):
                                if ea['role'] in list(label_dict[elp['event_type']].keys()):
                                    pred_count[elp["event_type"]][ea["role"]] += 1
                                else:
                                    noMeanCount += 1
                        elif ea['role'] == '' or ea['text'] == '' or ea['offset'] == []:
                            noMeanCount += 1

        if len(event_list_label) > 0:
            event_list_label = [eval(ell) for ell in event_list_label if ell != ""]
            for ell in event_list_label:
                for ea in ell["arguments"]:
                    label_count[ell["event_type"]][ea["role"]] += 1

        if len(event_list_pred) > 0 and len(event_list_label) > 0:
            match_list=[]
            for ell in event_list_label:
                match_index = [i for i in range(len(event_list_pred))]
                event_type_index = [ elp["event_type"] for elp in event_list_pred]
                ellarguments=ell['arguments']
                max_index,count = calculate_arg3_step(
                    match_index, ellarguments, event_list_pred,coref_arguments
                )


                for ind in range(len(count)):
                    match_list.append((count[ind],event_list_label.index(ell),max_index[ind],event_type_index[ind]))

            #match_list sort
            if len(match_list)>0:
                match_list=sorted(match_list,key=lambda x:x[0],reverse=True)
                max_match_num = min(len(event_list_label), len(event_list_pred))
                already_label_list = []
                already_predict_list = []
                already_match_num = 0
                for each_match in match_list:
                    if each_match[1] not in already_label_list and each_match[2] not in already_predict_list and already_match_num < max_match_num :
                        correct_num += each_match[0]
                        true_per_type_count[each_match[3]] += each_match[0]
                        already_label_list.append(each_match[1])
                        already_predict_list.append(each_match[2])
                        already_match_num += 1
                        
    P_UP = 0  
    P_DOWN = 0  
    R_UP = 0  
    R_DOWN = 0  

    P_UP_PER_TYPE = 0  
    P_DOWN_PER_TYPE = 0  
    R_UP_PER_TYPE = 0  
    R_DOWN_PER_TYPE= 0  

    true_count = 0
    for each_type, each_count in true_per_type_count.items():
        true_count += each_count 
    for et in [
        "Experiment",
        "Manoeuvre",
        "Deploy",
        "Support",
        "Accident",
        "Exhibit",
        "Conflict",
        "Injure",
    ]:
        P_UP =correct_num
        P_DOWN += sum(pred_count[et].values())
        R_UP =correct_num
        R_DOWN += sum(label_count[et].values())
        
        P_UP_PER_TYPE = true_per_type_count[et]
        P_DOWN_PER_TYPE = sum(pred_count[et].values())
        R_UP_PER_TYPE = true_per_type_count[et]
        R_DOWN_PER_TYPE = sum(label_count[et].values())

        P_PER_TYPE = (P_UP_PER_TYPE+ 1e-16) / (P_DOWN_PER_TYPE + 1e-16)
        R_PER_TYPE = (R_UP_PER_TYPE+ 1e-16) / (R_DOWN_PER_TYPE + 1e-16)
        F1_per_type_count[et] = (2 * P_PER_TYPE * R_PER_TYPE) / (P_PER_TYPE + R_PER_TYPE)


    P_DOWN+=noMeanCount
    P = (P_UP+ 1e-16) / (P_DOWN + 1e-16)
    R = (R_UP+ 1e-16) / (R_DOWN + 1e-16)
    
    return P, R, (2 * P * R) / (P + R), F1_per_type_count




def FNDEE_judge(standardResultFile, userCommitFile, evaltype=0):
    Trg_P, Trg_R, Trg_F1 = calculate_trg(standardResultFile, userCommitFile)
    Arg_P, Arg_R, Arg_F1, Per_type_list = calculate_arg3(standardResultFile, userCommitFile)

    return Trg_P, Trg_R, Trg_F1, Arg_P, Arg_R, Arg_F1, Per_type_list