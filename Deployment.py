
import streamlit as st
import numpy as np
import joblib 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df_cols_names = joblib.load('df_cols_names.pkl')
Model = joblib.load('Model.pkl')
target_encoder = joblib.load('target_encoder.pkl')
Objects_dict = joblib.load('Objects_dict.pkl')


## Create_Encoded_Features Function
def Create_Encoded_Features(df_test):
    encoded_dfs = {}
    df_language  = pd.DataFrame(Objects_dict['TF_languages'].transform([df_test['LanguageHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_languages'].get_feature_names_out())
    df_databases = pd.DataFrame(Objects_dict['TF_databases'].transform([df_test['DatabaseHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_databases'].get_feature_names_out())
    df_platform  = pd.DataFrame(Objects_dict['TF_platforms'].transform([df_test['PlatformHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_platforms'].get_feature_names_out())
    df_webframes = pd.DataFrame(Objects_dict['TF_webframes'].transform([df_test['WebframeHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_webframes'].get_feature_names_out())
    df_MiscTech  = pd.DataFrame(Objects_dict['TF_MiscTech'].transform([df_test['MiscTechHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_MiscTech'].get_feature_names_out())
    df_tools     = pd.DataFrame(Objects_dict['TF_tools'].transform([df_test['ToolsTechHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_tools'].get_feature_names_out())
    df_NEWCollabTools = pd.DataFrame(Objects_dict['TF_NEWCollabTools'].transform([df_test['NEWCollabToolsHaveWorkedWith'].fillna('NaN')[0]]).toarray(), columns=Objects_dict['TF_NEWCollabTools'].get_feature_names_out())
    encoded_dfs['LanguageHaveWorkedWith'] = df_language
    encoded_dfs['DatabaseHaveWorkedWith'] = df_databases
    encoded_dfs['PlatformHaveWorkedWith'] = df_platform
    encoded_dfs['WebframeHaveWorkedWith'] = df_webframes
    encoded_dfs['MiscTechHaveWorkedWith'] = df_MiscTech
    encoded_dfs['ToolsTechHaveWorkedWith'] = df_tools
    encoded_dfs['NEWCollabToolsHaveWorkedWith'] = df_NEWCollabTools
    res = pd.concat(encoded_dfs, axis=1)
    return res


LanguageHaveWorkedWith_Cols = df_cols_names['LanguageHaveWorkedWith'].columns.tolist()
LanguageHaveWorkedWith_Cols.remove('nan')

DatabaseHaveWorkedWith_Cols = df_cols_names['DatabaseHaveWorkedWith'].columns.tolist()
DatabaseHaveWorkedWith_Cols.remove('nan')

PlatformHaveWorkedWith_Cols = df_cols_names['PlatformHaveWorkedWith'].columns.tolist()
PlatformHaveWorkedWith_Cols.remove('nan')

WebframeHaveWorkedWith_Cols = df_cols_names['WebframeHaveWorkedWith'].columns.tolist()
WebframeHaveWorkedWith_Cols.remove('nan')

MiscTechHaveWorkedWith_Cols = df_cols_names['MiscTechHaveWorkedWith'].columns.tolist()
MiscTechHaveWorkedWith_Cols.remove('nan')

ToolsTechHaveWorkedWith_Cols = df_cols_names['ToolsTechHaveWorkedWith'].columns.tolist()
ToolsTechHaveWorkedWith_Cols.remove('nan')

NEWCollabToolsHaveWorkedWith_Cols = df_cols_names['NEWCollabToolsHaveWorkedWith'].columns.tolist()
NEWCollabToolsHaveWorkedWith_Cols.remove('nan')


TECH_COLS = ['LanguageHaveWorkedWith',
            'DatabaseHaveWorkedWith',
            'PlatformHaveWorkedWith',
            'WebframeHaveWorkedWith',
            'MiscTechHaveWorkedWith',
            'ToolsTechHaveWorkedWith',
            'NEWCollabToolsHaveWorkedWith']


def Prediction(Prog_Languages, Databases, Platforms, Webframes, MiscTechs, Tools, NewCollabTools):
    df_test = pd.DataFrame(columns=TECH_COLS)
    df_test.at[0,'LanguageHaveWorkedWith'] = Prog_Languages
    df_test.at[0,'DatabaseHaveWorkedWith'] = Databases
    df_test.at[0,'PlatformHaveWorkedWith'] = Platforms
    df_test.at[0,'WebframeHaveWorkedWith'] = Webframes
    df_test.at[0,'MiscTechHaveWorkedWith'] = MiscTechs
    df_test.at[0,'ToolsTechHaveWorkedWith'] =  Tools
    df_test.at[0,'NEWCollabToolsHaveWorkedWith'] = NewCollabTools
    df_test_encoded = Create_Encoded_Features(df_test)
    result = Model.predict(df_test_encoded)
    return result

def Main():
    st.markdown('<p style="font-size:50px;text-align:center;"><strong>IT Jobs Prediction</strong></p>',unsafe_allow_html=True)
    col1_1 , col1_2 = st.columns([2,2]) 
    col2_1 , col2_2 = st.columns([2,2])
    col3_1 , col3_2 = st.columns([2,2])
    col4_1 , col4_2 = st.columns([2,2])
    
    with col1_1:
        Prog_Languages = st.multiselect('Select Programming Languages you have worked with : ', LanguageHaveWorkedWith_Cols)
    with col1_2:
        Databases = st.multiselect('Select Databases that you have worked with : ', DatabaseHaveWorkedWith_Cols)
        
    with col2_1:    
        Platforms = st.multiselect('Select Platforms that you have worked with : ', PlatformHaveWorkedWith_Cols)
    with col2_2:
        Webframes = st.multiselect('Select Web Frames that you have worked with : ', WebframeHaveWorkedWith_Cols)
        
    with col3_1:    
        MiscTechs = st.multiselect('Select MiscTech that you have worked with : ', MiscTechHaveWorkedWith_Cols)
    with col3_2:
        Tools = st.multiselect('Select Tools that you have worked with : ', ToolsTechHaveWorkedWith_Cols)
        
    with col4_1:
        NewCollabTools = st.multiselect('Select NewCollabTools that you have worked with : ', NEWCollabToolsHaveWorkedWith_Cols)
    with col4_2:
        st.text('')
        st.text('')
        if st.button("Predict"):
            if Prog_Languages == []:
                Prog_Languages = np.nan
            else:
                Prog_Languages = ';'.join(Prog_Languages)
                
            if Databases == []:
                Databases = np.nan
            else:
                Databases = ';'.join(Databases)
                
            if Platforms == []:
                Platforms = np.nan
            else:
                Platforms = ';'.join(Platforms)
                
            if Webframes == []:
                Webframes = np.nan
            else:
                Webframes = ';'.join(Webframes)
                
            if MiscTechs == []:
                MiscTechs = np.nan
            else:
                MiscTechs = ';'.join(MiscTechs)
            if Tools == []:
                Tools = np.nan
            else:
                Tools = ';'.join(Tools)
                
            if NewCollabTools == []:
                NewCollabTools = np.nan
            else:
                NewCollabTools = ';'.join(NewCollabTools)
            
            res = Prediction(Prog_Languages, Databases, Platforms, Webframes, MiscTechs, Tools, NewCollabTools)
            st.write(f'You Can Work As {target_encoder.inverse_transform([res])[0]}')

Main()
