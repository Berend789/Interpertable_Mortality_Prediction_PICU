from all_own_functions import cnfl, value_filtering,growth
import typing
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
conv_dict={'pat_weight_adm':cnfl,'pat_weight_act':cnfl,'lab_bg_be':cnfl,'lab_bg_hco3':cnfl,'lab_bg_pco2':cnfl,'lab_bg_ph':cnfl,'lab_bg_po2':cnfl,'lab_bg_sat':cnfl,
'lab_bl_b2m':cnfl,'lab_bl_bil_d':cnfl,'lab_bl_bil_i':cnfl,'lab_bl_ca2':cnfl,'lab_bl_catot':cnfl,'lab_bl_cc':cnfl,'lab_bl_cl':cnfl,'lab_bl_cr':cnfl,'lab_bl_CRP':cnfl,
'lab_bl_f':cnfl,'lab_bl_gluc':cnfl,'lab_bl_hb': cnfl,'lab_bl_ht': cnfl,'lab_bl_k': cnfl, 'lab_bl_lactate':cnfl,'lab_bl_leuco':cnfl,'lab_bl_mg':cnfl,'lab_bl_na':cnfl,
'lab_bl_tr': cnfl, 'lab_bl_ur': cnfl, 'lab_u_alb_cr':cnfl, 'lab_u_b2m':cnfl,'lab_u_ca':cnfl,'lab_u_cr':cnfl,'lab_u_f':cnfl,'lab_u_gluc':cnfl,'lab_u_k':cnfl,'lab_u_mg':cnfl,
'lab_u_na':cnfl,'lab_u_ph':cnfl,'lab_u_pr':cnfl,'lab_u_pr_cr':cnfl,'lab_u_ur':cnfl,'lab_u_vol':cnfl,'mon_temp':cnfl,'mon_temp_skin':cnfl,'pat_length_adm':cnfl,
'pat_length_act': cnfl,'mon_etco2':cnfl,'vent_m_fio2':cnfl,'vent_m_no':cnfl,'vent_m_peep':cnfl,'vent_m_ppeak':cnfl,'mon_ibp_dia':cnfl,'mon_ibp_sys':cnfl,'mon_hr':cnfl,
'mon_ibp_mean':cnfl,'mon_nibp_dia':cnfl,'mon_nibp_sys':cnfl,'mon_nibp_mean':cnfl,'mon_rr':cnfl,'mon_sat':cnfl,'vent_set_fiO2':cnfl,'vent_set_p_insp':cnfl,'vent_set_peep':cnfl,
'vent_set_rr':cnfl,'vent_set_tv':cnfl,'vent_set_upl':cnfl,'vent_tube':cnfl,'vent_m_pplat':cnfl,'vent_m_rr':cnfl,'vent_m_tv_exp':cnfl,'vent_m_tv_insp':cnfl,'lab_bg_mode':cnfl}

dtype_dict={'pat_hosp_id': str,'pat_sex':"category",'lab_bg_origin':"category",
'lab_u_m':"category",'mon_temp_mod':"category", 'obs_pup_dia':"category",'obs_pup_light':"category",'vent_cat':"category",
'vent_fio2_mod':"category",
'vent_machine':"category",'vent_mode':"category",'Row_NUM':str,"Status": 'category','mon_hr':'np.float64','mon_rr':'np.float64'}

all_cols = [ 'pat_hosp_id','pat_bd','pat_datetime','lab_bg_be', 'lab_bg_hco3',
        'lab_bg_pco2', 'lab_bg_ph',
       'lab_bg_po2', 'lab_bg_sat', 'lab_bl_b2m', 'lab_bl_bil_d',
       'lab_bl_bil_i', 'lab_bl_ca2', 'lab_bl_catot', 'lab_bl_cc', 'lab_bl_cl',
       'lab_bl_cr', 'lab_bl_CRP', 'lab_bl_f', 'lab_bl_gluc', 'lab_bl_hb',
       'lab_bl_ht', 'lab_bl_k', 'lab_bl_lactate', 'lab_bl_leuco', 'lab_bl_mg',
       'lab_bl_na', 'lab_bl_tr', 'lab_bl_ur', 'mon_etco2','mon_hr',
       'mon_ibp_dia', 'mon_ibp_mean', 'mon_ibp_sys', 'mon_nibp_dia',
       'mon_nibp_mean', 'mon_nibp_sys', 'mon_rr', 'mon_sat', 'mon_temp', 'mon_temp_skin',
       'vent_m_fio2', 'vent_m_no', 'vent_m_peep', 'vent_m_ppeak',
       'vent_m_pplat', 'vent_m_rr', 'vent_m_tv_exp', 'vent_m_tv_insp', 'vent_mode', 'vent_tube','Status']

vent_cols=[ 'pat_hosp_id','pat_bd','pat_datetime','OK_datum',
       'vent_m_fio2', 'vent_m_no', 'vent_m_peep', 'vent_m_ppeak',
       'vent_m_pplat', 'vent_m_rr', 'vent_m_tv_exp', 'vent_m_tv_insp', 'vent_mode', 'vent_tube']

