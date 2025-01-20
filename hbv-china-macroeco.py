

import json
import pandas as pd
import atomica as at
import copy
import re
from matplotlib import pyplot as plt
import numpy as np
import os

def save_to_excel(data, file_path):
    # Ensure that the directory of the file path exists, and if it does not exist, create it.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with pd.ExcelWriter(file_path) as writer:
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                df = pd.DataFrame(value)
                df.to_excel(writer, sheet_name=key, index=False)
            else:
                df = pd.DataFrame(value)
                df.to_excel(writer, sheet_name=key, index=False)

def compute_ci(data):
    lower = np.percentile(data, 2.5, axis=1)
    center = np.percentile(data, 50, axis=1)
    upper = np.percentile(data, 97.5, axis=1)
    return lower, center, upper
# Run the model and conduct uncertainty analysis.
def model_results(F, data, calib, res_name, runs):

    import atomica as at
    import numpy as np

    # Run the model
    D = at.ProjectData.from_spreadsheet("databooks/" + data, framework=F)
    P = at.Project(framework=F, databook="databooks/" + data, sim_start=1990, sim_end=2051, sim_dt=0.25, do_run=False)
    cal = P.make_parset()
    cal.load_calibration("calibrations/" + calib)
    # central estimation
    res = P.run_sim(parset=cal, result_name=res_name)

    # PSA
    np.random.seed(25012024)
    psa = P.parsets[0]
    psa_res = P.run_sampled_sims(cal, n_samples=runs)
    # Initialize a dictionary to store the results
    central_est = {}
    store_runs = {}
    ci_95_upper = {}
    ci_95_lower = {}
    ci_95_cent = {}
    out_res = [
               {"pop_0_4M": ["alive", {"0-4M": ["0-4M"]}, "sum"]},
               {"pop_5_14M": ["alive", {"5-14M": ["5-14M"]}, "sum"]},
               {"pop_15_49M": ["alive", {"15-49M": ["15-49M"]}, "sum"]},
               {"pop_50_59M": ["alive", {"50-59M": ["50-59M"]}, "sum"]},
               {"pop_60M": ["alive", {"60+M": ["60+M"]}, "sum"]},

               {"pop_0_4F": ["alive", {"0-4M": ["0-4F"]}, "sum"]},
               {"pop_5_14F": ["alive", {"5-14M": ["5-14F"]}, "sum"]},
               {"pop_15_49F": ["alive", {"15-49F": ["15-49F"]}, "sum"]},
               {"pop_50_59F": ["alive", {"50-59F": ["50-59F"]}, "sum"]},
               {"pop_60F": ["alive", {"60+F": ["60+F"]}, "sum"]},

               {"yld_0_4M": ["yld", {"0-4M": ["0-4M"]}, "sum"]},
               {"yld_5_14M": ["yld", {"5-14M": ["5-14M"]}, "sum"]},
               {"yld_15_49M": ["yld", {"15-49M": ["15-49M"]}, "sum"]},
               {"yld_50_59M": ["yld", {"50-59M": ["50-59M"]}, "sum"]},
               {"yld_60M": ["yld", {"60+M": ["60+M"]}, "sum"]},

               {"yld_0_4F": ["yld", {"0-4F": ["0-4F"]}, "sum"]},
               {"yld_5_14F": ["yld", {"5-14F": ["5-14F"]}, "sum"]},
               {"yld_15_49F": ["yld", {"15-49F": ["15-49F"]}, "sum"]},
               {"yld_50_59F": ["yld", {"50-59F": ["50-59F"]}, "sum"]},
               {"yld_60F": ["yld", {"60+F": ["60+F"]}, "sum"]},

               {"yll_0_4M": ["yll", {"0-4M": ["0-4M"]}, "sum"]},
               {"yll_5_14M": ["yll", {"5-14M": ["5-14M"]}, "sum"]},
               {"yll_15_49M": ["yll", {"15-49M": ["15-49M"]}, "sum"]},
               {"yll_50_59M": ["yll", {"50-59M": ["50-59M"]}, "sum"]},
               {"yll_60M": ["yll", {"60+M": ["60+M"]}, "sum"]},

               {"yll_0_4F": ["yll", {"0-4F": ["0-4F"]}, "sum"]},
               {"yll_5_14F": ["yll", {"5-14F": ["5-14F"]}, "sum"]},
               {"yll_15_49F": ["yll", {"15-49F": ["15-49F"]}, "sum"]},
               {"yll_50_59F": ["yll", {"50-59F": ["50-59F"]}, "sum"]},
               {"yll_60F": ["yll", {"60+F": ["60+F"]}, "sum"]},

                {"mort_0_4M": [":dd_hbv", {"0-4M": ["0-4M"]}, "sum"]},
                {"mort_5_14M": [":dd_hbv", {"5-14M": ["5-14M"]}, "sum"]},
                {"mort_15_49M": [":dd_hbv", {"15-49M": ["15-49M"]}, "sum"]},
                {"mort_50_59M": [":dd_hbv", {"50-59M": ["50-59M"]}, "sum"]},
                {"mort_60M": [":dd_hbv", {"60+M": ["60+M"]}, "sum"]},

                {"mort_0_4F": [":dd_hbv", {"0-4F": ["0-4F"]}, "sum"]},
                {"mort_5_14F": [":dd_hbv", {"5-14F": ["5-14F"]}, "sum"]},
                {"mort_15_49F": [":dd_hbv", {"15-49F": ["15-49F"]}, "sum"]},
                {"mort_50_59F": [":dd_hbv", {"50-59F": ["50-59F"]}, "sum"]},
                {"mort_60F": [":dd_hbv", {"60+F": ["60+F"]}, "sum"]},

                {"yll": ["yll", "total", "sum"]},
                {"yld": ["yld", "total", "sum"]},
                {"pop": ["alive", "total", "sum"]},
                {"prev": ["prev", "total", "weighted"]},
                {"chb_pop": ["chb_pop", "total", "weighted"]},
                {"prev_u5": ["prev", {"Under 5": ["0-4M", "0-4F"]}, "weighted"]},
                {"mort": [":dd_hbv", "total", "sum"]},
                {"hcc_inc": ["flw_hcc", "total", "sum"]},
                {"chb_inc": ["tot_inc", "total", "sum"]},
                {"hbe_preg": ["eag_ott", "15-49F", "sum"]},
                {"births": ["b_rate", {"Under 5": ["0-4M", "0-4F"]}, "sum"]},
                {"bd_cov": ["bd", {"Under 5": ["0-4M", "0-4F"]}, "weighted"]},
                {"hb3_cov": ["hb3", {"Under 5": ["0-4M", "0-4F"]}, "weighted"]},
                {"dx_rate": ["tot_dx", "total", "sum"]},
                {"tx_cov": ["treat", "total", "sum"]},
                {"dm_dx": [[{"dm_dx": "it_dx+icl_dx+ict_dx+ie_dx+cc_dx+dc_dx+hcc_dx"}], "total", "sum"]},
                {"dm_tx": [[{"dm_tx": "icl_tx+ict_tx+ie_tx+cc_tx+dc_tx+hcc_tx"}], "total", "sum"]},
                {"pop_hcc": [[{"pop_hcc": "cc_dx+cc_tx+dc_dx+dc_tx"}], "total", "sum"]},
                {"tgt_hcc": [[{"tgt_hcc": "it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"50+": ["50-59M", "50-59F", "60+M", "60+F"]}, "sum"]},
                {"tgt_hcc_b": [[{"tgt_hcc": "it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"40+": ["15-49M", "15-49F"]}, "sum"]},
                {"hsp_tx": [[{"hsp_tx": "dc_tx+hcc_tx"}], "total", "sum"]},
                {"hsp_utx": [[{"hsp_utx": "dc+dc_dx+hcc+hcc_dx"}], "total", "sum"]},
                {"dx_prop": ["diag_cov", "total", "weighted"]},
                {"tx_prop": ["treat_cov", "total", "weighted"]},
                {"mav_n": ["mav_births", {"Under 5": ["0-4M", "0-4F"]}, "sum"]},
                {"prg_scr": ["preg_scr_num", "15-49F", "sum"]},
                {"prg_hrs": ["preg_scr_num", "15-49F", "sum"]}]

    for i in out_res:
        for key, val in i.items():
            df = at.PlotData(res, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1).series[0].vals
            central_est[key] = df

    for output in out_res:
        for key, val in output.items():
            mapping_function = lambda x: at.PlotData(x, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1)
            ensemble = at.Ensemble(mapping_function=mapping_function)
            ensemble.update(psa_res)
            df = pd.DataFrame([d.series[0].vals for d in ensemble.samples])
            store_runs[key] = np.array(df).T

            ci_95_lower[key] = np.percentile(store_runs[key], 2.5, axis=1)
            ci_95_cent[key] = np.percentile(store_runs[key], 50, axis=1)
            ci_95_upper[key] = np.percentile(store_runs[key], 97.5, axis=1)

    return store_runs, central_est


# Economic Calculation - Refer to the economics section code in ChrisSeaman-Burnet/hbvglobalinvcase on GitHub
def econ_analysis(cent, cent_s1, cent_s2, cent_s3, res, res_s1, res_s2, res_s3, cost_data, runs, cdisc_rate, hdisc_rate):
    # 创建折现率数组
    if cdisc_rate > 1:
        cdisc_rate = cdisc_rate / 100
    else:
        cdisc_rate = cdisc_rate

    if hdisc_rate > 1:
        hdisc_rate = hdisc_rate / 100
    else:
        hdisc_rate = hdisc_rate

    discount = np.zeros((len(np.arange(1990, 2051, 1)), 4))
    discount[:, 0] = np.arange(1990, 2051, 1)

    for idx, val in enumerate(discount[:, 0]):
        if val <= 2023:
            discount[idx, 1] = 1  # 消费折现
            discount[idx, 2] = 0
            discount[idx, 3] = 1  # 健康折现
        else:
            discount[idx, 1:3] = (1 + cdisc_rate) ** - (val - 2023)
            discount[idx, 3] = (1 + hdisc_rate) ** - (val - 2023)

    # Vaccine cost
    vax_costs = pd.read_excel("costs.xlsx", sheet_name="vax")
    np.random.seed(25012024)

    bd_vax = vax_costs.loc[0, 'China']
    hb3_vax = vax_costs.loc[1, 'China']

    bd_vax_samp = np.random.triangular(vax_costs.loc[6, 'China'], vax_costs.loc[0, 'China'], vax_costs.loc[7, 'China'], runs)
    hb3_vax_samp = np.random.triangular(vax_costs.loc[8, 'China'], vax_costs.loc[1, 'China'], vax_costs.loc[9, 'China'], runs)

    cbd_cost_bl, chb3_cost_bl, cbd_cost_s1, chb3_cost_s1, cbd_cost_s2, chb3_cost_s2, cbd_cost_s3, chb3_cost_s3 = np.zeros(
        (61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros(
        (61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    bd_cost_bl, hb3_cost_bl, bd_cost_s1, hb3_cost_s1, bd_cost_s2, hb3_cost_s2, bd_cost_s3, hb3_cost_s3 = np.zeros(
        (61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros(
        (61, runs)), np.zeros((61, runs)), np.zeros((61, runs))

    cbd_cost_bl[:, 0] = cent["bd_cov"][:] * cent["births"][:] * bd_vax * discount[:, 1]
    chb3_cost_bl[:, 0] = cent["hb3_cov"][:] * cent["births"][:] * hb3_vax * discount[:, 1]
    cbd_cost_s1[:, 0] = cent_s1["bd_cov"][:] * cent_s1["births"][:] * bd_vax * discount[:, 1]
    chb3_cost_s1[:, 0] = cent_s1["hb3_cov"][:] * cent_s1["births"][:] * hb3_vax * discount[:, 1]
    cbd_cost_s2[:, 0] = cent_s2["bd_cov"][:] * cent_s2["births"][:] * bd_vax * discount[:, 1]
    chb3_cost_s2[:, 0] = cent_s2["hb3_cov"][:] * cent_s2["births"][:] * hb3_vax * discount[:, 1]
    cbd_cost_s3[:, 0] = cent_s3["bd_cov"][:] * cent_s3["births"][:] * bd_vax * discount[:, 1]
    chb3_cost_s3[:, 0] = cent_s3["hb3_cov"][:] * cent_s3["births"][:] * hb3_vax * discount[:, 1]

    for run in range(runs):
        bd_cost_bl[:, run] = res["bd_cov"][:, run] * res["births"][:, run] * bd_vax_samp[run] * discount[:, 1]
        hb3_cost_bl[:, run] = res["hb3_cov"][:, run] * res["births"][:, run] * hb3_vax_samp[run] * discount[:, 1]
        bd_cost_s1[:, run] = res_s1["bd_cov"][:, run] * res_s1["births"][:, run] * bd_vax_samp[run] * discount[:, 1]
        hb3_cost_s1[:, run] = res_s1["hb3_cov"][:, run] * res_s1["births"][:, run] * hb3_vax_samp[run] * discount[:, 1]
        bd_cost_s2[:, run] = res_s2["bd_cov"][:, run] * res_s2["births"][:, run] * bd_vax_samp[run] * discount[:, 1]
        hb3_cost_s2[:, run] = res_s2["hb3_cov"][:, run] * res_s2["births"][:, run] * hb3_vax_samp[run] * discount[:, 1]
        bd_cost_s3[:, run] = res_s3["bd_cov"][:, run] * res_s3["births"][:, run] * bd_vax_samp[run] * discount[:, 1]
        hb3_cost_s3[:, run] = res_s3["hb3_cov"][:, run] * res_s3["births"][:, run] * hb3_vax_samp[run] * discount[:, 1]

    care_costs = pd.read_excel(cost_data, sheet_name="care")
    dx_cost = care_costs.loc[0, 'China']
    hrs_cost = vax_costs.loc[10, 'China']
    mav_hbig = vax_costs.loc[12, 'China']

    # Maternal and child blocking plus screening cost (including HBIG)
    cmsc_cost_bl, cmsc_cost_s1, cmsc_cost_s2, cmsc_cost_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    msc_cost_bl, msc_cost_s1, msc_cost_s2, msc_cost_s3 = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))

    cmsc_cost_bl[:, 0] =  (cent["mav_n"][:] * mav_hbig * discount[:, 1]) + (cent["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent["prg_hrs"][:] * hrs_cost * discount[:, 2])  # 中国基线有母婴阻断，有改动
    cmsc_cost_s1[:, 0] = (cent_s1["mav_n"][:] * mav_hbig * discount[:, 1]) + (cent_s1["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost * discount[:, 2]) #假设后面的情景在2022年后检测e抗原
    cmsc_cost_s2[:, 0] = (cent_s2["mav_n"][:] * mav_hbig * discount[:, 1]) + (cent_s2["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s2["prg_hrs"][:] * hrs_cost * discount[:, 2])
    cmsc_cost_s3[:, 0] = (cent_s3["mav_n"][:] * mav_hbig * discount[:, 1]) + (cent_s3["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s3["prg_hrs"][:] * hrs_cost * discount[:, 2])

    for run in range(runs):
        msc_cost_bl[:, run] = (res["mav_n"][:, run] * mav_hbig * discount[:, 1]) + (res["prg_scr"][:, run] * dx_cost * discount[:, 1]) + (res["prg_hrs"][:, run] * hrs_cost * discount[:, 2]) # 中国基线有母婴阻断，有改动
        msc_cost_s1[:, run] = (res_s1["mav_n"][:, run] * mav_hbig * discount[:, 1]) + (res_s1["prg_scr"][:, run] * dx_cost * discount[:, 1]) + (res_s1["prg_hrs"][:, run] * hrs_cost * discount[:, 2])
        msc_cost_s2[:, run] = (res_s2["mav_n"][:, run] * mav_hbig * discount[:, 1]) + (res_s2["prg_scr"][:, run] * dx_cost * discount[:, 1]) + (res_s2["prg_hrs"][:, run] * hrs_cost * discount[:, 2])
        msc_cost_s3[:, run] = (res_s3["mav_n"][:, run] * mav_hbig * discount[:, 1]) + (res_s3["prg_scr"][:, run] * dx_cost * discount[:, 1]) + (res_s3["prg_hrs"][:, run] * hrs_cost * discount[:, 2])

    ## Diagnostic Cost: Calculate the cost based on the proportion of negative diagnoses. The increase in diagnosis is referred to as "dx_inc".
    cdx_cost_bl, cdx_cost_s1, cdx_cost_s2, cdx_cost_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    cbl_dx_inc, cs1_dx_inc, cs2_dx_inc, cs3_dx_inc = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    bl_dx_inc, s1_dx_inc, s2_dx_inc, s3_dx_inc = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))

    for i in range(len(res["dx_prop"])):
        if i < 1:
            cbl_dx_inc[i, 0] = cent["dx_prop"][i]
            cs1_dx_inc[i, 0] = cent_s1["dx_prop"][i]
            cs2_dx_inc[i, 0] = cent_s2["dx_prop"][i]
            cs3_dx_inc[i, 0] = cent_s3["dx_prop"][i]
        else:
            cbl_dx_inc[i, 0] = max(cent["dx_prop"][i] - cent["dx_prop"][i - 1], 0)
            cs1_dx_inc[i, 0] = max(cent_s1["dx_prop"][i] - cent_s1["dx_prop"][i - 1], 0)
            cs2_dx_inc[i, 0] = max(cent_s2["dx_prop"][i] - cent_s2["dx_prop"][i - 1], 0)
            cs3_dx_inc[i, 0] = max(cent_s3["dx_prop"][i] - cent_s3["dx_prop"][i - 1], 0)

    for i in range(len(res["dx_prop"])):
        for run in range(runs):
            if i < 1:
                bl_dx_inc[i, run] = res["dx_prop"][i, run]
                s1_dx_inc[i, run] = res_s1["dx_prop"][i, run]
                s2_dx_inc[i, run] = res_s2["dx_prop"][i, run]
                s3_dx_inc[i, run] = res_s3["dx_prop"][i, run]
            else:
                bl_dx_inc[i, run] = max(res["dx_prop"][i, run] - res["dx_prop"][i - 1, run], 0)
                s1_dx_inc[i, run] = max(res_s1["dx_prop"][i, run] - res_s1["dx_prop"][i - 1, run], 0)
                s2_dx_inc[i, run] = max(res_s2["dx_prop"][i, run] - res_s2["dx_prop"][i - 1, run], 0)
                s3_dx_inc[i, run] = max(res_s3["dx_prop"][i, run] - res_s3["dx_prop"][i - 1, run], 0)

    cdx_costb_bl, cdx_costb_s1, cdx_costb_s2, cdx_costb_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    dx_costb_bl, dx_costb_s1, dx_costb_s2, dx_costb_s3 = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))

    for yr in range(len(cdx_costb_bl)):
        cdx_costb_bl[yr, 0] = (cent["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * cbl_dx_inc[yr, 0] * (cent["pop"][yr] - cent["dx_rate"][yr]) * discount[yr, 1])
        cdx_costb_s1[yr, 0] = (cent_s1["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * cs1_dx_inc[yr, 0] * (cent_s1["pop"][yr] - cent_s1["dx_rate"][yr]) * discount[yr, 1])
        cdx_costb_s2[yr, 0] = (cent_s2["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * cs2_dx_inc[yr, 0] * (cent_s2["pop"][yr] - cent_s2["dx_rate"][yr]) * discount[yr, 1])
        cdx_costb_s3[yr, 0] = (cent_s3["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * cs3_dx_inc[yr, 0] * (cent_s3["pop"][yr] - cent_s3["dx_rate"][yr]) * discount[yr, 1])

    for run in range(runs):
        for yr in range(len(dx_costb_bl)):
            dx_costb_bl[yr, run] = (res["dx_rate"][yr, run] * dx_cost * discount[yr, 1]) + (dx_cost * bl_dx_inc[yr, run] * (res["pop"][yr, run] - res["dx_rate"][yr, run]) * discount[yr, 1])
            dx_costb_s1[yr, run] = (res_s1["dx_rate"][yr, run] * dx_cost * discount[yr, 1]) + (dx_cost * s1_dx_inc[yr, run] * (res_s1["pop"][yr, run] - res_s1["dx_rate"][yr, run]) * discount[yr, 1])
            dx_costb_s2[yr, run] = (res_s2["dx_rate"][yr, run] * dx_cost * discount[yr, 1]) + (dx_cost * s2_dx_inc[yr, run] * (res_s2["pop"][yr, run] - res_s2["dx_rate"][yr, run]) * discount[yr, 1])
            dx_costb_s3[yr, run] = (res_s3["dx_rate"][yr, run] * dx_cost * discount[yr, 1]) + (dx_cost * s3_dx_inc[yr, run] * (res_s3["pop"][yr, run] - res_s3["dx_rate"][yr, run]) * discount[yr, 1])

    # Treatment costs(TDF)
    tx_cost = care_costs.loc[3, 'China']


    ctx_cost_bl, ctx_cost_s1, ctx_cost_s2, ctx_cost_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    tx_cost_bl, tx_cost_s1, tx_cost_s2, tx_cost_s3 = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))

    ctx_cost_bl[:, 0] = cent["tx_cov"][:].astype(float) * tx_cost * discount[:, 1]
    ctx_cost_s1[:, 0] = cent_s1["tx_cov"][:].astype(float) * tx_cost * discount[:, 1]
    ctx_cost_s2[:, 0] = cent_s2["tx_cov"][:].astype(float) * tx_cost * discount[:, 1]
    ctx_cost_s3[:, 0] = cent_s3["tx_cov"][:].astype(float) * tx_cost * discount[:, 1]


    for run in range(runs):
        tx_cost_bl[:, run] = res["tx_cov"][:, run] * tx_cost * discount[:, 1]
        tx_cost_s1[:, run] = res_s1["tx_cov"][:, run] * tx_cost * discount[:, 1]
        tx_cost_s2[:, run] = res_s2["tx_cov"][:, run] * tx_cost * discount[:, 1]
        tx_cost_s3[:, run] = res_s3["tx_cov"][:, run] * tx_cost * discount[:, 1]

    ## Diagnostic Cost (Point Estimation)
    dx_cent_bl = cdx_cost_bl + cmsc_cost_bl
    dx_cent_s1 = cdx_cost_s1 + cmsc_cost_s1
    dx_cent_s2 = cdx_cost_s2 + cmsc_cost_s2
    dx_cent_s3 = cdx_cost_s3 + cmsc_cost_s3

    ## Vaccine Cost (Point Estimation)
    vax_cent_bl = cbd_cost_bl + chb3_cost_bl
    vax_cent_s1 = cbd_cost_s1 + chb3_cost_s1
    vax_cent_s2 = cbd_cost_s2 + chb3_cost_s2
    vax_cent_s3 = cbd_cost_s3 + chb3_cost_s3

    ## Direct Medical Cost (Point Estimation)
    dmc_cent_bl = cbd_cost_bl + chb3_cost_bl + cdx_costb_bl + ctx_cost_bl + cmsc_cost_bl
    dmc_cent_s1 = cbd_cost_s1 + chb3_cost_s1 + cdx_costb_s1 + ctx_cost_s1 + cmsc_cost_s1
    dmc_cent_s2 = cbd_cost_s2 + chb3_cost_s2 + cdx_costb_s2 + ctx_cost_s2 + cmsc_cost_s2
    dmc_cent_s3 = cbd_cost_s3 + chb3_cost_s3 + cdx_costb_s3 + ctx_cost_s3 + cmsc_cost_s3

    ## Direct Medical Cost (PSA)
    dmc_psa_bl = bd_cost_bl + hb3_cost_bl + dx_costb_bl + tx_cost_bl + msc_cost_bl
    dmc_psa_s1 = bd_cost_s1 + hb3_cost_s1 + dx_costb_s1 + tx_cost_s1 + msc_cost_s1
    dmc_psa_s2 = bd_cost_s2 + hb3_cost_s2 + dx_costb_s2 + tx_cost_s2 + msc_cost_s2
    dmc_psa_s3 = bd_cost_s3 + hb3_cost_s3 + dx_costb_s3 + tx_cost_s3 + msc_cost_s3

    ## Disease management cost
    util = 0.25
    tx_hosp = 0.5

    ## Indirect Disease Management Cost
    dx_dmc = care_costs.loc[4, 'China']
    tx_dmc = care_costs.loc[5, 'China']
    hosp_cost = care_costs.loc[7, 'China']
    hcc_cost = care_costs.loc[6, 'China']
    hcc_prp = care_costs.loc[8, 'China']

    cmc_cost_bl, cmc_cost_s1, cmc_cost_s2, cmc_cost_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    chosp_cost_bl, chosp_cost_s1, chosp_cost_s2, chosp_cost_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    chcc_cost_bl, chcc_cost_s1, chcc_cost_s2, chcc_cost_s3 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    ## Disease Management Cost（diagnose）
    cmc_cost_bl[:, 0] = ((cent["dm_dx"][:] * dx_dmc * util) + (cent["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]
    cmc_cost_s1[:, 0] = ((cent_s1["dm_dx"][:] * dx_dmc * util) + (cent_s1["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]
    cmc_cost_s2[:, 0] = ((cent_s2["dm_dx"][:] * dx_dmc * util) + (cent_s2["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]
    cmc_cost_s3[:, 0] = ((cent_s3["dm_dx"][:] * dx_dmc * util) + (cent_s3["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]

    ## Hospitalization Cost
    chosp_cost_bl[:, 0] = ((cent["hsp_utx"][:] * hosp_cost * util) + (cent["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    chosp_cost_s1[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * util) + (cent_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    chosp_cost_s2[:, 0] = ((cent_s2["hsp_utx"][:] * hosp_cost * util) + (cent_s2["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    chosp_cost_s3[:, 0] = ((cent_s3["hsp_utx"][:] * hosp_cost * util) + (cent_s3["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]

    ## HCC Monitoring Cost (assume 50% HR covered under UHC; 25% resource utilization)
    chcc_cost_bl[:, 0] = ((cent["pop_hcc"][:] * hcc_cost * util) + (cent["tgt_hcc"][:] * hcc_cost * util) + (cent["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]
    chcc_cost_s1[:, 0] = ((cent_s1["pop_hcc"][:] * hcc_cost * util) + (cent_s1["tgt_hcc"][:] * hcc_cost * util) + (cent_s1["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]
    chcc_cost_s2[:, 0] = ((cent_s2["pop_hcc"][:] * hcc_cost * util) + (cent_s2["tgt_hcc"][:] * hcc_cost * util) + (cent_s2["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]
    chcc_cost_s3[:, 0] = ((cent_s3["pop_hcc"][:] * hcc_cost * util) + (cent_s3["tgt_hcc"][:] * hcc_cost * util) + (cent_s3["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]

    mc_cost_bl, mc_cost_s1, mc_cost_s2, mc_cost_s3 = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))
    hosp_cost_bl, hosp_cost_s1, hosp_cost_s2, hosp_cost_s3 = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))
    hcc_cost_bl, hcc_cost_s1, hcc_cost_s2, hcc_cost_s3 = np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs)), np.zeros((61, runs))

    for run in range(runs):
        ## Disease Management Cost（diagnose）
        mc_cost_bl[:, run] = ((res["dm_dx"][:, run] * dx_dmc * util) + (res["dm_tx"][:, run] * tx_dmc * util)) * discount[:, 1]
        mc_cost_s1[:, run] = ((res_s1["dm_dx"][:, run] * dx_dmc * util) + (res_s1["dm_tx"][:, run] * tx_dmc * util)) * discount[:, 1]
        mc_cost_s2[:, run] = ((res_s2["dm_dx"][:, run] * dx_dmc * util) + (res_s2["dm_tx"][:, run] * tx_dmc * util)) * discount[:, 1]
        mc_cost_s3[:, run] = ((res_s3["dm_dx"][:, run] * dx_dmc * util) + (res_s3["dm_tx"][:, run] * tx_dmc * util)) * discount[:, 1]
        ## Hospitalization Cost
        hosp_cost_bl[:, run] = ((res["hsp_utx"][:, run] * hosp_cost * util) + (res["hsp_tx"][:, run] * hosp_cost * util * tx_hosp)) * discount[:, 1]
        hosp_cost_s1[:, run] = ((res_s1["hsp_utx"][:, run] * hosp_cost * util) + (res_s1["hsp_tx"][:, run] * hosp_cost * util * tx_hosp)) * discount[:, 1]
        hosp_cost_s2[:, run] = ((res_s2["hsp_utx"][:, run] * hosp_cost * util) + (res_s2["hsp_tx"][:, run] * hosp_cost * util * tx_hosp)) * discount[:, 1]
        hosp_cost_s3[:, run] = ((res_s3["hsp_utx"][:, run] * hosp_cost * util) + (res_s3["hsp_tx"][:, run] * hosp_cost * util * tx_hosp)) * discount[:, 1]
        ## HCC Monitoring Cost
        hcc_cost_bl[:, run] = ((res["pop_hcc"][:, run] * hcc_cost * util) + (res["tgt_hcc"][:, run] * hcc_cost * util) + (res["tgt_hcc_b"][:, run] * hcc_prp * hcc_cost * util)) * discount[:, 1]
        hcc_cost_s1[:, run] = ((res_s1["pop_hcc"][:, run] * hcc_cost * util) + (res_s1["tgt_hcc"][:, run] * hcc_cost * util) + (res_s1["tgt_hcc_b"][:, run] * hcc_prp * hcc_cost * util)) * discount[:, 1]
        hcc_cost_s2[:, run] = ((res_s2["pop_hcc"][:, run] * hcc_cost * util) + (res_s2["tgt_hcc"][:, run] * hcc_cost * util) + (res_s2["tgt_hcc_b"][:, run] * hcc_prp * hcc_cost * util)) * discount[:, 1]
        hcc_cost_s3[:, run] = ((res_s3["pop_hcc"][:, run] * hcc_cost * util) + (res_s3["tgt_hcc"][:, run] * hcc_cost * util) + (res_s3["tgt_hcc_b"][:, run] * hcc_prp * hcc_cost * util)) * discount[:, 1]

    ## Indirect Disease Management Cost(Point estimated)
    imc_cent_bl = cmc_cost_bl + chosp_cost_bl + chcc_cost_bl
    imc_cent_s1 = cmc_cost_s1 + chosp_cost_s1 + chcc_cost_s1
    imc_cent_s2 = cmc_cost_s2 + chosp_cost_s2 + chcc_cost_s2
    imc_cent_s3 = cmc_cost_s3 + chosp_cost_s3 + chcc_cost_s3

    ## Indirect Disease Management Cost (discounted; PSA)
    imc_psa_bl = mc_cost_bl + hosp_cost_bl + hcc_cost_bl
    imc_psa_s1 = mc_cost_s1 + hosp_cost_s1 + hcc_cost_s1
    imc_psa_s2 = mc_cost_s2 + hosp_cost_s2 + hcc_cost_s2
    imc_psa_s3 = mc_cost_s3 + hosp_cost_s3 + hcc_cost_s3

    ## Productive loss
    prod_costs = pd.read_excel(cost_data, sheet_name="emp_gdp_lex")
    etp_ratio = prod_costs.loc[0,'China']
    gdp = prod_costs.loc[1, 'China']
    life_exp = prod_costs.loc[2, 'China']

    ## GDP grow
    gdp_grw = np.zeros((len(np.arange(1990, 2051, 1)), 4))
    gdp_grw[:, 0] = np.arange(1990, 2051, 1)
    gdp_grw[:, 1:4] = gdp

    for i, val in enumerate(gdp_grw[:, 0]):
        if val > 2023:
            gdp_grw[i, 1] = gdp_grw[i - 1, 1] * 1.00
            gdp_grw[i, 2] = gdp_grw[i - 1, 2] * 1.015
            gdp_grw[i, 3] = gdp_grw[i - 1, 3] * 1.03


    age_of_deaths = np.array([0.01, 0.031, 0.253, 0.341, 0.365])
    prop_leaving_age_categories = np.array([1 / 15, 1 / 15, 1 / 20, 1 / 15])
    all_cause_mort = np.array([0.003, 0.0013, 0.0022, 0.0103, (1 / life_exp)])

    #===================
    ## Baseline Mortality Distribution(Point estimated)
    cbl_deaths = np.zeros((len(cent["mort"]), 2))
    cbl_deaths[:, 0] = np.arange(1990, 2051, 1)
    cbl_deaths[:, 1] = cent["mort"]

    for idx, val in enumerate(cbl_deaths[:, 0]):
        if val < 2023:
            cbl_deaths[idx, 1] = 0  # only the impacts after 2023 are of concern.

    ghosts_cbl = np.zeros((len(cbl_deaths), len(age_of_deaths)))
    ghosts_cbl[0, :] = cbl_deaths[0, 1] * age_of_deaths # Mortality distribution by age group

    for t in range(1, len(cbl_deaths)):
        ppl_who_age = ghosts_cbl[t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
        ghosts_cbl[t, 0] = max(0, ghosts_cbl[t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_cbl[t - 1, 0])
        ghosts_cbl[t, 1] = max(0, ghosts_cbl[t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_cbl[t - 1, 1])
        ghosts_cbl[t, 2] = max(0, ghosts_cbl[t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_cbl[t - 1, 2])
        ghosts_cbl[t, 3] = max(0, ghosts_cbl[t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_cbl[t - 1, 3])
        ghosts_cbl[t, 4] = max(0, ghosts_cbl[t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_cbl[t - 1, 4])

        ghosts_cbl[t, :] = ghosts_cbl[t, :] + cbl_deaths[t, 1] * age_of_deaths

     ## Baseline Mortality Distribution(PSA)
    bl_deaths = np.zeros((len(res["mort"]), runs + 1))
    bl_deaths[:, 0] = np.arange(1990, 2051, 1)
    bl_deaths[:, 1:] = res["mort"]

    for idx, val in enumerate(bl_deaths[:, 0]):
        if val < 2023:
            bl_deaths[idx, 1:] = 0

    ghosts_bl = np.zeros((runs, len(bl_deaths), len(age_of_deaths)))

    for run in range(runs):
        ghosts_bl[run, 0, :] = bl_deaths[0, run + 1] * age_of_deaths

    for run in range(runs):
        for t in range(1, len(bl_deaths)):
            ppl_who_age = ghosts_bl[run, t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
            ghosts_bl[run, t, 0] = max(0, ghosts_bl[run, t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_bl[run, t - 1, 0])
            ghosts_bl[run, t, 1] = max(0, ghosts_bl[run, t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_bl[run, t - 1, 1])
            ghosts_bl[run, t, 2] = max(0, ghosts_bl[run, t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_bl[run, t - 1, 2])
            ghosts_bl[run, t, 3] = max(0, ghosts_bl[run, t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_bl[run, t - 1, 3])
            ghosts_bl[run, t, 4] = max(0, ghosts_bl[run, t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_bl[run, t - 1, 4])

            ghosts_bl[run, t, :] = ghosts_bl[run, t, :] + bl_deaths[t, run + 1] * age_of_deaths

    ## Scenario 1 (Point estimated)
    cs1_deaths = np.zeros((len(cent_s1["mort"]), 2))
    cs1_deaths[:, 0] = np.arange(1990, 2051, 1)
    cs1_deaths[:, 1] = cent_s1["mort"]

    for idx, val in enumerate(cs1_deaths[:, 0]):
        if val < 2023:
            cs1_deaths[idx, 1] = 0

    ghosts_cs1 = np.zeros((len(cs1_deaths), len(age_of_deaths)))
    ghosts_cs1[0, :] = cs1_deaths[0, 1] * age_of_deaths


    for t in range(1, len(cs1_deaths)):
        ppl_who_age = ghosts_cs1[t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
        ghosts_cs1[t, 0] = max(0, ghosts_cs1[t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_cs1[t - 1, 0])
        ghosts_cs1[t, 1] = max(0, ghosts_cs1[t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_cs1[t - 1, 1])
        ghosts_cs1[t, 2] = max(0, ghosts_cs1[t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_cs1[t - 1, 2])
        ghosts_cs1[t, 3] = max(0, ghosts_cs1[t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_cs1[t - 1, 3])
        ghosts_cs1[t, 4] = max(0, ghosts_cs1[t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_cs1[t - 1, 4])

        ghosts_cs1[t, :] = ghosts_cs1[t, :] + cs1_deaths[t, 1] * age_of_deaths

    ## Scenario 1 (PSA)
    s1_deaths = np.zeros((len(res_s1["mort"]), runs + 1))
    s1_deaths[:, 0] = np.arange(1990, 2051, 1)
    s1_deaths[:, 1:] = res_s1["mort"]

    for idx, val in enumerate(s1_deaths[:, 0]):
        if val < 2023:
            s1_deaths[idx, 1:] = 0

    ghosts_s1 = np.zeros((runs, len(s1_deaths), len(age_of_deaths)))

    for run in range(runs):
        ghosts_s1[run, 0, :] = s1_deaths[0, run + 1] * age_of_deaths

    for run in range(runs):
        for t in range(1, len(s1_deaths)):
            ppl_who_age = ghosts_s1[run, t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
            ghosts_s1[run, t, 0] = max(0, ghosts_s1[run, t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_s1[run, t - 1, 0])
            ghosts_s1[run, t, 1] = max(0, ghosts_s1[run, t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_s1[run, t - 1, 1])
            ghosts_s1[run, t, 2] = max(0, ghosts_s1[run, t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_s1[run, t - 1, 2])
            ghosts_s1[run, t, 3] = max(0, ghosts_s1[run, t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_s1[run, t - 1, 3])
            ghosts_s1[run, t, 4] = max(0, ghosts_s1[run, t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_s1[run, t - 1, 4])

            ghosts_s1[run, t, :] = ghosts_s1[run, t, :] + s1_deaths[t, run + 1] * age_of_deaths

    ## Scenario 2 (Point estimated)
    cs2_deaths = np.zeros((len(cent_s2["mort"]), 2))
    cs2_deaths[:, 0] = np.arange(1990, 2051, 1)
    cs2_deaths[:, 1] = cent_s2["mort"]

    for idx, val in enumerate(cs2_deaths[:, 0]):
        if val < 2023:
            cs2_deaths[idx, 1] = 0

    ghosts_cs2 = np.zeros((len(cs2_deaths), len(age_of_deaths)))
    ghosts_cs2[0, :] = cs2_deaths[0, 1] * age_of_deaths

    for t in range(1, len(cs2_deaths)):
        ppl_who_age = ghosts_cs2[t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
        ghosts_cs2[t, 0] = max(0, ghosts_cs2[t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_cs2[t - 1, 0])
        ghosts_cs2[t, 1] = max(0, ghosts_cs2[t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_cs2[t - 1, 1])
        ghosts_cs2[t, 2] = max(0, ghosts_cs2[t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_cs2[t - 1, 2])
        ghosts_cs2[t, 3] = max(0, ghosts_cs2[t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_cs2[t - 1, 3])
        ghosts_cs2[t, 4] = max(0, ghosts_cs2[t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_cs2[t - 1, 4])

        ghosts_cs2[t, :] = ghosts_cs2[t, :] + cs2_deaths[t, 1] * age_of_deaths

    ## Scenario 2 （PSA）
    s2_deaths = np.zeros((len(res_s2["mort"]), runs + 1))
    s2_deaths[:, 0] = np.arange(1990, 2051, 1)
    s2_deaths[:, 1:] = res_s2["mort"]

    for idx, val in enumerate(s2_deaths[:, 0]):
        if val < 2023:
            s2_deaths[idx, 1:] = 0

    ghosts_s2 = np.zeros((runs, len(s2_deaths), len(age_of_deaths)))

    for run in range(runs):
        ghosts_s2[run, 0, :] = s2_deaths[0, run + 1] * age_of_deaths

    for run in range(runs):
        for t in range(1, len(s2_deaths)):
            ppl_who_age = ghosts_s2[run, t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
            ghosts_s2[run, t, 0] = max(0, ghosts_s2[run, t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_s2[run, t - 1, 0])
            ghosts_s2[run, t, 1] = max(0, ghosts_s2[run, t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_s2[run, t - 1, 1])
            ghosts_s2[run, t, 2] = max(0, ghosts_s2[run, t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_s2[run, t - 1, 2])
            ghosts_s2[run, t, 3] = max(0, ghosts_s2[run, t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_s2[run, t - 1, 3])
            ghosts_s2[run, t, 4] = max(0, ghosts_s2[run, t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_s2[run, t - 1, 4])

            ghosts_s2[run, t, :] = ghosts_s2[run, t, :] + s2_deaths[t, run + 1] * age_of_deaths

    ## Scenario 3 (Point estimated)
    cs3_deaths = np.zeros((len(cent_s3["mort"]), 2))
    cs3_deaths[:, 0] = np.arange(1990, 2051, 1)
    cs3_deaths[:, 1] = cent_s3["mort"]

    for idx, val in enumerate(cs3_deaths[:, 0]):
        if val < 2023:
            cs3_deaths[idx, 1] = 0

    ghosts_cs3 = np.zeros((len(cs3_deaths), len(age_of_deaths)))
    ghosts_cs3[0, :] = cs3_deaths[0, 1] * age_of_deaths

    for t in range(1, len(cs3_deaths)):
        ppl_who_age = ghosts_cs3[t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
        ghosts_cs3[t, 0] = max(0, ghosts_cs3[t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_cs3[t - 1, 0])
        ghosts_cs3[t, 1] = max(0, ghosts_cs3[t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_cs3[t - 1, 1])
        ghosts_cs3[t, 2] = max(0, ghosts_cs3[t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_cs3[t - 1, 2])
        ghosts_cs3[t, 3] = max(0, ghosts_cs3[t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_cs3[t - 1, 3])
        ghosts_cs3[t, 4] = max(0, ghosts_cs3[t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_cs3[t - 1, 4])

        ghosts_cs3[t, :] = ghosts_cs3[t, :] + cs3_deaths[t, 1] * age_of_deaths

    ## Scenario 3 (PSA)
    s3_deaths = np.zeros((len(res_s3["mort"]), runs + 1))
    s3_deaths[:, 0] = np.arange(1990, 2051, 1)
    s3_deaths[:, 1:] = res_s3["mort"]

    for idx, val in enumerate(s3_deaths[:, 0]):
        if val < 2023:
            s3_deaths[idx, 1:] = 0

    ghosts_s3 = np.zeros((runs, len(s3_deaths), len(age_of_deaths)))

    for run in range(runs):
        ghosts_s3[run, 0, :] = s3_deaths[0, run + 1] * age_of_deaths

    for run in range(runs):
        for t in range(1, len(s3_deaths)):
            ppl_who_age = ghosts_s3[run, t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
            ghosts_s3[run, t, 0] = max(0, ghosts_s3[run, t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_s3[run, t - 1, 0])
            ghosts_s3[run, t, 1] = max(0, ghosts_s3[run, t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_s3[run, t - 1, 1])
            ghosts_s3[run, t, 2] = max(0, ghosts_s3[run, t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_s3[run, t - 1, 2])
            ghosts_s3[run, t, 3] = max(0, ghosts_s3[run, t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_s3[run, t - 1, 3])
            ghosts_s3[run, t, 4] = max(0, ghosts_s3[run, t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_s3[run, t - 1, 4])

            ghosts_s3[run, t, :] = ghosts_s3[run, t, :] + s3_deaths[t, run + 1] * age_of_deaths

    ## DALYS
    cbl_yll = np.sum(ghosts_cbl[:, 0:5], axis=1) * discount[:, 3]
    cs1_yll = np.sum(ghosts_cs1[:, 0:5], axis=1) * discount[:, 3]
    cs2_yll = np.sum(ghosts_cs2[:, 0:5], axis=1) * discount[:, 3]
    cs3_yll = np.sum(ghosts_cs3[:, 0:5], axis=1) * discount[:, 3]

    bl_yll = (np.sum(ghosts_bl[:, :, 0:5], axis=2).T)
    s1_yll = (np.sum(ghosts_s1[:, :, 0:5], axis=2).T)
    s2_yll = (np.sum(ghosts_s2[:, :, 0:5], axis=2).T)
    s3_yll = (np.sum(ghosts_s3[:, :, 0:5], axis=2).T)

    for idx in range(len(bl_yll)):
        bl_yll[idx, :] = bl_yll[idx, :] * discount[idx, 3]
        s1_yll[idx, :] = s1_yll[idx, :] * discount[idx, 3]
        s2_yll[idx, :] = s2_yll[idx, :] * discount[idx, 3]
        s3_yll[idx, :] = s3_yll[idx, :] * discount[idx, 3]

    cbl_dalys = cbl_yll + cent["yld"]
    cs1_dalys = cs1_yll + cent_s1["yld"]
    cs2_dalys = cs2_yll + cent_s2["yld"]
    cs3_dalys = cs3_yll + cent_s3["yld"]

    bl_dalys = bl_yll + res["yld"]
    s1_dalys = s1_yll + res_s1["yld"]
    s2_dalys = s2_yll + res_s2["yld"]
    s3_dalys = s3_yll + res_s3["yld"]

    ## Lost productivity, using a GDP growth rate of 3%.
    cbl_prod = np.sum(ghosts_cbl[:, 2:4], axis=1) * gdp_grw[:, 3] * etp_ratio * discount[:, 3]
    cs1_prod = np.sum(ghosts_cs1[:, 2:4], axis=1) * gdp_grw[:, 3] * etp_ratio * discount[:, 3]
    cs2_prod = np.sum(ghosts_cs2[:, 2:4], axis=1) * gdp_grw[:, 3] * etp_ratio * discount[:, 3]
    cs3_prod = np.sum(ghosts_cs3[:, 2:4], axis=1) * gdp_grw[:, 3] * etp_ratio * discount[:, 3]

    bl_prod = (np.sum(ghosts_bl[:, :, 2:4], axis=2).T)
    s1_prod = (np.sum(ghosts_s1[:, :, 2:4], axis=2).T)
    s2_prod = (np.sum(ghosts_s2[:, :, 2:4], axis=2).T)
    s3_prod = (np.sum(ghosts_s3[:, :, 2:4], axis=2).T)

    for idx in range(len(bl_yll)):
        bl_prod[idx, :] = bl_prod[idx, :] * gdp_grw[idx, 3] * etp_ratio * discount[idx, 3]
        s1_prod[idx, :] = s1_prod[idx, :] * gdp_grw[idx, 3] * etp_ratio * discount[idx, 3]
        s2_prod[idx, :] = s2_prod[idx, :] * gdp_grw[idx, 3] * etp_ratio * discount[idx, 3]
        s3_prod[idx, :] = s3_prod[idx, :] * gdp_grw[idx, 3] * etp_ratio * discount[idx, 3]

    ## NEB
    cbl_ins_tc = cbl_prod[:] + dmc_cent_bl[:, 0] + imc_cent_bl[:, 0]  # Total annual economic loss due to disease.
    cs1_ins_tc = cs1_prod[:] + dmc_cent_s1[:, 0] + imc_cent_s1[:, 0]
    cs2_ins_tc = cs2_prod[:] + dmc_cent_s2[:, 0] + imc_cent_s2[:, 0]
    cs3_ins_tc = cs3_prod[:] + dmc_cent_s3[:, 0] + imc_cent_s3[:, 0]

    # Before calculating the cumulative value, ensure that there are no NaN values in the input data.
    cbl_ins_tc = np.nan_to_num(cbl_ins_tc, nan=0)
    cs1_ins_tc = np.nan_to_num(cs1_ins_tc, nan=0)
    cs2_ins_tc = np.nan_to_num(cs2_ins_tc, nan=0)
    cs3_ins_tc = np.nan_to_num(cs3_ins_tc, nan=0)

    cbl_cum_tc = np.zeros(np.shape(cbl_ins_tc))
    cs1_cum_tc = np.zeros(np.shape(cs1_ins_tc))
    cs2_cum_tc = np.zeros(np.shape(cs2_ins_tc))
    cs3_cum_tc = np.zeros(np.shape(cs3_ins_tc))

    for i in range(len(cbl_ins_tc)):
        if i < 1:
            cbl_cum_tc[i] = cbl_ins_tc[i]
            cs1_cum_tc[i] = cs1_ins_tc[i]
            cs2_cum_tc[i] = cs2_ins_tc[i]
            cs3_cum_tc[i] = cs3_ins_tc[i]
        else:
            cbl_cum_tc[i] = cbl_cum_tc[i - 1] + cbl_ins_tc[i]
            cs1_cum_tc[i] = cs1_cum_tc[i - 1] + cs1_ins_tc[i]
            cs2_cum_tc[i] = cs2_cum_tc[i - 1] + cs2_ins_tc[i]
            cs3_cum_tc[i] = cs3_cum_tc[i - 1] + cs3_ins_tc[i]


    bl_ins_tc = bl_prod + dmc_psa_bl + imc_psa_bl
    s1_ins_tc = s1_prod + dmc_psa_s1 + imc_psa_s1
    s2_ins_tc = s2_prod + dmc_psa_s2 + imc_psa_s2
    s3_ins_tc = s3_prod + dmc_psa_s3 + imc_psa_s3

    # Before calculating the cumulative value, ensure that there are no NaN values in the input data.
    bl_ins_tc = np.nan_to_num(bl_ins_tc, nan=0)
    s1_ins_tc = np.nan_to_num(s1_ins_tc, nan=0)
    s2_ins_tc = np.nan_to_num(s2_ins_tc, nan=0)
    s3_ins_tc = np.nan_to_num(s3_ins_tc, nan=0)

    bl_cum_tc = np.zeros(np.shape(bl_ins_tc))
    s1_cum_tc = np.zeros(np.shape(s1_ins_tc))
    s2_cum_tc = np.zeros(np.shape(s2_ins_tc))
    s3_cum_tc = np.zeros(np.shape(s3_ins_tc))

    for i in range(len(bl_ins_tc)):
        if i < 1:
            bl_cum_tc[i, :] = bl_ins_tc[i, :]
            s1_cum_tc[i, :] = s1_ins_tc[i, :]
            s2_cum_tc[i, :] = s2_ins_tc[i, :]
            s3_cum_tc[i, :] = s3_ins_tc[i, :]
        else:
            bl_cum_tc[i, :] = bl_cum_tc[i - 1, :] + bl_ins_tc[i, :]
            s1_cum_tc[i, :] = s1_cum_tc[i - 1, :] + s1_ins_tc[i, :]
            s2_cum_tc[i, :] = s2_cum_tc[i - 1, :] + s2_ins_tc[i, :]
            s3_cum_tc[i, :] = s3_cum_tc[i - 1, :] + s3_ins_tc[i, :]

    cs1_neb = cbl_cum_tc - cs1_cum_tc
    cs2_neb = cbl_cum_tc - cs2_cum_tc
    cs3_neb = cbl_cum_tc - cs3_cum_tc

    s1_neb = bl_cum_tc - s1_cum_tc
    s2_neb = bl_cum_tc - s2_cum_tc
    s3_neb = bl_cum_tc - s3_cum_tc

    ## ICER
    cbl_ins_dc = dmc_cent_bl[:, 0] + imc_cent_bl[:, 0]
    cs1_ins_dc = dmc_cent_s1[:, 0] + imc_cent_s1[:, 0]
    cs2_ins_dc = dmc_cent_s2[:, 0] + imc_cent_s2[:, 0]
    cs3_ins_dc = dmc_cent_s3[:, 0] + imc_cent_s3[:, 0]

    # Before calculating the cumulative value, ensure that there are no NaN values in the input data.
    cbl_ins_dc = np.nan_to_num(cbl_ins_dc, nan=0)
    cs1_ins_dc = np.nan_to_num(cs1_ins_dc, nan=0)
    cs2_ins_dc = np.nan_to_num(cs2_ins_dc, nan=0)
    cs3_ins_dc = np.nan_to_num(cs3_ins_dc, nan=0)

    cbl_cum_dc = np.zeros(np.shape(cbl_ins_dc))
    cs1_cum_dc = np.zeros(np.shape(cs1_ins_dc))
    cs2_cum_dc = np.zeros(np.shape(cs2_ins_dc))
    cs3_cum_dc = np.zeros(np.shape(cs3_ins_dc))

    # Before calculating the cumulative value, ensure that there are no NaN values in the input data.
    cbl_dalys = np.nan_to_num(cbl_dalys, nan=0)
    cs1_dalys = np.nan_to_num(cs1_dalys, nan=0)
    cs2_dalys = np.nan_to_num(cs2_dalys, nan=0)
    cs3_dalys = np.nan_to_num(cs3_dalys, nan=0)

    cbl_cum_daly = np.zeros(np.shape(cbl_dalys))
    cs1_cum_daly = np.zeros(np.shape(cs1_dalys))
    cs2_cum_daly = np.zeros(np.shape(cs2_dalys))
    cs3_cum_daly = np.zeros(np.shape(cs3_dalys))

    for i in range(len(cbl_ins_dc)):
        if i < 1:
            cbl_cum_dc[i] = cbl_ins_dc[i]
            cs1_cum_dc[i] = cs1_ins_dc[i]
            cs2_cum_dc[i] = cs2_ins_dc[i]
            cs3_cum_dc[i] = cs3_ins_dc[i]

            cbl_cum_daly[i] = cbl_dalys[i]
            cs1_cum_daly[i] = cs1_dalys[i]
            cs2_cum_daly[i] = cs2_dalys[i]
            cs3_cum_daly[i] = cs3_dalys[i]
        else:
            cbl_cum_dc[i] = cbl_cum_dc[i - 1] + cbl_ins_dc[i]
            cs1_cum_dc[i] = cs1_cum_dc[i - 1] + cs1_ins_dc[i]
            cs2_cum_dc[i] = cs2_cum_dc[i - 1] + cs2_ins_dc[i]
            cs3_cum_dc[i] = cs3_cum_dc[i - 1] + cs3_ins_dc[i]

            cbl_cum_daly[i] = cbl_cum_daly[i - 1] + cbl_dalys[i]
            cs1_cum_daly[i] = cs1_cum_daly[i - 1] + cs1_dalys[i]
            cs2_cum_daly[i] = cs2_cum_daly[i - 1] + cs2_dalys[i]
            cs3_cum_daly[i] = cs3_cum_daly[i - 1] + cs3_dalys[i]


    cs1_numerator = -(cbl_cum_dc - cs1_cum_dc)
    cs1_denominator = (cbl_cum_daly - cs1_cum_daly)
    cs1_icer = np.divide(cs1_numerator, cs1_denominator, out=np.zeros_like(cs1_numerator), where=cs1_denominator != 0)

    # 计算cs2_icer
    cs2_numerator = -(cbl_cum_dc - cs2_cum_dc)
    cs2_denominator = (cbl_cum_daly - cs2_cum_daly)
    cs2_icer = np.divide(cs2_numerator, cs2_denominator, out=np.zeros_like(cs2_numerator), where=cs2_denominator != 0)

    # 计算cs3_icer
    cs3_numerator = -(cbl_cum_dc - cs3_cum_dc)
    cs3_denominator = (cbl_cum_daly - cs3_cum_daly)
    cs3_icer = np.divide(cs3_numerator, cs3_denominator, out=np.zeros_like(cs3_numerator), where=cs3_denominator != 0)

    # Before calculating, ensure that there are no NaN values in the input data.
    cs1_icer = np.nan_to_num(cs1_icer, nan=0.0)
    cs2_icer = np.nan_to_num(cs2_icer, nan=0.0)
    cs3_icer = np.nan_to_num(cs3_icer, nan=0.0)

    bl_ins_dc = dmc_psa_bl + imc_psa_bl
    s1_ins_dc = dmc_psa_s1 + imc_psa_s1
    s2_ins_dc = dmc_psa_s2 + imc_psa_s2
    s3_ins_dc = dmc_psa_s3 + imc_psa_s3

    # Before calculating the cumulative value, ensure that there are no NaN values in the input data.
    bl_ins_dc = np.nan_to_num(bl_ins_dc, nan=0)
    s1_ins_dc = np.nan_to_num(s1_ins_dc, nan=0)
    s2_ins_dc = np.nan_to_num(s2_ins_dc, nan=0)
    s3_ins_dc = np.nan_to_num(s3_ins_dc, nan=0)

    bl_cum_dc = np.zeros(np.shape(bl_ins_dc))
    s1_cum_dc = np.zeros(np.shape(s1_ins_dc))
    s2_cum_dc = np.zeros(np.shape(s2_ins_dc))
    s3_cum_dc = np.zeros(np.shape(s3_ins_dc))

    # Before calculating the cumulative value, ensure that there are no NaN values in the input data.
    bl_dalys = np.nan_to_num(bl_dalys, nan=0)
    s1_dalys = np.nan_to_num(s1_dalys, nan=0)
    s2_dalys = np.nan_to_num(s2_dalys, nan=0)
    s3_dalys = np.nan_to_num(s3_dalys, nan=0)

    bl_cum_daly = np.zeros(np.shape(bl_dalys))
    s1_cum_daly = np.zeros(np.shape(s1_dalys))
    s2_cum_daly = np.zeros(np.shape(s2_dalys))
    s3_cum_daly = np.zeros(np.shape(s3_dalys))

    for i in range(len(bl_ins_dc)):
        if i < 1:
            bl_cum_dc[i, :] = bl_ins_dc[i, :]
            s1_cum_dc[i, :] = s1_ins_dc[i, :]
            s2_cum_dc[i, :] = s2_ins_dc[i, :]
            s3_cum_dc[i, :] = s3_ins_dc[i, :]

            bl_cum_daly[i, :] = bl_dalys[i, :]
            s1_cum_daly[i, :] = s1_dalys[i, :]
            s2_cum_daly[i, :] = s2_dalys[i, :]
            s3_cum_daly[i, :] = s3_dalys[i, :]
        else:
            bl_cum_dc[i, :] = bl_cum_dc[i - 1, :] + bl_ins_dc[i, :]
            s1_cum_dc[i, :] = s1_cum_dc[i - 1, :] + s1_ins_dc[i, :]
            s2_cum_dc[i, :] = s2_cum_dc[i - 1, :] + s2_ins_dc[i, :]
            s3_cum_dc[i, :] = s3_cum_dc[i - 1, :] + s3_ins_dc[i, :]

            bl_cum_daly[i, :] = bl_cum_daly[i - 1, :] + bl_dalys[i, :]
            s1_cum_daly[i, :] = s1_cum_daly[i - 1, :] + s1_dalys[i, :]
            s2_cum_daly[i, :] = s2_cum_daly[i - 1, :] + s2_dalys[i, :]
            s3_cum_daly[i, :] = s3_cum_daly[i - 1, :] + s3_dalys[i, :]


    # 计算 s1_icer
    s1_numerator = -(bl_cum_dc - s1_cum_dc)
    s1_denominator = (bl_cum_daly - s1_cum_daly)
    s1_icer = np.divide(s1_numerator, s1_denominator, out=np.zeros_like(s1_numerator), where=s1_denominator != 0)

    # 计算 s2_icer
    s2_numerator = -(bl_cum_dc - s2_cum_dc)
    s2_denominator = (bl_cum_daly - s2_cum_daly)
    s2_icer = np.divide(s2_numerator, s2_denominator, out=np.zeros_like(s2_numerator), where=s2_denominator != 0)

    # 计算 s3_icer
    s3_numerator = -(bl_cum_dc - s3_cum_dc)
    s3_denominator = (bl_cum_daly - s3_cum_daly)
    s3_icer = np.divide(s3_numerator, s3_denominator, out=np.zeros_like(s3_numerator), where=s3_denominator != 0)

    # Before calculating, ensure that there are no NaN values in the input data.
    s1_icer = np.nan_to_num(s1_icer, nan=0.0)
    s2_icer = np.nan_to_num(s2_icer, nan=0.0)
    s3_icer = np.nan_to_num(s3_icer, nan=0.0)


    econ_dict = {}
    # Year by year DALYs
    econ_dict["bl_daly"] = bl_dalys
    econ_dict["s1_daly"] = s1_dalys
    econ_dict["s2_daly"] = s2_dalys
    econ_dict["s3_daly"] = s3_dalys

    econ_dict["cbl_daly"] = cbl_dalys
    econ_dict["cs1_daly"] = cs1_dalys
    econ_dict["cs2_daly"] = cs2_dalys
    econ_dict["cs3_daly"] = cs3_dalys

    # Year by year intervention costs
    econ_dict["bl_intc"] = dmc_psa_bl
    econ_dict["s1_intc"] = dmc_psa_s1
    econ_dict["s2_intc"] = dmc_psa_s2
    econ_dict["s3_intc"] = dmc_psa_s3

    econ_dict["cbl_intc"] = dmc_cent_bl
    econ_dict["cs1_intc"] = dmc_cent_s1
    econ_dict["cs2_intc"] = dmc_cent_s2
    econ_dict["cs3_intc"] = dmc_cent_s3

    # Year by year medical costs
    econ_dict["bl_medc"] = imc_psa_bl
    econ_dict["s1_medc"] = imc_psa_s1
    econ_dict["s2_medc"] = imc_psa_s2
    econ_dict["s3_medc"] = imc_psa_s3

    econ_dict["cbl_medc"] = imc_cent_bl
    econ_dict["cs1_medc"] = imc_cent_s1
    econ_dict["cs2_medc"] = imc_cent_s2
    econ_dict["cs3_medc"] = imc_cent_s3

    # Year by year productivity costs=============
    econ_dict["bl_prod"] = bl_prod
    econ_dict["s1_prod"] = s1_prod
    econ_dict["s2_prod"] = s2_prod
    econ_dict["s3_prod"] = s3_prod

    econ_dict["cbl_prod"] = cbl_prod
    econ_dict["cs1_prod"] = cs1_prod
    econ_dict["cs2_prod"] = cs2_prod
    econ_dict["cs3_prod"] = cs3_prod

    # Year by year direct costs (for plot)=============
    econ_dict["bl_dirc"] = bl_ins_dc
    econ_dict["s1_dirc"] = s1_ins_dc
    econ_dict["s2_dirc"] = s2_ins_dc
    econ_dict["s3_dirc"] = s3_ins_dc

    econ_dict["cbl_dirc"] = cbl_ins_dc
    econ_dict["cs1_dirc"] = cs1_ins_dc
    econ_dict["cs2_dirc"] = cs2_ins_dc
    econ_dict["cs3_dirc"] = cs3_ins_dc

    # Year by year detail of direct costs (for plot)
    econ_dict["cbl_dx"] = dx_cent_bl
    econ_dict["cs1_dx"] = dx_cent_s1
    econ_dict["cs2_dx"] = dx_cent_s2
    econ_dict["cs3_dx"] = dx_cent_s3

    econ_dict["cbl_vax"] = vax_cent_bl
    econ_dict["cs1_vax"] = vax_cent_s1
    econ_dict["cs2_vax"] = vax_cent_s2
    econ_dict["cs3_vax"] = vax_cent_s3

    econ_dict["cbl_tx"] = ctx_cost_bl
    econ_dict["cs1_tx"] = ctx_cost_s1
    econ_dict["cs2_tx"] = ctx_cost_s2
    econ_dict["cs3_tx"] = ctx_cost_s3

    econ_dict["cbl_mc"] = cmc_cost_bl
    econ_dict["cs1_mc"] = cmc_cost_s1
    econ_dict["cs2_mc"] = cmc_cost_s2
    econ_dict["cs3_mc"] = cmc_cost_s3

    econ_dict["cbl_hosp"] = chosp_cost_bl
    econ_dict["cs1_hosp"] = chosp_cost_s1
    econ_dict["cs2_hosp"] = chosp_cost_s2
    econ_dict["cs3_hosp"] = chosp_cost_s3

    econ_dict["cbl_hcc"] = chcc_cost_bl
    econ_dict["cs1_hcc"] = chcc_cost_s1
    econ_dict["cs2_hcc"] = chcc_cost_s2
    econ_dict["cs3_hcc"] = chcc_cost_s3

    # Year by year total costs (just in case)=======
    econ_dict["bl_tcos"] = bl_ins_tc
    econ_dict["s1_tcos"] = s1_ins_tc
    econ_dict["s2_tcos"] = s2_ins_tc
    econ_dict["s3_tcos"] = s3_ins_tc

    econ_dict["cbl_tcos"] = cbl_ins_tc
    econ_dict["cs1_tcos"] = cs1_ins_tc
    econ_dict["cs2_tcos"] = cs2_ins_tc
    econ_dict["cs3_tcos"] = cs3_ins_tc

    # Year by year Net Economic Benefit
    econ_dict["s1_neb"] = s1_neb
    econ_dict["s2_neb"] = s2_neb
    econ_dict["s3_neb"] = s3_neb

    econ_dict["cs1_neb"] = cs1_neb
    econ_dict["cs2_neb"] = cs2_neb
    econ_dict["cs3_neb"] = cs3_neb

    # Year by year ICERs
    econ_dict["s1_icer"] = s1_icer
    econ_dict["s2_icer"] = s2_icer
    econ_dict["s3_icer"] = s3_icer

    econ_dict["cs1_icer"] = cs1_icer
    econ_dict["cs2_icer"] = cs2_icer
    econ_dict["cs3_icer"] = cs3_icer

    return econ_dict



def save_dict_as_json(data_dict, file_path):
    serializable_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, pd.DataFrame):
            serializable_dict[key] = value.to_dict('records')
        elif isinstance(value, pd.Series):
            serializable_dict[key] = value.tolist()
        else:
            serializable_dict[key] = value
    with open(file_path, 'w') as file:
        json.dump(serializable_dict, file, ensure_ascii=False, indent=4)

## Macroeconomic burden
#==============================================①The first step to get Ht_education level H, and finally get edu_level=========================================================================
def calculate_h_values(edu_data, eta_dict):

    h_values = pd.DataFrame(index=edu_data.index)
    age_dict = {
        '15_49': 32,
        '50_59': 55,
        '60': 67
    }
    # Calculate the value of human capital by iterating different age-gender combinations
    for age_sex in edu_data.columns.levels[0]:
        age_group = age_sex[:-1]
        gender = age_sex[-1]

        age = age_dict[age_group]
        # Initialize the weighted sum of years of education
        weighted_ys_sum = pd.Series(0, index=edu_data.index)

        # Calculate the sum of the years of education multiplied by the corresponding semi-elasticity coefficient
        for education in edu_data.columns.levels[1]:
            education_type = education.split('_')[0]
            ys = edu_data[(age_sex, education)]
            weighted_ys_sum += eta_dict['ela_' + education_type] * ys

        # Calculating the total value of human capital
        h = np.exp(
            weighted_ys_sum +
            eta_dict['ela_wf'] * (age - 15) -
            eta_dict['ela_wfs'] * (age - 15) ** 2
        )
        h_values[age_sex] = h

    h_values.reset_index(inplace=True)
    h_values.rename(columns={'index': 'Year'}, inplace=True)

    return h_values

def extract_age_group_and_sex(input_string):
    # Regular expression matching format: "15_49M", "60F", or "15_59"
    match = re.search(r"(\d{1,2}_\d{1,2}|\d{2})([MF]?)$", input_string)
    if match:
        age_group = match.group(1)
        sex = match.group(2) if match.group(2) else ''
        return age_group + sex
    else:
        raise ValueError(f"Invalid age group and sex format: {input_string}")


def calculate_labor_participation(yld_data, yll_data, macro_data_path, mort_data, pop_data, ages, start_year, p_j=1,
                                  elimination_scenario=False):

    import numpy as np
    import pandas as pd

    # Initialize the sigma dictionary
    sigma = {}

    # Traverse mort_data and calculate sigma
    for scenario in mort_data:
        for key in mort_data[scenario]:
            mort_value = mort_data[scenario][key]
            # Extract age-sex groups and replace '-' with '_'
            if key.startswith('mort_'):
                age_sex_group = key[len('mort_'):].replace('-', '_')
            else:
                age_sex_group = key.replace('-', '_')
            # Construct the key name of pop_data
            pop_key = 'pop_' + age_sex_group
            pop_value = pop_data[scenario][pop_key]

            valid_data = np.where(pop_value != 0, mort_value / pop_value, 0)
            year_to_data = {start_year + i: valid_data[i] for i in range(len(valid_data))}

            if age_sex_group not in sigma:
                sigma[age_sex_group] = {}
            sigma[age_sex_group][scenario] = year_to_data

    # Initialize the yld_yll_ratio dictionary
    yld_yll_ratio = {}
    for scenario in yld_data:
        for key in yld_data[scenario]:
            yld_value = yld_data[scenario][key]
            # Extract age-sex groups and replace '-' with '_'
            if key.startswith('yld_'):
                age_sex_group = key[len('yld_'):].replace('-', '_')
            else:
                age_sex_group = key.replace('-', '_')
            # Construct the key name of yll_data. Note that yll_data may still use '-'
            yll_key = 'yll_' + age_sex_group.replace('-', '_')
            yll_value = yll_data[scenario][yll_key]
            # Calculating the YLD/YLL ratio
            valid_data = np.where(yll_value != 0, yld_value / yll_value, 0)
            year_to_data = {start_year + i: valid_data[i] for i in range(len(valid_data))}
            if age_sex_group not in yld_yll_ratio:
                yld_yll_ratio[age_sex_group] = {}
            yld_yll_ratio[age_sex_group][scenario] = year_to_data

    # Accessing labor force participation data
    labor_data = pd.read_excel(macro_data_path, sheet_name='lab_r')
    labor_data.set_index('year', inplace=True)
    # Replace '-' in column names with '_'
    labor_data.columns = [col.replace('-', '_') for col in labor_data.columns]

    results_dict = {}
    mi = {key: {scenario: {} for scenario in mort_data} for key in yld_yll_ratio}

    def determine_age_group(age):
        return ('0_4' if age <= 4 else
                '5_14' if age <= 14 else
                '15_49' if age <= 49 else
                '50_59' if age <= 59 else
                '60')

    # Traverse yld_yll_ratio, calculate mi and the result
    for age_sex_group in yld_yll_ratio:
        for scenario in yld_yll_ratio[age_sex_group]:
            age_group = age_sex_group[:-1].replace('+', '')
            sex = age_sex_group[-1]
            age = ages[age_group]
            length = list(range(1990, 2051))
            for i in range(len(length)):
                year = start_year + i
                sigma_value = sigma.get(age_sex_group, {}).get(scenario, {}).get(year, 0)
                yld_yll_ratio_value = yld_yll_ratio[age_sex_group][scenario].get(year, 0)
                mi[age_sex_group][scenario][year] = sigma_value * yld_yll_ratio_value
            results_key = f"{scenario}_{age_sex_group}"

            # Elimination of scenarios
            if elimination_scenario:
                # Calculate the accumulation factor for the baseline scenario
                bl_factors = np.ones(len(length))
                for i in range(1, len(length)):
                    year = start_year + i
                    if year >= 2022:
                        current_age = age + (year - 2022)
                        current_age_group = determine_age_group(current_age)
                        current_age_sex_group = current_age_group + sex
                        bl_factor = mi.get(current_age_sex_group, {}).get('bl', {}).get(year, 0)
                        bl_factors[i] = bl_factors[i - 1] * (1 - p_j * bl_factor)

                scenario_factors = np.ones(len(length))
                for i in range(1, len(length)):
                    year = start_year + i
                    if year >= 2022:
                        current_age = age + (year - 2022)
                        current_age_group = determine_age_group(current_age)
                        current_age_sex_group = current_age_group + sex
                        scenario_factor = mi.get(current_age_sex_group, {}).get(scenario, {}).get(year, 0)
                        scenario_factors[i] = scenario_factors[i - 1] * (1 - p_j * scenario_factor)

                adjusted_factors = scenario_factors / bl_factors if scenario != 'bl' else 1 / bl_factors

                lt_numerator = labor_data.loc[:, age_sex_group].values
                lt_adjusted = lt_numerator * adjusted_factors

                years = list(range(start_year, start_year + len(length)))
                results_key = f"{scenario}_{age_sex_group}"
                results_dict[results_key] = {
                    "Years": years,
                    "Data": lt_adjusted.tolist()
                }

            else:
                lt_approx = labor_data.loc[:, age_sex_group].values
                years = list(range(start_year, start_year + len(length)))
                results_key = f"{scenario}_{age_sex_group}"
                results_dict[results_key] = {
                    "Years": years,
                    "Data": lt_approx.tolist()
                }
    print(sigma)

    return yld_yll_ratio, mi, sigma, results_dict


def merge_population_data(sigma, ages, result_dict, pop_data_dict, apply_elimination=False, start_correction_year=2022):
    years = list(range(1990, 2051))
    new_result_dict = {}

    def determine_age_group(age):
        if age <= 4:
            return '0_4'
        elif age <= 14:
            return '5_14'
        elif age <= 49:
            return '15_49'
        elif age <= 59:
            return '50_59'
        else:
            return '60'

    for label, content in result_dict.items():
        content = copy.deepcopy(content)
        if not isinstance(content, pd.Series):
            content = pd.Series(content, index=years)
        adjustment_factors = pd.Series(1.0, index=years)

        if apply_elimination:
            parts = label.split('_')
            scenario = parts[0]
            if "60" in parts[2]:
                age_group_with_sex = parts[2]
            else:
                age_group_with_sex = parts[2] + '_' + parts[3]

            sigma_key = age_group_with_sex
            if sigma_key not in sigma or scenario not in sigma[sigma_key]:
                print(f"报错: 在场景{scenario}中，Sigma数据没有对应键{sigma_key}")
                continue

            for year in range(max(start_correction_year, 1990), 2051):
                if year >= start_correction_year:
                    age_group_key = sigma_key.rstrip('MF')
                    initial_age = ages.get(age_group_key)
                    current_age = initial_age - (year - 2022)
                    current_age_group = determine_age_group(current_age)
                    current_age_group_sex = current_age_group + sigma_key[-1]

                    if current_age_group_sex in sigma and scenario in sigma[current_age_group_sex] and year in \
                            sigma[current_age_group_sex][scenario]:
                        factor = sigma[current_age_group_sex][scenario][year]
                        if year == max(start_correction_year, 1990):
                            adjustment_factors[year] = 1 - factor
                        else:
                            adjustment_factors[year] = adjustment_factors[year - 1] * (1 - factor)

            content /= adjustment_factors
        else:
            adjustment_factors = pd.Series(1.0, index=years)
            content /= adjustment_factors

        if label in pop_data_dict:

            for subkey, subvalue in pop_data_dict[label].items():
                for group_key, group_value in subvalue.items():
                    group_data = pd.Series(group_value, index=years)
                    content += group_data

        new_result_dict[label] = {
            'Years': years,
            'Data': content.tolist()
        }

    return new_result_dict


def process_scenario(results_N, results_L, results_H, scenario_suffix):
    years = list(range(1990, 2051))
    final_product = pd.DataFrame(index=years)
    age_sex_groups = ['15_49M', '15_49F', '50_59M', '50_59F', '60M', '60F']
    # age_sex_groups = ['15_49M', '15_49F', '50_59M', '50_59F']
    # Transformation of Education Factor Data
    edu_df = pd.DataFrame(results_H)
    edu_df.set_index('Year', inplace=True)

    for group in age_sex_groups:
        pop_key = f"{scenario_suffix}_population_{group}"
        lab_key = f"{scenario_suffix}_{group}"

        if pop_key in results_N:
            pop_data = pd.Series(results_N[pop_key]['Data'], index=results_N[pop_key]['Years'])
        else:
            print(f"Missing population data for {pop_key} in results_N")
            continue

        if lab_key in results_L:
            lab_data = pd.Series(results_L[lab_key]['Data'], index=results_L[lab_key]['Years'])
        else:
            print(f"Missing laboratory data for {lab_key} in results_L")
            continue


        product_data = pop_data.reindex(years).multiply(lab_data.reindex(years))


        if group in edu_df.columns:
            edu_factor = edu_df[group].reindex(years)
            if edu_factor.isna().any():
                print(f"Educational factor data contains NaN for {group}")
            adjusted_data = product_data.multiply(edu_factor)
        else:
            print(f"No education factor data for {group}, using product data.")
            adjusted_data = product_data

        final_product[group] = adjusted_data

    final_product['Total'] = final_product.sum(axis=1)
    return final_product


def generate_economic_comparison(econ, is_owsa=False, owsa_type=None):

    import pandas as pd
    import numpy as np

    data_length = len(next(iter(econ.values())))
    start_year = 2015
    years = list(range(start_year, start_year + data_length))

    if is_owsa:
        allowed_owsa_types = [
            'dirc', 'cdirc', 'dx_lb', 'dx_ub', 'trt_lb', 'trt_ub',
            'util_lb', 'util_ub', 'fmc_lb', 'fmc_ub', 'surv_lb','surv_ub','hospc_lb','hospc_ub',
            'dchcc_lb', 'dchcc_ub'
        ]
        if owsa_type not in allowed_owsa_types:
            raise ValueError("owsa_type must be one of the predefined types.")

        owsa_prefix_map = {
            'dirc': 'dirc',
            'cdirc': 'cdirc',
            'dx_lb': 'dx_lb',
            'dx_ub': 'dx_ub',
            'trt_lb': 'trt_lb',
            'trt_ub': 'trt_ub',
            'util_lb': 'util_lb',
            'util_ub': 'util_ub',
            'fmc_lb': 'fmc_lb',
            'fmc_ub': 'fmc_ub',
            'surv_lb': 'surv_lb',
            'surv_ub': 'surv_ub',
            'hospc_lb': 'hospc_lb',
            'hospc_ub': 'hospc_ub',
            'dchcc_lb': 'dchcc_lb',
            'dchcc_ub': 'dchcc_ub'
        }

        prefix = owsa_prefix_map.get(owsa_type, owsa_type)

        def select_owsa_data(scenario):
            key = f'{prefix}_{scenario}'
            if key in econ:
                data = econ[key]
                #print(f"Key: {key} exists in econ.")
                # 检查数据类型
                if isinstance(data, (list, np.ndarray, pd.Series)):
                    print(f"Data for key '{key}': {data[:5]}... (显示前5个元素)")
                else:
                    print(f"Data for key '{key}' is not a list, ndarray, or Series.")
                #print(f"Data type: {type(data)}, Data shape: {np.shape(data)}")

                if isinstance(data, np.ndarray) and data.ndim > 1:
                    raise ValueError(f"Data for key '{key}' must be 1-dimensional, but got shape {data.shape}.")
                return pd.Series(data)
            else:
                #print(f"Warning: Key '{key}' not found in econ. Returning NaN series.")
                return pd.Series([np.nan] * data_length)

        # Extract data from each scene
        bl_intc = select_owsa_data('bl')
        s1_intc = select_owsa_data('s1')
        s2_intc = select_owsa_data('s2')
        s3_intc = select_owsa_data('s3')
    else:
        bl_intc = pd.Series(econ["bl"]) if "bl" in econ else pd.Series([np.nan] * data_length)
        s1_intc = pd.Series(econ["s1"]) if "s1" in econ else pd.Series([np.nan] * data_length)
        s2_intc = pd.Series(econ["s2"]) if "s2" in econ else pd.Series([np.nan] * data_length)
        s3_intc = pd.Series(econ["s3"]) if "s3" in econ else pd.Series([np.nan] * data_length)

    if len(bl_intc) != len(years):
        print(f"bl_intc length: {len(bl_intc)}, expected length: {len(years)}")
        raise ValueError("Data length mismatch for baseline scenario")
    if len(s1_intc) != len(years):
        print(f"s1_intc length: {len(s1_intc)}, expected length: {len(years)}")
        raise ValueError("Data length mismatch for scenario 1")
    if len(s2_intc) != len(years):
        print(f"s2_intc length: {len(s2_intc)}, expected length: {len(years)}")
        raise ValueError("Data length mismatch for scenario 2")
    if len(s3_intc) != len(years):
        print(f"s3_intc length: {len(s3_intc)}, expected length: {len(years)}")
        raise ValueError("Data length mismatch for scenario 3")

    cost_adjustment = pd.DataFrame({'Year': years})
    cost_adjustment['bl'] = bl_intc.values
    cost_adjustment['s1'] = s1_intc.values
    cost_adjustment['s2'] = s2_intc.values
    cost_adjustment['s3'] = s3_intc.values

    # Calculate the difference between the baseline and each scenario
    for scenario in ['s1', 's2', 's3']:
        if scenario in cost_adjustment.columns:
            cost_adjustment[f'd_bl_{scenario}'] = cost_adjustment['bl'] - cost_adjustment[scenario]

    cost_adjustment_no_elimination = cost_adjustment.copy()
    cost_adjustment_elimination = cost_adjustment.copy()
    cost_adjustment_elimination['d_bl_bl'] = cost_adjustment['bl']

    return cost_adjustment_no_elimination, cost_adjustment_elimination



def calculate_scenario(scenario, macro_data_path, eta_dict, ht_data, cost_adjustment=0):

    prod_data = pd.read_excel(macro_data_path, sheet_name='prod', index_col='Year')['prod']

    K_t = {}
    Y_t = {}

    # Initialize the value of 2023
    initial_savings = eta_dict.get('sav', 0)
    initial_gdp = eta_dict.get('gdp', 0)
    initial_depreciation_rate = eta_dict.get('dep_r', 0)
    initial_physical_stock = eta_dict.get('phy_s', 0)
    elasticity_physical = eta_dict.get('ela_phy', 0)

    K_t[2023] = initial_savings * initial_gdp + cost_adjustment + (
                1 - initial_depreciation_rate) * initial_physical_stock
    Y_t[2023] = prod_data[2023] * (K_t[2023] ** elasticity_physical) * (ht_data[2023] ** (1 - elasticity_physical))

    for year in range(2024, 2051):
        K_t[year] = initial_savings * Y_t[year - 1] + cost_adjustment + (1 - initial_depreciation_rate) * K_t[year - 1]
        Y_t[year] = prod_data[year] * (K_t[year] ** elasticity_physical) * (ht_data[year] ** (1 - elasticity_physical))

    return pd.DataFrame(list(Y_t.items()), columns=['Year', 'Y_t']), pd.DataFrame(list(K_t.items()),
                                                                                  columns=['Year', 'K_t'])

def process_scenarios(scenarios, cost_data, ht_data, macro_data_path, eta_dict, eliminate=True):
    scenario_results = {}
    data_year = cost_data[cost_data['Year'] == 2023]
    #print("DataFrame columns:", data_year.columns)
    for scenario in scenarios:
        if scenario == 'bl':
            if eliminate:
                cost_adjustment = eta_dict.get('fra_tc', 0) * data_year['bl'].iloc[0]
            else:
                cost_adjustment = 0
        else:
            column_name = f'd_bl_{scenario}'
            if column_name in data_year.columns:
                cost_adjustment = eta_dict.get('fra_tc', 0) * data_year[column_name].iloc[0]
            else:
                #print(f"Scenario '{scenario}' has no cost difference for year 2023.")
                cost_adjustment = 0

        #print(f"Processing scenario '{scenario}', cost adjustment: {cost_adjustment}")
        scenario_results[scenario] = calculate_scenario(scenario, macro_data_path, eta_dict, ht_data.get(scenario, {}),
                                                        cost_adjustment)

    return scenario_results

def discounted_sum(scenario_bl, scenario_compare, discount_rate):
    df_bl = pd.DataFrame(scenario_bl)
    df_compare = pd.DataFrame(scenario_compare)
    df_difference = df_bl['Y_t'] - df_compare['Y_t']
    df_discounted = df_difference / ((1 + discount_rate) ** (df_bl['Year'] - 2023))
    total_discounted_sum = df_discounted.sum()
    return total_discounted_sum

def yearly_discounted_difference(scenario_bl, scenario_compare, discount_rate):
    df_bl = pd.DataFrame(scenario_bl)
    df_compare = pd.DataFrame(scenario_compare)
    df_difference = df_bl['Y_t'] - df_compare['Y_t']
    df_discounted = df_difference / ((1 + discount_rate) ** (df_bl['Year'] - 2023))

    df_result = pd.DataFrame({
        'Year': df_bl['Year'],
        'Discounted Difference': df_discounted
    })
    return df_result

# 定义辅助函数
def extract_data_for_run(store_runs, run_index, data_type):
    data_run = {}
    for key, value in store_runs.items():
        if key.startswith(data_type + '_'):
            data_run[key] = value[:, run_index]
    return data_run


# #Comprehensive functions
def integrated_scenario_analysis(macro_data_path, ages, start_year, cost_adjustment, discount_rate,
                                 pop_data, yld_data, yll_data, mort_data):

    results = {
        'total_discounted_bl_bl_elimination': 0,
        'total_discounted_bl_s1_no_elimination': 0,
        'total_discounted_bl_s2_no_elimination': 0,
        'total_discounted_bl_s3_no_elimination': 0
    }
    eta = pd.read_excel(macro_data_path, sheet_name='data_2023', header=None)
    eta_dict = dict(zip(eta.iloc[:, 0], eta.iloc[:, 1]))
    edu_data = pd.read_excel(
        macro_data_path,
        sheet_name='edu_data',
        header=[0, 1],
        index_col=0)

    scenarios = ["bl", "s1", "s2", "s3"]
    age_sex_groups_raw = ["15_49M", "50_59M", "60M", "15_49F", "50_59F", "60F"]

    result_dict_no_elimination = {
        f"{scenario}_population_{group.replace('-', '_').replace('+', '')}": pop_data[scenario]['pop_' + group]
        for scenario in scenarios
        for group in age_sex_groups_raw
    }

    ht_data_no_elimination = {}
    ht_data_elimination = {}
    results_H = calculate_h_values(edu_data, eta_dict)

    # Labor force participation data calculation
    ratio_no_elimination, mi_no_elimination, sigma_no_elimination, results_L_no_elimination = calculate_labor_participation(
        yld_data, yll_data, macro_data_path, mort_data, pop_data, ages, start_year, p_j=1,
        elimination_scenario=False)

    ratio_elimination, mi_elimination, sigma_elimination, results_L_elimination = calculate_labor_participation(
        yld_data, yll_data, macro_data_path, mort_data, pop_data, ages, start_year, p_j=1,
        elimination_scenario=True)

    results_N_no_elimination = merge_population_data(
        sigma_no_elimination, ages, result_dict_no_elimination,pop_data_dict, apply_elimination=False)
    results_N_elimination = merge_population_data(
        sigma_elimination, ages, result_dict_no_elimination,pop_data_dict, apply_elimination=True)

    for scenario in ['bl', 's1', 's2', 's3']:
        ht_data_no_elimination[scenario] = process_scenario(
            results_N_no_elimination, results_L_no_elimination, results_H, scenario)['Total']
        ht_data_elimination[scenario] = process_scenario(
            results_N_elimination, results_L_elimination, results_H, scenario)['Total']


    # Processing and comparing scenes
    scenario_results_no_elimination = process_scenarios(
        ['bl', 's1', 's2', 's3'], cost_adjustment,
        ht_data_no_elimination, macro_data_path, eta_dict,
        eliminate=False)

    scenario_results_elimination = process_scenarios(
        ['bl', 's1', 's2', 's3'], cost_adjustment,
        ht_data_elimination, macro_data_path, eta_dict, eliminate=True)

    bl_Yt_elimination, bl_Kt_elimination = scenario_results_elimination["bl"]
    bl_Yt_no_elimination, bl_Kt_no_elimination = scenario_results_no_elimination["bl"]
    s1_Yt_no_elimination, s1_Kt_no_elimination = scenario_results_no_elimination["s1"]
    s2_Yt_no_elimination, s2_Kt_no_elimination = scenario_results_no_elimination["s2"]
    s3_Yt_no_elimination, s3_Kt_no_elimination = scenario_results_no_elimination["s3"]

    total_discounted_bl_bl_elimination = discounted_sum(
        bl_Yt_elimination, bl_Yt_no_elimination, discount_rate)
    total_discounted_bl_s1_no_elimination = discounted_sum(
        bl_Yt_no_elimination, s1_Yt_no_elimination, discount_rate)
    total_discounted_bl_s2_no_elimination = discounted_sum(
        bl_Yt_no_elimination, s2_Yt_no_elimination, discount_rate)
    total_discounted_bl_s3_no_elimination = discounted_sum(
        bl_Yt_no_elimination, s3_Yt_no_elimination, discount_rate)

    # Calculation of the difference each year
    yearly_difference_bl_bl_elimination = yearly_discounted_difference(
        bl_Yt_elimination, bl_Yt_no_elimination, discount_rate)
    yearly_difference_bl_s1_no_elimination = yearly_discounted_difference(
        bl_Yt_no_elimination, s1_Yt_no_elimination, discount_rate)
    yearly_difference_bl_s2_no_elimination = yearly_discounted_difference(
        bl_Yt_no_elimination, s2_Yt_no_elimination, discount_rate)
    yearly_difference_bl_s3_no_elimination = yearly_discounted_difference(
        bl_Yt_no_elimination, s3_Yt_no_elimination, discount_rate)

    return {
        'Yt_total_discounted': {'total_discounted_bl_bl_elimination': total_discounted_bl_bl_elimination / 1e8,
        'total_discounted_bl_s1_no_elimination': total_discounted_bl_s1_no_elimination / 1e8,
        'total_discounted_bl_s2_no_elimination': total_discounted_bl_s2_no_elimination / 1e8,
        'total_discounted_bl_s3_no_elimination': total_discounted_bl_s3_no_elimination / 1e8,
        },
        'yearly_discounted_differences': {
            'yearly_bl_bl_elimination': yearly_difference_bl_bl_elimination / 1e8,
            'yearly_bl_s1_no_elimination': yearly_difference_bl_s1_no_elimination / 1e8,
            'yearly_bl_s2_no_elimination': yearly_difference_bl_s2_no_elimination / 1e8,
            'yearly_bl_s3_no_elimination': yearly_difference_bl_s3_no_elimination / 1e8,
        },
        'Yt_params': {
            'bl_Yt_elimination': bl_Yt_elimination,
            'bl_Yt_no_elimination': bl_Yt_no_elimination,
            's1_Yt_no_elimination': s1_Yt_no_elimination,
            's2_Yt_no_elimination': s2_Yt_no_elimination,
            's3_Yt_no_elimination': s3_Yt_no_elimination
        }
    }



def print_structure(simulation_data):

    for key, value in simulation_data.items():
        print(f"Key: {key}")
        if isinstance(value, list):
            print(f"Type: List, Length: {len(value)}")
            if len(value) > 0:
                print(f"Element type: {type(value[0])}")
                if hasattr(value[0], 'shape'):
                    print(f"Element shape: {value[0].shape}")
        elif isinstance(value, np.ndarray):
            print(f"Type: NumPy array, Shape: {value.shape}")
        else:
            print(f"Type: {type(value)}")
        print("\n")

## Sensitivity analysis
def OWSA_analysis(macro_data_path, ages, start_year, econ, discount_rate, is_owsa, owsa_type, owsa_type_econ,yld_data, yll_data,
                  pop_data, mort_data):

    eta = pd.read_excel(macro_data_path, sheet_name='data_2023', header=None)
    eta_dict = dict(zip(eta.iloc[:, 0], eta.iloc[:, 1]))
    edu_data = pd.read_excel(
        macro_data_path,
        sheet_name='edu_data',
        header=[0, 1],
        index_col=0
    )

    scenarios = ["bl", "s1"]
    age_sex_groups_raw = ["15_49M", "50_59M", "60M", "15_49F", "50_59F", "60F"]

    result_dict_no_elimination = {
        f"{scenario}_population_{group.replace('-', '_').replace('+', '')}": pop_data_dict[owsa_type][scenario][
            f'pop_{group}']
        for scenario in scenarios
        for group in age_sex_groups_raw
    }

    # Calculation of H values
    results_H = calculate_h_values(edu_data, eta_dict)
    # Calculating labor force participation data
    ratio_no_elimination, mi_no_elimination, sigma_no_elimination, results_L_no_elimination = calculate_labor_participation(
        yld_data, yll_data, macro_data_path, mort_data, pop_data, ages, start_year, p_j=1, elimination_scenario=False)

    ratio_elimination, mi_elimination, sigma_elimination, results_L_elimination = calculate_labor_participation(
        yld_data, yll_data, macro_data_path, mort_data, pop_data, ages, start_year, p_j=1, elimination_scenario=True)

    # Merging population data
    results_N_no_elimination = merge_population_data(
        sigma_no_elimination, ages, result_dict_no_elimination, pop_data_dict, apply_elimination=False)
    results_N_elimination = merge_population_data(
        sigma_elimination, ages, result_dict_no_elimination, pop_data_dict, apply_elimination=True)

    # Process data by scenario and calculate total output
    ht_data_no_elimination = {}
    ht_data_elimination = {}
    print("Process data by scenario and calculate total output")
    for scenario in scenarios:
        ht_data_no_elimination[scenario] = process_scenario(
            results_N_no_elimination, results_L_no_elimination, results_H, scenario)['Total']
        ht_data_elimination[scenario] = process_scenario(
            results_N_elimination, results_L_elimination, results_H, scenario)['Total']

    # Generate economic comparison data (OWSA specific)
    cost_data_no_elimination, cost_data_elimination = generate_economic_comparison(econ, is_owsa, owsa_type_econ)

    # Process economic data by scenario
    scenario_results_no_elimination = process_scenarios(
        scenarios, cost_data_no_elimination, ht_data_no_elimination, macro_data_path, eta_dict, eliminate=False)
    scenario_results_elimination = process_scenarios(
        scenarios, cost_data_elimination, ht_data_elimination, macro_data_path, eta_dict, eliminate=True)

    bl_Yt_elimination, bl_Kt_elimination = scenario_results_elimination["bl"]
    bl_Yt_no_elimination, bl_Kt_no_elimination = scenario_results_no_elimination["bl"]
    s1_Yt_no_elimination, s1_Kt_no_elimination = scenario_results_no_elimination["s1"]

    total_discounted_bl_bl_elimination = discounted_sum(bl_Yt_elimination, bl_Yt_no_elimination, discount_rate)
    total_discounted_bl_s1_no_elimination = discounted_sum(bl_Yt_no_elimination, s1_Yt_no_elimination, discount_rate)

    return {
        'Yt_total_discounted': {
            'total_discounted_bl_bl_elimination': total_discounted_bl_bl_elimination / 1e8,
            'total_discounted_bl_s1_no_elimination': total_discounted_bl_s1_no_elimination / 1e8
        }
    }



# MTCT worst case
mtct_wc = [  # HBeAg negative HepB-BD effectiveness
    {"0-4M_sve": ["sag_ve", "0-4M", 1990, 2021, 2023, 0.95, 0.95, 0.86]},
    {"0-4F_sve": ["sag_ve", "0-4F", 1990, 2021, 2023, 0.95, 0.95, 0.86]},
    # HBeAg positive HepB-BD effectiveness
    {"0-4M_eve": ["eag_ve", "0-4M", 1990, 2021, 2023, 0.85, 0.85, 0.65]},
    {"0-4F_eve": ["eag_ve", "0-4F", 1990, 2021, 2023, 0.85, 0.85, 0.65]},
    # HBeAg positive HepB-BD+mAV effectiveness
    {"0-4M_mve": ["mav_ve", "0-4M", 1990, 2021, 2023, 0.98, 0.98, 0.97]},
    {"0-4F_mve": ["mav_ve", "0-4F", 1990, 2021, 2023, 0.98, 0.98, 0.97]},
    # HVL among HBeAg positive pregnancies
    {"0-4M_evl": ["eag_hvl", "0-4M", 1990, 2021, 2023, 0.91, 0.91, 0.96]},
    {"0-4F_evl": ["eag_hvl", "0-4F", 1990, 2021, 2023, 0.91, 0.91, 0.96]},
    # HVL among HBeAg negative pregnancies
    {"0-4M_svl": ["sag_hvl", "0-4M", 1990, 2021, 2023, 0.11, 0.11, 0.16]},
    {"0-4F_svl": ["sag_hvl", "0-4F", 1990, 2021, 2023, 0.11, 0.11, 0.16]},
    # HVL transmission risk
    {"0-4M_rsk": ["hvl_trisk", "0-4M", 1990, 2021, 2023, 0.85, 0.85, 1]},
    {"0-4F_rsk": ["hvl_trisk", "0-4F", 1990, 2021, 2023, 0.85, 0.85, 1]},
    # Risk of CHB following MTCT
    {"0-4M_cip": ["ci_p", "0-4M", 1990, 2021, 2023, 0.885, 0.885, 0.93]},
    {"0-4F_cip": ["ci_p", "0-4F", 1990, 2021, 2023, 0.885, 0.885, 0.93]}]

## MTCT best case
mtct_bc = [  # HBeAg negative HepB-BD effectiveness
    {"0-4M_sve": ["sag_ve", "0-4M", 1990, 2021, 2023, 0.95, 0.95, 0.99]},
    {"0-4F_sve": ["sag_ve", "0-4F", 1990, 2021, 2023, 0.95, 0.95, 0.99]},
    # HBeAg positive HepB-BD effectiveness
    {"0-4M_eve": ["eag_ve", "0-4M", 1990, 2021, 2023, 0.85, 0.85, 0.90]},
    {"0-4F_eve": ["eag_ve", "0-4F", 1990, 2021, 2023, 0.85, 0.85, 0.90]},
    # HBeAg positive HepB-BD+mAV effectiveness
    {"0-4M_mve": ["mav_ve", "0-4M", 1990, 2021, 2023, 0.98, 0.98, 0.99]},
    {"0-4F_mve": ["mav_ve", "0-4F", 1990, 2021, 2023, 0.98, 0.98, 0.99]},
    # HVL among HBeAg positive pregnancies
    {"0-4M_evl": ["eag_hvl", "0-4M", 1990, 2021, 2023, 0.91, 0.91, 0.84]},
    {"0-4F_evl": ["eag_hvl", "0-4F", 1990, 2021, 2023, 0.91, 0.91, 0.84]},
    # HVL among HBeAg negative pregnancies
    {"0-4M_svl": ["sag_hvl", "0-4M", 1990, 2021, 2023, 0.11, 0.11, 0.05]},
    {"0-4F_svl": ["sag_hvl", "0-4F", 1990, 2021, 2023, 0.11, 0.11, 0.05]},
    # HVL transmission risk
    {"0-4M_rsk": ["hvl_trisk", "0-4M", 1990, 2021, 2023, 0.85, 0.85, 0.7]},
    {"0-4F_rsk": ["hvl_trisk", "0-4F", 1990, 2021, 2023, 0.85, 0.85, 0.7]},
    # Risk of CHB following MTCT
    {"0-4M_cip": ["ci_p", "0-4M", 1990, 2021, 2023, 0.885, 0.885, 0.84]},
    {"0-4F_cip": ["ci_p", "0-4F", 1990, 2021, 2023, 0.885, 0.885, 0.84]}]

## Mortality worst case
mort_wc = [  # Acute mortality rate
    {"0-4M_acu": ["m_acu", "0-4M", 1990, 2021, 2023, 0.001, 0.001, 0.0015]},
    {"0-4F_acu": ["m_acu", "0-4F", 1990, 2021, 2023, 0.001, 0.001, 0.0015]},
    {"5-14M_acu": ["m_acu", "5-14M", 1990, 2021, 2023, 0.0011, 0.0011, 0.0015]},
    {"5-14F_acu": ["m_acu", "5-14F", 1990, 2021, 2023, 0.0011, 0.0011, 0.0015]},
    {"15-49M_acu": ["m_acu", "15-49M", 1990, 2021, 2023, 0.0026, 0.0026, 0.00375]},
    {"15-49F_acu": ["m_acu", "15-49F", 1990, 2021, 2023, 0.0026, 0.0026, 0.00375]},
    {"50-59M_acu": ["m_acu", "50-59M", 1990, 2021, 2023, 0.005, 0.005, 0.0055]},
    {"50-59F_acu": ["m_acu", "50-59F", 1990, 2021, 2023, 0.005, 0.005, 0.0055]},
    {"60+M_acu": ["m_acu", "60+M", 1990, 2021, 2023, 0.005, 0.005, 0.0055]},
    {"60+F_acu": ["m_acu", "60+F", 1990, 2021, 2023, 0.005, 0.005, 0.0055]},
    # Decompensated cirrhosis mortality rate
    {"0-4M_dc": ["m_dc", "0-4M", 1990, 2021, 2023, 0.0, 0.0, 0.0]},
    {"0-4F_dc": ["m_dc", "0-4F", 1990, 2021, 2023, 0.0, 0.0, 0.0]},
    {"5-14M_dc": ["m_dc", "5-14M", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"5-14F_dc": ["m_dc", "5-14F", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"15-49M_dc": ["m_dc", "15-49M", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"15-49F_dc": ["m_dc", "15-49F", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"50-59M_dc": ["m_dc", "50-59M", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"50-59F_dc": ["m_dc", "50-59F", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"60+M_dc": ["m_dc", "60+M", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    {"60+F_dc": ["m_dc", "60+F", 1990, 2021, 2023, 0.34, 0.34, 0.4]},
    # Hepatocellular carcinoma mortality rate
    {"5-14M_hcc": ["m_hcc", "5-14M", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"5-14F_hcc": ["m_hcc", "5-14F", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"15-49M_hcc": ["m_hcc", "15-49M", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"15-49F_hcc": ["m_hcc", "15-49F", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"50-59M_hcc": ["m_hcc", "50-59M", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"50-59F_hcc": ["m_hcc", "50-59F", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"60+M_hcc": ["m_hcc", "60+M", 1990, 2021, 2023, 0.6, 0.6, 0.72]},
    {"60+F_hcc": ["m_hcc", "60+F", 1990, 2021, 2023, 0.6, 0.6, 0.72]}]

## Mortalility best case
mort_bc = [  # Acute mortality rate
    {"0-4M_acu": ["m_acu", "0-4M", 1990, 2021, 2023, 0.001, 0.001, 0.0007]},
    {"0-4F_acu": ["m_acu", "0-4F", 1990, 2021, 2023, 0.001, 0.001, 0.0007]},
    {"5-14M_acu": ["m_acu", "5-14M", 1990, 2021, 2023, 0.0011, 0.0011, 0.0007]},
    {"5-14F_acu": ["m_acu", "5-14F", 1990, 2021, 2023, 0.0011, 0.0011, 0.0007]},
    {"15-49M_acu": ["m_acu", "15-49M", 1990, 2021, 2023, 0.0026, 0.0026, 0.0007]},
    {"15-49F_acu": ["m_acu", "15-49F", 1990, 2021, 2023, 0.0026, 0.0026, 0.0007]},
    {"50-59M_acu": ["m_acu", "50-59M", 1990, 2021, 2023, 0.005, 0.005, 0.003]},
    {"50-59F_acu": ["m_acu", "50-59F", 1990, 2021, 2023, 0.005, 0.005, 0.003]},
    {"60+M_acu": ["m_acu", "60+M", 1990, 2021, 2023, 0.005, 0.005, 0.003]},
    {"60+F_acu": ["m_acu", "60+F", 1990, 2021, 2023, 0.005, 0.005, 0.003]},
    # Decompensated cirrhosis mortality rate
    {"0-4M_dc": ["m_dc", "0-4M", 1990, 2021, 2023, 0, 0, 0]},
    {"0-4F_dc": ["m_dc", "0-4F", 1990, 2021, 2023, 0, 0, 0]},
    {"5-14M_dc": ["m_dc", "5-14M", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"5-14F_dc": ["m_dc", "5-14F", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"15-49M_dc": ["m_dc", "15-49M", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"15-49F_dc": ["m_dc", "15-49F", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"50-59M_dc": ["m_dc", "50-59M", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"50-59F_dc": ["m_dc", "50-59F", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"60+M_dc": ["m_dc", "60+M", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    {"60+F_dc": ["m_dc", "60+F", 1990, 2021, 2023, 0.34, 0.34, 0.074]},
    # Hepatocellular carcinoma mortality rate
    {"5-14M_hcc": ["m_hcc", "5-14M", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"5-14F_hcc": ["m_hcc", "5-14F", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"15-49M_hcc": ["m_hcc", "15-49M", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"15-49F_hcc": ["m_hcc", "15-49F", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"50-59M_hcc": ["m_hcc", "50-59M", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"50-59F_hcc": ["m_hcc", "50-59F", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"60+M_hcc": ["m_hcc", "60+M", 1990, 2021, 2023, 0.6, 0.6, 0.2]},
    {"60+F_hcc": ["m_hcc", "60+F", 1990, 2021, 2023, 0.6, 0.6, 0.2]}]

## Treatment Effectiveness worst case
tex_wc = [  # HBeAg seroclearance
    {"0-4M_te_icl_ict": ["te_icl_ict", "0-4M", 1990, 2021, 2023, 2, 2, 1]},
    {"0-4F_te_icl_ict": ["te_icl_ict", "0-4F", 1990, 2021, 2023, 2, 2, 1]},
    {"5-14M_te_icl_ict": ["te_icl_ict", "5-14M", 1990, 2021, 2023, 2, 2, 1]},
    {"5-14F_te_icl_ict": ["te_icl_ict", "5-14F", 1990, 2021, 2023, 2, 2, 1]},
    {"15-49M_te_icl_ict": ["te_icl_ict", "15-49M", 1990, 2021, 2023, 2, 2, 1]},
    {"15-49F_te_icl_ict": ["te_icl_ict", "15-49F", 1990, 2021, 2023, 2, 2, 1]},
    {"50-59M_te_icl_ict": ["te_icl_ict", "50-59M", 1990, 2021, 2023, 2, 2, 1]},
    {"50-59F_te_icl_ict": ["te_icl_ict", "50-59F", 1990, 2021, 2023, 2, 2, 1]},
    {"60+M_te_icl_ict": ["te_icl_ict", "60+M", 1990, 2021, 2023, 2, 2, 1]},
    {"60+F_te_icl_ict": ["te_icl_ict", "60+F", 1990, 2021, 2023, 2, 2, 1]},
    # Immune clearance to compensated cirrhosis
    {"0-4M_te_icl_cc": ["te_icl_cc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"0-4F_te_icl_cc": ["te_icl_cc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"5-14M_te_icl_cc": ["te_icl_cc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"5-14F_te_icl_cc": ["te_icl_cc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"15-49M_te_icl_cc": ["te_icl_cc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"15-49F_te_icl_cc": ["te_icl_cc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"50-59M_te_icl_cc": ["te_icl_cc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"50-59F_te_icl_cc": ["te_icl_cc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"60+M_te_icl_cc": ["te_icl_cc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"60+F_te_icl_cc": ["te_icl_cc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    # Immune escape to compensated cirrhosis
    {"0-4M_te_ie_cc": ["te_ie_cc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"0-4F_te_ie_cc": ["te_ie_cc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"5-14M_te_ie_cc": ["te_ie_cc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"5-14F_te_ie_cc": ["te_ie_cc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"15-49M_te_ie_cc": ["te_ie_cc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"15-49F_te_ie_cc": ["te_ie_cc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"50-59M_te_ie_cc": ["te_ie_cc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"50-59F_te_ie_cc": ["te_ie_cc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"60+M_te_ie_cc": ["te_ie_cc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    {"60+F_te_ie_cc": ["te_ie_cc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.4]},
    # Compensated cirrhosis to decompensated cirrhosis
    {"0-4M_te_cc_dc": ["te_cc_dc", "0-4M", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"0-4F_te_cc_dc": ["te_cc_dc", "0-4F", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"5-14M_te_cc_dc": ["te_cc_dc", "5-14M", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"5-14F_te_cc_dc": ["te_cc_dc", "5-14F", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"15-49M_te_cc_dc": ["te_cc_dc", "15-49M", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"15-49F_te_cc_dc": ["te_cc_dc", "15-49F", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"50-59M_te_cc_dc": ["te_cc_dc", "50-59M", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"50-59F_te_cc_dc": ["te_cc_dc", "50-59F", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"60+M_te_cc_dc": ["te_cc_dc", "60+M", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    {"60+F_te_cc_dc": ["te_cc_dc", "60+M", 1990, 2021, 2023, 0.5, 0.5, 0.75]},
    # Decompensated cirrhosis to compensated cirrhosis
    {"0-4M_te_dc_cc": ["te_dc_cc", "0-4M", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"0-4F_te_dc_cc": ["te_dc_cc", "0-4F", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"5-14M_te_dc_cc": ["te_dc_cc", "5-14M", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"5-14F_te_dc_cc": ["te_dc_cc", "5-14F", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"15-49M_te_dc_cc": ["te_dc_cc", "15-49M", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"15-49F_te_dc_cc": ["te_dc_cc", "15-49F", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"50-59M_te_dc_cc": ["te_dc_cc", "50-59M", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"50-59F_te_dc_cc": ["te_dc_cc", "50-59F", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"60+M_te_dc_cc": ["te_dc_cc", "60+M", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    {"60+F_te_dc_cc": ["te_dc_cc", "60+F", 1990, 2021, 2023, 0.15, 0.15, 0.1]},
    # Immune clearance to hepatocellular carcinoma
    {"0-4M_te_icl_hcc": ["te_icl_hcc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"0-4F_te_icl_hcc": ["te_icl_hcc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"5-14M_te_icl_hcc": ["te_icl_hcc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"5-14F_te_icl_hcc": ["te_icl_hcc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"15-49M_te_icl_hcc": ["te_icl_hcc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"15-49F_te_icl_hcc": ["te_icl_hcc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"50-59M_te_icl_hcc": ["te_icl_hcc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"50-59F_te_icl_hcc": ["te_icl_hcc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"60+M_te_icl_hcc": ["te_icl_hcc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"60+F_te_icl_hcc": ["te_icl_hcc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    # Immune control to hepatocellular carcinoma
    {"0-4M_te_ict_hcc": ["te_ict_hcc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"0-4F_te_ict_hcc": ["te_ict_hcc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"5-14M_te_ict_hcc": ["te_ict_hcc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"5-14F_te_ict_hcc": ["te_ict_hcc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"15-49M_te_ict_hcc": ["te_ict_hcc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"15-49F_te_ict_hcc": ["te_ict_hcc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"50-59M_te_ict_hcc": ["te_ict_hcc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"50-59F_te_ict_hcc": ["te_ict_hcc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"60+M_te_ict_hcc": ["te_ict_hcc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"60+F_te_ict_hcc": ["te_ict_hcc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    # Immune escape to hepatocellular carcinoma
    {"0-4M_te_ie_hcc": ["te_ie_hcc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"0-4F_te_ie_hcc": ["te_ie_hcc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"5-14M_te_ie_hcc": ["te_ie_hcc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"5-14F_te_ie_hcc": ["te_ie_hcc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"15-49M_te_ie_hcc": ["te_ie_hcc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"15-49F_te_ie_hcc": ["te_ie_hcc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"50-59M_te_ie_hcc": ["te_ie_hcc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"50-59F_te_ie_hcc": ["te_ie_hcc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"60+M_te_ie_hcc": ["te_ie_hcc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    {"60+F_te_ie_hcc": ["te_ie_hcc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.5]},
    # Compensated cirrhosis to hepatocellular carcinoma
    {"0-4M_te_cc_hcc": ["te_cc_hcc", "0-4M", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"0-4F_te_cc_hcc": ["te_cc_hcc", "0-4F", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"5-14M_te_cc_hcc": ["te_cc_hcc", "5-14M", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"5-14F_te_cc_hcc": ["te_cc_hcc", "5-14F", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"15-49M_te_cc_hcc": ["te_cc_hcc", "15-49M", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"15-49F_te_cc_hcc": ["te_cc_hcc", "15-49F", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"50-59M_te_cc_hcc": ["te_cc_hcc", "50-59M", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"50-59F_te_cc_hcc": ["te_cc_hcc", "50-59F", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"60+M_te_cc_hcc": ["te_cc_hcc", "60+M", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    {"60+F_te_cc_hcc": ["te_cc_hcc", "60+F", 1990, 2021, 2023, 0.7, 0.7, 0.85]},
    # Decompensated cirrhosis to hepatocellular carcinoma
    {"0-4M_te_dc_hcc": ["te_dc_hcc", "0-4M", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"0-4F_te_dc_hcc": ["te_dc_hcc", "0-4F", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"5-14M_te_dc_hcc": ["te_dc_hcc", "5-14M", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"5-14F_te_dc_hcc": ["te_dc_hcc", "5-14F", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"15-49M_te_dc_hcc": ["te_dc_hcc", "15-49M", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"15-49F_te_dc_hcc": ["te_dc_hcc", "15-49F", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"50-59M_te_dc_hcc": ["te_dc_hcc", "50-59M", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"50-59F_te_dc_hcc": ["te_dc_hcc", "50-59F", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"60+M_te_dc_hcc": ["te_dc_hcc", "60+M", 1990, 2021, 2023, 0.75, 0.75, 1]},
    {"60+F_te_dc_hcc": ["te_dc_hcc", "60+F", 1990, 2021, 2023, 0.75, 0.75, 1]},
    # Mortality from decompensated cirrhosis
    {"0-4M_te_m_dc": ["te_m_dc", "0-4M", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"0-4F_te_m_dc": ["te_m_dc", "0-4F", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"5-14M_te_m_dc": ["te_m_dc", "5-14M", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"5-14F_te_m_dc": ["te_m_dc", "5-14F", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"15-49M_te_m_dc": ["te_m_dc", "15-49M", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"15-49F_te_m_dc": ["te_m_dc", "15-49F", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"50-59M_te_m_dc": ["te_m_dc", "50-59M", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"50-59F_te_m_dc": ["te_m_dc", "50-59F", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"60+M_te_m_dc": ["te_m_dc", "60+M", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    {"60+F_te_m_dc": ["te_m_dc", "60+F", 1990, 2021, 2023, 0.5, 0.5, 0.7]},
    # Mortality from hepatocellular carcinoma
    {"0-4M_te_m_hcc": ["te_m_hcc", "0-4M", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"0-4F_te_m_hcc": ["te_m_hcc", "0-4F", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"5-14M_te_m_hcc": ["te_m_hcc", "5-14M", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"5-14F_te_m_hcc": ["te_m_hcc", "5-14F", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"15-49M_te_m_hcc": ["te_m_hcc", "15-49M", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"15-49F_te_m_hcc": ["te_m_hcc", "15-49F", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"50-59M_te_m_hcc": ["te_m_hcc", "50-59M", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"50-59F_te_m_hcc": ["te_m_hcc", "50-59F", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"60+M_te_m_hcc": ["te_m_hcc", "60+M", 1990, 2021, 2023, 0.8, 0.8, 1]},
    {"60+F_te_m_hcc": ["te_m_hcc", "60+F", 1990, 2021, 2023, 0.8, 0.8, 1]}
]

## Treatment Effectiveness best case
tex_bc = [  # HBeAg seroclearance
    {"0-4M_te_icl_ict": ["te_icl_ict", "0-4M", 1990, 2021, 2023, 2, 2, 3]},
    {"0-4F_te_icl_ict": ["te_icl_ict", "0-4F", 1990, 2021, 2023, 2, 2, 3]},
    {"5-14M_te_icl_ict": ["te_icl_ict", "5-14M", 1990, 2021, 2023, 2, 2, 3]},
    {"5-14F_te_icl_ict": ["te_icl_ict", "5-14F", 1990, 2021, 2023, 2, 2, 3]},
    {"15-49M_te_icl_ict": ["te_icl_ict", "15-49M", 1990, 2021, 2023, 2, 2, 3]},
    {"15-49F_te_icl_ict": ["te_icl_ict", "15-49F", 1990, 2021, 2023, 2, 2, 3]},
    {"50-59M_te_icl_ict": ["te_icl_ict", "50-59M", 1990, 2021, 2023, 2, 2, 3]},
    {"50-59F_te_icl_ict": ["te_icl_ict", "50-59F", 1990, 2021, 2023, 2, 2, 3]},
    {"60+M_te_icl_ict": ["te_icl_ict", "60+M", 1990, 2021, 2023, 2, 2, 3]},
    {"60+F_te_icl_ict": ["te_icl_ict", "60+F", 1990, 2021, 2023, 2, 2, 3]},
    # Immune clearance to compensated cirrhosis
    {"0-4M_te_icl_cc": ["te_icl_cc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"0-4F_te_icl_cc": ["te_icl_cc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"5-14M_te_icl_cc": ["te_icl_cc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"5-14F_te_icl_cc": ["te_icl_cc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"15-49M_te_icl_cc": ["te_icl_cc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"15-49F_te_icl_cc": ["te_icl_cc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"50-59M_te_icl_cc": ["te_icl_cc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"50-59F_te_icl_cc": ["te_icl_cc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"60+M_te_icl_cc": ["te_icl_cc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"60+F_te_icl_cc": ["te_icl_cc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    # Immune escape to compensated cirrhosis
    {"0-4M_te_ie_cc": ["te_ie_cc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"0-4F_te_ie_cc": ["te_ie_cc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"5-14M_te_ie_cc": ["te_ie_cc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"5-14F_te_ie_cc": ["te_ie_cc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"15-49M_te_ie_cc": ["te_ie_cc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"15-49F_te_ie_cc": ["te_ie_cc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"50-59M_te_ie_cc": ["te_ie_cc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"50-59F_te_ie_cc": ["te_ie_cc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"60+M_te_ie_cc": ["te_ie_cc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    {"60+F_te_ie_cc": ["te_ie_cc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.1]},
    # Compensated cirrhosis to decompensated cirrhosis
    {"0-4M_te_cc_dc": ["te_cc_dc", "0-4M", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"0-4F_te_cc_dc": ["te_cc_dc", "0-4F", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"5-14M_te_cc_dc": ["te_cc_dc", "5-14M", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"5-14F_te_cc_dc": ["te_cc_dc", "5-14F", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"15-49M_te_cc_dc": ["te_cc_dc", "15-49M", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"15-49F_te_cc_dc": ["te_cc_dc", "15-49F", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"50-59M_te_cc_dc": ["te_cc_dc", "50-59M", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"50-59F_te_cc_dc": ["te_cc_dc", "50-59F", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"60+M_te_cc_dc": ["te_cc_dc", "60+M", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    {"60+F_te_cc_dc": ["te_cc_dc", "60+M", 1990, 2021, 2023, 0.5, 0.5, 0.35]},
    # Decompensated cirrhosis to compensated cirrhosis
    {"0-4M_te_dc_cc": ["te_dc_cc", "0-4M", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"0-4F_te_dc_cc": ["te_dc_cc", "0-4F", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"5-14M_te_dc_cc": ["te_dc_cc", "5-14M", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"5-14F_te_dc_cc": ["te_dc_cc", "5-14F", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"15-49M_te_dc_cc": ["te_dc_cc", "15-49M", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"15-49F_te_dc_cc": ["te_dc_cc", "15-49F", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"50-59M_te_dc_cc": ["te_dc_cc", "50-59M", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"50-59F_te_dc_cc": ["te_dc_cc", "50-59F", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"60+M_te_dc_cc": ["te_dc_cc", "60+M", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    {"60+F_te_dc_cc": ["te_dc_cc", "60+F", 1990, 2021, 2023, 0.15, 0.15, 0.25]},
    # Immune clearance to hepatocellular carcinoma
    {"0-4M_te_icl_hcc": ["te_icl_hcc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"0-4F_te_icl_hcc": ["te_icl_hcc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"5-14M_te_icl_hcc": ["te_icl_hcc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"5-14F_te_icl_hcc": ["te_icl_hcc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"15-49M_te_icl_hcc": ["te_icl_hcc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"15-49F_te_icl_hcc": ["te_icl_hcc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"50-59M_te_icl_hcc": ["te_icl_hcc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"50-59F_te_icl_hcc": ["te_icl_hcc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"60+M_te_icl_hcc": ["te_icl_hcc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"60+F_te_icl_hcc": ["te_icl_hcc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    # Immune control to hepatocellular carcinoma
    {"0-4M_te_ict_hcc": ["te_ict_hcc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"0-4F_te_ict_hcc": ["te_ict_hcc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"5-14M_te_ict_hcc": ["te_ict_hcc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"5-14F_te_ict_hcc": ["te_ict_hcc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"15-49M_te_ict_hcc": ["te_ict_hcc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"15-49F_te_ict_hcc": ["te_ict_hcc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"50-59M_te_ict_hcc": ["te_ict_hcc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"50-59F_te_ict_hcc": ["te_ict_hcc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"60+M_te_ict_hcc": ["te_ict_hcc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"60+F_te_ict_hcc": ["te_ict_hcc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    # Immune escape to hepatocellular carcinoma
    {"0-4M_te_ie_hcc": ["te_ie_hcc", "0-4M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"0-4F_te_ie_hcc": ["te_ie_hcc", "0-4F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"5-14M_te_ie_hcc": ["te_ie_hcc", "5-14M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"5-14F_te_ie_hcc": ["te_ie_hcc", "5-14F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"15-49M_te_ie_hcc": ["te_ie_hcc", "15-49M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"15-49F_te_ie_hcc": ["te_ie_hcc", "15-49F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"50-59M_te_ie_hcc": ["te_ie_hcc", "50-59M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"50-59F_te_ie_hcc": ["te_ie_hcc", "50-59F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"60+M_te_ie_hcc": ["te_ie_hcc", "60+M", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    {"60+F_te_ie_hcc": ["te_ie_hcc", "60+F", 1990, 2021, 2023, 0.2, 0.2, 0.05]},
    # Compensated cirrhosis to hepatocellular carcinoma
    {"0-4M_te_cc_hcc": ["te_cc_hcc", "0-4M", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"0-4F_te_cc_hcc": ["te_cc_hcc", "0-4F", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"5-14M_te_cc_hcc": ["te_cc_hcc", "5-14M", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"5-14F_te_cc_hcc": ["te_cc_hcc", "5-14F", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"15-49M_te_cc_hcc": ["te_cc_hcc", "15-49M", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"15-49F_te_cc_hcc": ["te_cc_hcc", "15-49F", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"50-59M_te_cc_hcc": ["te_cc_hcc", "50-59M", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"50-59F_te_cc_hcc": ["te_cc_hcc", "50-59F", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"60+M_te_cc_hcc": ["te_cc_hcc", "60+M", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    {"60+F_te_cc_hcc": ["te_cc_hcc", "60+F", 1990, 2021, 2023, 0.7, 0.7, 0.5]},
    # Decompensated cirrhosis to hepatocellular carcinoma
    {"0-4M_te_dc_hcc": ["te_dc_hcc", "0-4M", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"0-4F_te_dc_hcc": ["te_dc_hcc", "0-4F", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"5-14M_te_dc_hcc": ["te_dc_hcc", "5-14M", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"5-14F_te_dc_hcc": ["te_dc_hcc", "5-14F", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"15-49M_te_dc_hcc": ["te_dc_hcc", "15-49M", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"15-49F_te_dc_hcc": ["te_dc_hcc", "15-49F", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"50-59M_te_dc_hcc": ["te_dc_hcc", "50-59M", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"50-59F_te_dc_hcc": ["te_dc_hcc", "50-59F", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"60+M_te_dc_hcc": ["te_dc_hcc", "60+M", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    {"60+F_te_dc_hcc": ["te_dc_hcc", "60+F", 1990, 2021, 2023, 0.75, 0.75, 0.5]},
    # Mortality from decompensated cirrhosis
    {"0-4M_te_m_dc": ["te_m_dc", "0-4M", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"0-4F_te_m_dc": ["te_m_dc", "0-4F", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"5-14M_te_m_dc": ["te_m_dc", "5-14M", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"5-14F_te_m_dc": ["te_m_dc", "5-14F", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"15-49M_te_m_dc": ["te_m_dc", "15-49M", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"15-49F_te_m_dc": ["te_m_dc", "15-49F", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"50-59M_te_m_dc": ["te_m_dc", "50-59M", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"50-59F_te_m_dc": ["te_m_dc", "50-59F", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"60+M_te_m_dc": ["te_m_dc", "60+M", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    {"60+F_te_m_dc": ["te_m_dc", "60+F", 1990, 2021, 2023, 0.5, 0.5, 0.25]},
    # Mortality from hepatocellular carcinoma
    {"0-4M_te_m_hcc": ["te_m_hcc", "0-4M", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"0-4F_te_m_hcc": ["te_m_hcc", "0-4F", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"5-14M_te_m_hcc": ["te_m_hcc", "5-14M", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"5-14F_te_m_hcc": ["te_m_hcc", "5-14F", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"15-49M_te_m_hcc": ["te_m_hcc", "15-49M", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"15-49F_te_m_hcc": ["te_m_hcc", "15-49F", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"50-59M_te_m_hcc": ["te_m_hcc", "50-59M", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"50-59F_te_m_hcc": ["te_m_hcc", "50-59F", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"60+M_te_m_hcc": ["te_m_hcc", "60+M", 1990, 2021, 2023, 0.8, 0.8, 0.5]},
    {"60+F_te_m_hcc": ["te_m_hcc", "60+F", 1990, 2021, 2023, 0.8, 0.8, 0.5]}
]


# Sensitivity analysis
def par_sens(F, db_bl, db_s1, calib, par_dict, runs):
    out_res = [
        {"pop_0_4M": ["alive", {"0-4M": ["0-4M"]}, "sum"]},
        {"pop_5_14M": ["alive", {"5-14M": ["5-14M"]}, "sum"]},
        {"pop_15_49M": ["alive", {"15-49M": ["15-49M"]}, "sum"]},
        {"pop_50_59M": ["alive", {"50-59M": ["50-59M"]}, "sum"]},
        {"pop_60M": ["alive", {"60+M": ["60+M"]}, "sum"]},

        {"pop_0_4F": ["alive", {"0-4M": ["0-4F"]}, "sum"]},
        {"pop_5_14F": ["alive", {"5-14M": ["5-14F"]}, "sum"]},
        {"pop_15_49F": ["alive", {"15-49F": ["15-49F"]}, "sum"]},
        {"pop_50_59F": ["alive", {"50-59F": ["50-59F"]}, "sum"]},
        {"pop_60F": ["alive", {"60+F": ["60+F"]}, "sum"]},

        {"yld_0_4M": ["yld", {"0-4M": ["0-4M"]}, "sum"]},
        {"yld_5_14M": ["yld", {"5-14M": ["5-14M"]}, "sum"]},
        {"yld_15_49M": ["yld", {"15-49M": ["15-49M"]}, "sum"]},
        {"yld_50_59M": ["yld", {"50-59M": ["50-59M"]}, "sum"]},
        {"yld_60M": ["yld", {"60+M": ["60+M"]}, "sum"]},

        {"yld_0_4F": ["yld", {"0-4F": ["0-4F"]}, "sum"]},
        {"yld_5_14F": ["yld", {"5-14F": ["5-14F"]}, "sum"]},
        {"yld_15_49F": ["yld", {"15-49F": ["15-49F"]}, "sum"]},
        {"yld_50_59F": ["yld", {"50-59F": ["50-59F"]}, "sum"]},
        {"yld_60F": ["yld", {"60+F": ["60+F"]}, "sum"]},

        {"yll_0_4M": ["yll", {"0-4M": ["0-4M"]}, "sum"]},
        {"yll_5_14M": ["yll", {"5-14M": ["5-14M"]}, "sum"]},
        {"yll_15_49M": ["yll", {"15-49M": ["15-49M"]}, "sum"]},
        {"yll_50_59M": ["yll", {"50-59M": ["50-59M"]}, "sum"]},
        {"yll_60M": ["yll", {"60+M": ["60+M"]}, "sum"]},

        {"yll_0_4F": ["yll", {"0-4F": ["0-4F"]}, "sum"]},
        {"yll_5_14F": ["yll", {"5-14F": ["5-14F"]}, "sum"]},
        {"yll_15_49F": ["yll", {"15-49F": ["15-49F"]}, "sum"]},
        {"yll_50_59F": ["yll", {"50-59F": ["50-59F"]}, "sum"]},
        {"yll_60F": ["yll", {"60+F": ["60+F"]}, "sum"]},

        {"mort_0_4M": [":dd_hbv", {"0-4M": ["0-4M"]}, "sum"]},
        {"mort_5_14M": [":dd_hbv", {"5-14M": ["5-14M"]}, "sum"]},
        {"mort_15_49M": [":dd_hbv", {"15-49M": ["15-49M"]}, "sum"]},
        {"mort_50_59M": [":dd_hbv", {"50-59M": ["50-59M"]}, "sum"]},
        {"mort_60M": [":dd_hbv", {"60+M": ["60+M"]}, "sum"]},

        {"mort_0_4F": [":dd_hbv", {"0-4F": ["0-4F"]}, "sum"]},
        {"mort_5_14F": [":dd_hbv", {"5-14F": ["5-14F"]}, "sum"]},
        {"mort_15_49F": [":dd_hbv", {"15-49F": ["15-49F"]}, "sum"]},
        {"mort_50_59F": [":dd_hbv", {"50-59F": ["50-59F"]}, "sum"]},
        {"mort_60F": [":dd_hbv", {"60+F": ["60+F"]}, "sum"]},

        {"yll": ["yll", "total", "sum"]},
        {"yld": ["yld", "total", "sum"]},
        {"pop": ["alive", "total", "sum"]},
        {"pop_u5": ["alive", {"Under 5": ["0-4M", "0-4F"]}, "sum"]},
        {"prev": ["prev", "total", "weighted"]},
        {"chb_pop": ["chb_pop", "total", "weighted"]},
        {"prev_u5": ["prev", {"Under 5": ["0-4M", "0-4F"]}, "weighted"]},
        {"mort": [":dd_hbv", "total", "sum"]},
        {"hcc_inc": ["flw_hcc", "total", "sum"]},
        {"chb_inc": ["tot_inc", "total", "sum"]},
        {"hbe_preg": ["eag_ott", "15-49F", "sum"]},
        {"births": ["b_rate", {"Under 5": ["0-4M", "0-4F"]}, "sum"]},
        {"bd_cov": ["bd", {"Under 5": ["0-4M", "0-4F"]}, "weighted"]},
        {"hb3_cov": ["hb3", {"Under 5": ["0-4M", "0-4F"]}, "weighted"]},
        {"dx_rate": ["tot_dx", "total", "sum"]},
        {"tx_cov": ["treat", "total", "sum"]},
        {"dm_dx": [[{"dm_dx": "it_dx+icl_dx+ict_dx+ie_dx+cc_dx+dc_dx+hcc_dx"}], "total", "sum"]},
        {"dm_tx": [[{"dm_tx": "icl_tx+ict_tx+ie_tx+cc_tx+dc_tx+hcc_tx"}], "total", "sum"]},
        {"pop_hcc": [[{"pop_hcc": "cc_dx+cc_tx+dc_dx+dc_tx"}], "total", "sum"]},
        {"tgt_hcc": [[{"tgt_hcc": "it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}],
                     {"50+": ["50-59M", "50-59F", "60+M", "60+F"]}, "sum"]},
        {"tgt_hcc_b": [[{"tgt_hcc": "it_dx+icl_dx+icl_tx+ict_dx+ict_tx+ie_dx+ie_tx"}], {"40+": ["15-49M", "15-49F"]},
                       "sum"]},
        {"hsp_tx": [[{"hsp_tx": "dc_tx+hcc_tx"}], "total", "sum"]},
        {"hsp_utx": [[{"hsp_utx": "dc+dc_dx+hcc+hcc_dx"}], "total", "sum"]},
        {"dx_prop": ["diag_cov", "total", "weighted"]},
        {"tx_prop": ["treat_cov", "total", "weighted"]},
        {"mav_n": ["mav_births", {"Under 5": ["0-4M", "0-4F"]}, "sum"]},
        {"prg_scr": ["preg_scr_num", "15-49F", "sum"]},
        {"prg_hrs": ["preg_scr_num", "15-49F", "sum"]}]

    # Load and run the BL model
    D_bl = at.ProjectData.from_spreadsheet("databooks/" + db_bl, framework=F)
    P_bl = at.Project(framework=F, databook="databooks/" + db_bl, sim_start=1990, sim_end=2051, sim_dt=0.25, do_run=False)
    cal_bl = P_bl.make_parset()
    cal_bl.load_calibration("calibrations/" + calib)

    bl = P_bl.run_sim(parset=cal_bl, result_name="Status Quo")

    # Perform parameter sensitivity analysis (PSA) BL
    np.random.seed(25012024)
    psa_res_bl = P_bl.run_sampled_sims(cal_bl, n_samples=runs)

    # Generate BL output
    central_est_bl = {}
    for i in out_res:
        for key, val in i.items():
            df = at.PlotData(bl, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1).series[0].vals
            central_est_bl[key] = df


    # Generate S1 output
    D_s1 = at.ProjectData.from_spreadsheet("databooks/" + db_s1, framework=F)
    P_s1 = at.Project(framework=F, databook="databooks/" + db_s1, sim_start=1990, sim_end=2051, sim_dt=0.25, do_run=False)
    cal_s1 = P_s1.make_parset()
    cal_s1.load_calibration("calibrations/" + calib)

    s1 = P_s1.run_sim(parset=cal_s1, result_name="Scenario 1: 2030 Target")

    # Perform parameter sensitivity analysis (PSA) S1
    psa_res_s1 = P_s1.run_sampled_sims(cal_s1, n_samples=runs)

    # Generate S1 output
    central_est_s1 = {}
    for i in out_res:
        for key, val in i.items():
            df = at.PlotData(s1, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1).series[0].vals
            central_est_s1[key] = df

    store_runs_bl = {}
    store_runs_s1 = {}

    for output in out_res:
        for key, val in output.items():
            # BL
            mapping_function_bl = lambda x: at.PlotData(x, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1)
            ensemble_bl = at.Ensemble(mapping_function=mapping_function_bl)
            ensemble_bl.update(psa_res_bl)
            df_bl = pd.DataFrame([d.series[0].vals for d in ensemble_bl.samples])
            store_runs_bl[key] = np.array(df_bl).T

            # S1
            mapping_function_s1 = lambda x: at.PlotData(x, outputs=val[0], pops=val[1], pop_aggregation=val[2], t_bins=1)
            ensemble_s1 = at.Ensemble(mapping_function=mapping_function_s1)
            ensemble_s1.update(psa_res_s1)
            df_s1 = pd.DataFrame([d.series[0].vals for d in ensemble_s1.samples])
            store_runs_s1[key] = np.array(df_s1).T

    return central_est_bl, central_est_s1, store_runs_bl, store_runs_s1


def parsens_econ(psens_bl, psens_s1, cost_data, disc_rate):

    if disc_rate > 1:
        disc_rate = disc_rate / 100
    else:
        disc_rate = disc_rate

    discount = np.zeros((len(np.arange(1990, 2051, 1)), 3))
    discount[:, 0] = np.arange(1990, 2051, 1)

    for idx, val in enumerate(discount[:, 0]):
        if val <= 2023:
            discount[idx, 1] = 1
            discount[idx, 2] = 0  # use this for high risk screening as a check
        else:
            discount[idx, 1:] = (1 + disc_rate) ** - (val - 2023)

    vax_costs = pd.read_excel(cost_data, sheet_name="vax")
    bd_vax = vax_costs.loc[0, 'China']
    hb3_vax = vax_costs.loc[1, 'China']
    hrs_cost = vax_costs.loc[10, 'China']
    mav_hbig = vax_costs.loc[12, 'China']

    care_costs = pd.read_excel(cost_data, sheet_name="care")
    dx_cost =  care_costs.loc[0, 'China']
    tx_cost = care_costs.loc[3, 'China']
    dx_dmc = care_costs.loc[4, 'China']
    tx_dmc = care_costs.loc[5, 'China']
    hosp_cost = care_costs.loc[7, 'China']
    hcc_cost = care_costs.loc[6, 'China']
    hcc_prp = care_costs.loc[8, 'China']

    ## Vaccine and vowel blocking plus screening costs (including HBIG)

    bd_cost_bl, hb3_cost_bl, bd_cost_s1, hb3_cost_s1 = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    bd_cost_bl[:, 0] = psens_bl["bd_cov"][:] * psens_bl["births"][:] * bd_vax * discount[:, 1]
    hb3_cost_bl[:, 0] = psens_bl["hb3_cov"][:] * psens_bl["births"][:] * hb3_vax * discount[:, 1]
    bd_cost_s1[:, 0] = psens_s1["bd_cov"][:] * psens_s1["births"][:] * bd_vax * discount[:, 1]
    hb3_cost_s1[:, 0] = psens_s1["hb3_cov"][:] * psens_s1["births"][:] * hb3_vax * discount[:, 1]

    bl_cost_mav, s1_cost_mav = np.zeros((61, 1)), np.zeros((61, 1))
    bl_cost_mav[:, 0] = (psens_bl["mav_n"][:] * mav_hbig * discount[:, 1]) + (psens_bl["prg_scr"][:] * dx_cost * discount[:,1])+ (psens_bl["prg_hrs"][:] * hrs_cost * discount[:, 1])
    s1_cost_mav[:, 0] = (psens_s1["mav_n"][:] * mav_hbig * discount[:, 1]) + (psens_s1["prg_scr"][:] * dx_cost * discount[:, 1]) + (psens_s1["prg_hrs"][:] * hrs_cost * discount[:, 1])

    ## Diagnosis cost
    bl_dx_inc, s1_dx_inc = np.zeros((61, 1)), np.zeros((61, 1))

    for i in range(len(psens_bl["dx_prop"])):
        if i < 1:
            bl_dx_inc[i, 0] = psens_bl["dx_prop"][i]
            s1_dx_inc[i, 0] = psens_s1["dx_prop"][i]
        else:
            bl_dx_inc[i, 0] = max(psens_bl["dx_prop"][i] - psens_bl["dx_prop"][i - 1], 0)
            s1_dx_inc[i, 0] = max(psens_s1["dx_prop"][i] - psens_s1["dx_prop"][i - 1], 0)

    dx_costb_bl, dx_costb_s1 = np.zeros((61, 1)), np.zeros((61, 1))

    for yr in range(len(bl_dx_inc)):
        dx_costb_bl[yr, 0] = (psens_bl["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * bl_dx_inc[yr, 0] * (psens_bl["pop"][yr] - psens_bl["dx_rate"][yr]) * discount[yr, 1])
        dx_costb_s1[yr, 0] = (psens_s1["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * s1_dx_inc[yr, 0] * (psens_s1["pop"][yr] - psens_s1["dx_rate"][yr]) * discount[yr, 1])

    ## Treatment cost
    bl_cost_tx, s1_cost_tx = np.zeros((61, 1)), np.zeros((61, 1))

    bl_cost_tx[:, 0] = psens_bl["tx_cov"][:] * tx_cost * discount[:, 1]
    s1_cost_tx[:, 0] = psens_s1["tx_cov"][:] * tx_cost * discount[:, 1]

    ##  Direct medical cost
    dmc_bl = bd_cost_bl + hb3_cost_bl + bl_cost_mav + dx_costb_bl + bl_cost_tx
    dmc_s1 = bd_cost_s1 + hb3_cost_s1 + s1_cost_mav + dx_costb_s1 + s1_cost_tx

    ## Disease management cost
    util = 0.25
    tx_hosp = 0.5

    manc_bl, manc_s1 = np.zeros((61, 1)), np.zeros((61, 1))
    hospc_bl, hospc_s1 = np.zeros((61, 1)), np.zeros((61, 1))
    hccs_bl, hccs_s1 = np.zeros((61, 1)), np.zeros((61, 1))


    manc_bl[:, 0] = ((psens_bl["dm_dx"][:] * dx_dmc * util) + (psens_bl["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]
    manc_s1[:, 0] = ((psens_s1["dm_dx"][:] * dx_dmc * util) + (psens_s1["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]

    ## Hospitalization cost
    hospc_bl[:, 0] = ((psens_bl["hsp_utx"][:] * hosp_cost * util) + (psens_bl["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    hospc_s1[:, 0] = ((psens_s1["hsp_utx"][:] * hosp_cost * util) + (psens_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]

    ## Cost of HCC surveillance
    hccs_bl[:, 0] = ((psens_bl["pop_hcc"][:] * hcc_cost * util) + (psens_bl["tgt_hcc"][:] * hcc_cost * util) + (psens_bl["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]
    hccs_s1[:, 0] = ((psens_s1["pop_hcc"][:] * hcc_cost * util) + (psens_s1["tgt_hcc"][:] * hcc_cost * util) + (psens_s1["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]

    ## Indirect medical costs\
    imc_bl = manc_bl + hospc_bl + hccs_bl
    imc_s1 = manc_s1 + hospc_s1 + hccs_s1

    ## Loss of productivity, YLL, YPLL, DALYs
    prod_costs = pd.read_excel(cost_data, sheet_name="emp_gdp_lex")
    etp_ratio = prod_costs.loc[0,'China']
    gdp = prod_costs.loc[1, 'China']
    life_exp = prod_costs.loc[2, 'China']

    ## GDP growth
    gdp_grw = np.zeros((len(np.arange(1990, 2051, 1)), 4))
    gdp_grw[:, 0] = np.arange(1990, 2051, 1)
    gdp_grw[:, 1:4] = gdp

    for i, val in enumerate(gdp_grw[:, 0]):
        if val > 2023:
            gdp_grw[i, 1] = gdp_grw[i - 1, 1] * 1.00
            gdp_grw[i, 2] = gdp_grw[i - 1, 2] * 1.015
            gdp_grw[i, 3] = gdp_grw[i - 1, 3] * 1.03

    age_of_deaths = np.array([0.01, 0.031, 0.253, 0.341, 0.365])
    prop_leaving_age_categories = np.array([1 / 15, 1 / 15, 1 / 20, 1 / 15])
    all_cause_mort = np.array([0.003, 0.0013, 0.0022, 0.0103, (1 / life_exp)])

    # Baseline
    cbl_deaths = np.zeros((len(psens_bl["mort"]), 2))
    cbl_deaths[:, 0] = np.arange(1990, 2051, 1)
    cbl_deaths[:, 1] = psens_bl["mort"]

    for idx, val in enumerate(cbl_deaths[:, 0]):
        if val < 2023:
            cbl_deaths[idx, 1] = 0

    ghosts_cbl = np.zeros((len(cbl_deaths), len(age_of_deaths)))
    ghosts_cbl[0, :] = cbl_deaths[0, 1] * age_of_deaths

    for t in range(1, len(cbl_deaths)):
        ppl_who_age = ghosts_cbl[t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
        ghosts_cbl[t, 0] = max(0, ghosts_cbl[t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_cbl[t - 1, 0])
        ghosts_cbl[t, 1] = max(0, ghosts_cbl[t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_cbl[t - 1, 1])
        ghosts_cbl[t, 2] = max(0, ghosts_cbl[t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_cbl[t - 1, 2])
        ghosts_cbl[t, 3] = max(0, ghosts_cbl[t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_cbl[t - 1, 3])
        ghosts_cbl[t, 4] = max(0, ghosts_cbl[t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_cbl[t - 1, 4])

        ghosts_cbl[t, :] = ghosts_cbl[t, :] + cbl_deaths[t, 1] * age_of_deaths

    ## Scenario 1 (Point estimated)
    cs1_deaths = np.zeros((len(psens_s1["mort"]), 2))
    cs1_deaths[:, 0] = np.arange(1990, 2051, 1)
    cs1_deaths[:, 1] = psens_s1["mort"]

    for idx, val in enumerate(cs1_deaths[:, 0]):
        if val < 2023:
            cs1_deaths[idx, 1] = 0

    ghosts_cs1 = np.zeros((len(cs1_deaths), len(age_of_deaths)))
    ghosts_cs1[0, :] = cs1_deaths[0, 1] * age_of_deaths

    for t in range(1, len(cs1_deaths)):
        ppl_who_age = ghosts_cs1[t, 0:len(prop_leaving_age_categories)] * prop_leaving_age_categories
        ghosts_cs1[t, 0] = max(0, ghosts_cs1[t - 1, 0] - ppl_who_age[0] - all_cause_mort[0] * ghosts_cs1[t - 1, 0])
        ghosts_cs1[t, 1] = max(0, ghosts_cs1[t - 1, 1] - ppl_who_age[1] + ppl_who_age[0] - all_cause_mort[1] * ghosts_cs1[t - 1, 1])
        ghosts_cs1[t, 2] = max(0, ghosts_cs1[t - 1, 2] - ppl_who_age[2] + ppl_who_age[1] - all_cause_mort[2] * ghosts_cs1[t - 1, 2])
        ghosts_cs1[t, 3] = max(0, ghosts_cs1[t - 1, 3] - ppl_who_age[3] + ppl_who_age[2] - all_cause_mort[3] * ghosts_cs1[t - 1, 3])
        ghosts_cs1[t, 4] = max(0, ghosts_cs1[t - 1, 4] + ppl_who_age[3] - all_cause_mort[4] * ghosts_cs1[t - 1, 4])

        ghosts_cs1[t, :] = ghosts_cs1[t, :] + cs1_deaths[t, 1] * age_of_deaths

    # DALYs
    bl_yll = np.sum(ghosts_cbl[:, 0:5], axis=1) * discount[:, 1]
    s1_yll = np.sum(ghosts_cs1[:, 0:5], axis=1) * discount[:, 1]

    bl_dalys = bl_yll + psens_bl["yld"]
    s1_dalys = s1_yll + psens_s1["yld"]

    # Productivity
    bl_prod = np.sum(ghosts_cbl[:, 2:4], axis=1) * discount[:, 1] * etp_ratio * gdp_grw[:, 3]
    s1_prod = np.sum(ghosts_cs1[:, 2:4], axis=1) * discount[:, 1] * etp_ratio * gdp_grw[:, 3]

    ## NEB
    bl_tc_ins = bl_prod[:] + dmc_bl[:, 0] + imc_bl[:, 0]
    s1_tc_ins = s1_prod[:] + dmc_s1[:, 0] + imc_s1[:, 0]

    bl_tc_ins = np.nan_to_num(bl_tc_ins, nan=0)
    s1_tc_ins = np.nan_to_num(s1_tc_ins, nan=0)

    bl_cum_tc = np.zeros(np.shape(bl_tc_ins))
    s1_cum_tc = np.zeros(np.shape(s1_tc_ins))

    for i in range(len(bl_tc_ins)):
        if i < 1:
            bl_cum_tc[i] = bl_tc_ins[i]
            s1_cum_tc[i] = s1_tc_ins[i]
        else:
            bl_cum_tc[i] = bl_cum_tc[i - 1] + bl_tc_ins[i]
            s1_cum_tc[i] = s1_cum_tc[i - 1] + s1_tc_ins[i]

    s1_neb = bl_cum_tc - s1_cum_tc

    ## ICER
    bl_dc_ins = dmc_bl[:, 0] + imc_bl[:, 0]
    s1_dc_ins = dmc_s1[:, 0] + imc_s1[:, 0]

    bl_dc_ins = np.nan_to_num(bl_dc_ins, nan=0)
    s1_dc_ins = np.nan_to_num(s1_dc_ins, nan=0)
    bl_dalys = np.nan_to_num(bl_dalys, nan=0)
    s1_dalys = np.nan_to_num(s1_dalys, nan=0)

    bl_cum_dc = np.zeros(np.shape(bl_dc_ins))
    s1_cum_dc = np.zeros(np.shape(s1_dc_ins))
    bl_cum_daly = np.zeros(np.shape(bl_dalys))
    s1_cum_daly = np.zeros(np.shape(s1_dalys))

    for i in range(len(bl_dc_ins)):
        if i < 1:
            bl_cum_dc[i] = bl_dc_ins[i]
            bl_cum_daly[i] = bl_dalys[i]

            s1_cum_dc[i] = s1_dc_ins[i]
            s1_cum_daly[i] = s1_dalys[i]
        else:
            bl_cum_dc[i] = bl_dc_ins[i] + bl_cum_dc[i - 1]
            bl_cum_daly[i] = bl_dalys[i] + bl_cum_daly[i - 1]

            s1_cum_dc[i] = s1_dc_ins[i] + s1_cum_dc[i - 1]
            s1_cum_daly[i] = s1_dalys[i] + s1_cum_daly[i - 1]

    s1_icer = -(bl_cum_dc - s1_cum_dc) / (bl_cum_daly - s1_cum_daly)
    s1_icer = np.nan_to_num(s1_icer, 0)

    ## Return results needed for tabulation (within the input dictionaries for simplicity)

    psens_s1["daly_bl"] = bl_dalys
    psens_s1["daly_s1"] = s1_dalys

    psens_s1["cdaly_cum_bl"] = bl_cum_daly
    psens_s1["cdaly_cum_s1"] = s1_cum_daly

    psens_s1["cdirc_bl"] = bl_dc_ins
    psens_s1["cdirc_s1"] = s1_dc_ins

    psens_s1["cdirc_cum_bl"] = bl_cum_dc
    psens_s1["cdirc_cum_s1"] = s1_cum_dc

    psens_s1["prod_bl"] = bl_prod
    psens_s1["prod_s1"] = s1_prod

    psens_s1["icer"] = s1_icer
    psens_s1["neb"] = s1_neb

    return psens_s1

def econsens_econ(bl_cent, cent_s1, econ, cost_data, disc_rate):

    # Discounting Array
    if disc_rate > 1:
        disc_rate = disc_rate / 100
    else:
        disc_rate = disc_rate

    discount = np.zeros((len(np.arange(1990, 2051, 1)), 3))
    discount[:, 0] = np.arange(1990, 2051, 1)

    for idx, val in enumerate(discount[:, 0]):
        if val <= 2023:
            discount[idx, 1] = 1
            discount[idx, 2] = 0  # use this for high risk screening as a check
        else:
            discount[idx, 1:] = (1 + disc_rate) ** - (val - 2023)

    vax_costs = pd.read_excel(cost_data, sheet_name="vax")
    care_costs = pd.read_excel(cost_data, sheet_name="care")

    mav_hbig = vax_costs.loc[12, 'China']
    dx_cost = care_costs.loc[0, 'China']
    dx_cost_lb = care_costs.loc[2, 'China']
    dx_cost_ub = care_costs.loc[1, 'China']
    hrs_cost = vax_costs.loc[10, 'China']
    hrs_cost_lb = hrs_cost * 0.5
    hrs_cost_ub = hrs_cost * 2

    """Diagnose costs"""
    ## Vowel blocking plus screening costs
    bl_cost_mav_pe, s1_cost_mav_pe,bl_cost_mav_lb, s1_cost_mav_lb,bl_cost_mav_ub, s1_cost_mav_ub = np.zeros((61, 1)), np.zeros((61, 1)),np.zeros((61, 1)), np.zeros((61, 1)),np.zeros((61, 1)), np.zeros((61, 1))

    bl_cost_mav_pe[:, 0] = (bl_cent["mav_n"][:] * mav_hbig * discount[:, 1]) + (bl_cent["prg_scr"][:] * dx_cost * discount[:, 1]) + (bl_cent["prg_hrs"][:] * hrs_cost * discount[:, 1])
    s1_cost_mav_pe[:, 0] = (cent_s1["mav_n"][:] * mav_hbig * discount[:, 1]) + (cent_s1["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost * discount[:, 1])

    ## half
    bl_cost_mav_lb[:, 0] = (bl_cent["mav_n"][:] * mav_hbig * discount[:, 1]) + (
                bl_cent["prg_scr"][:] * dx_cost_lb * discount[:, 1]) + (bl_cent["prg_hrs"][:] * hrs_cost_lb * discount[:, 1])
    s1_cost_mav_lb[:, 0] = (cent_s1["mav_n"][:] * mav_hbig * discount[:, 1]) + (
                cent_s1["prg_scr"][:] * dx_cost_lb * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost_lb * discount[:, 1])

    ## double
    bl_cost_mav_ub[:, 0] = (bl_cent["mav_n"][:] * mav_hbig * discount[:, 1]) + (
            bl_cent["prg_scr"][:] * dx_cost_ub * discount[:, 1]) + (bl_cent["prg_hrs"][:] * hrs_cost_ub * discount[:, 1])
    s1_cost_mav_ub[:, 0] = (cent_s1["mav_n"][:] * mav_hbig * discount[:, 1]) + (
            cent_s1["prg_scr"][:] * dx_cost_ub * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost_ub * discount[:, 1])

    ## Diagnosis costs
    bl_dx_inc, s1_dx_inc = np.zeros((61, 1)), np.zeros((61, 1))

    for i in range(len(bl_cent["dx_prop"])):
        if i < 1:
            bl_dx_inc[i, 0] = bl_cent["dx_prop"][i]
            s1_dx_inc[i, 0] = cent_s1["dx_prop"][i]
        else:
            bl_dx_inc[i, 0] = max(bl_cent["dx_prop"][i] - bl_cent["dx_prop"][i - 1], 0)
            s1_dx_inc[i, 0] = max(cent_s1["dx_prop"][i] - cent_s1["dx_prop"][i - 1], 0)

    dx_costb_bl_pe, dx_costb_s1_pe,dx_costb_bl_lb, dx_costb_s1_lb,dx_costb_bl_ub, dx_costb_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)),np.zeros((61, 1)), np.zeros((61, 1)),np.zeros((61, 1)), np.zeros((61, 1))

    for yr in range(len(bl_dx_inc)):
        dx_costb_bl_pe[yr, 0] = (bl_cent["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * bl_dx_inc[yr, 0] * (bl_cent["pop"][yr] - bl_cent["dx_rate"][yr]) * discount[yr, 1])
        dx_costb_bl_pe[yr, 0] = (cent_s1["dx_rate"][yr] * dx_cost * discount[yr, 1]) + (dx_cost * s1_dx_inc[yr, 0] * (cent_s1["pop"][yr] - cent_s1["dx_rate"][yr]) * discount[yr, 1])

    ## half
        dx_costb_bl_lb[yr, 0] = (bl_cent["dx_rate"][yr] * dx_cost_lb * discount[yr, 1]) + (
                    dx_cost_lb * bl_dx_inc[yr, 0] * (bl_cent["pop"][yr] - bl_cent["dx_rate"][yr]) * discount[yr, 1])
        dx_costb_bl_lb[yr, 0] = (cent_s1["dx_rate"][yr] * dx_cost_lb * discount[yr, 1]) + (
                    dx_cost_lb * s1_dx_inc[yr, 0] * (cent_s1["pop"][yr] - cent_s1["dx_rate"][yr]) * discount[yr, 1])

    ## double
        dx_costb_bl_ub[yr, 0] = (bl_cent["dx_rate"][yr] * dx_cost_ub * discount[yr, 1]) + (
                dx_cost_ub * bl_dx_inc[yr, 0] * (bl_cent["pop"][yr] - bl_cent["dx_rate"][yr]) * discount[yr, 1])
        dx_costb_bl_ub[yr, 0] = (cent_s1["dx_rate"][yr] * dx_cost_ub * discount[yr, 1]) + (
                dx_cost_ub * s1_dx_inc[yr, 0] * (cent_s1["pop"][yr] - cent_s1["dx_rate"][yr]) * discount[yr, 1])

    # Difference (to be added to direct costs)

    dx_bl_lbd = (bl_cost_mav_lb + dx_costb_bl_lb) - (bl_cost_mav_pe + dx_costb_bl_pe)
    dx_s1_lbd = (s1_cost_mav_lb + dx_costb_s1_lb) - (s1_cost_mav_pe + dx_costb_s1_pe)

    dx_bl_ubd = (bl_cost_mav_ub + dx_costb_bl_ub) - (bl_cost_mav_pe + dx_costb_bl_pe)
    dx_s1_ubd = (s1_cost_mav_ub + dx_costb_s1_ub) - (s1_cost_mav_pe + dx_costb_s1_pe)

    # total direct costs for tabulation, NEB, and ICER
    dcost_lb_bl = econ["cbl_dirc"] + dx_bl_lbd[:, 0]
    dcost_ub_bl = econ["cbl_dirc"] + dx_bl_ubd[:, 0]

    dcost_lb_s1 = econ["cs1_dirc"] + dx_s1_lbd[:, 0]
    dcost_ub_s1 = econ["cs1_dirc"] + dx_s1_ubd[:, 0]

    """Vaccine costs"""
    # Current
    bd_pe = vax_costs.loc[0, 'China']
    hb3_pe = vax_costs.loc[1, 'China']

    bl_bd_pe, s1_bd_pe, bl_hb3_pe, s1_hb3_pe = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    bl_bd_pe[:, 0] = bl_cent["bd_cov"][:] * bl_cent["births"][:] * bd_pe * discount[:, 1]
    bl_hb3_pe[:, 0] = bl_cent["hb3_cov"][:] * bl_cent["births"][:] * hb3_pe * discount[:, 1]
    s1_bd_pe[:, 0] = cent_s1["bd_cov"][:] * cent_s1["births"][:] * bd_pe * discount[:, 1]
    s1_hb3_pe[:, 0] = cent_s1["hb3_cov"][:] * cent_s1["births"][:] * hb3_pe * discount[:, 1]

    # Lower Bound
    bd_lb = vax_costs.loc[6, 'China']
    hb3_lb = vax_costs.loc[8, 'China']

    bl_bd_lb, s1_bd_lb, bl_hb3_lb, s1_hb3_lb = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    bl_bd_lb[:, 0] = bl_cent["bd_cov"][:] * bl_cent["births"][:] * bd_lb * discount[:, 1]
    bl_hb3_lb[:, 0] = bl_cent["hb3_cov"][:] * bl_cent["births"][:] * hb3_lb * discount[:, 1]
    s1_bd_lb[:, 0] = cent_s1["bd_cov"][:] * cent_s1["births"][:] * bd_lb * discount[:, 1]
    s1_hb3_lb[:, 0] = cent_s1["hb3_cov"][:] * cent_s1["births"][:] * hb3_lb * discount[:, 1]

    # Upper Bound
    bd_ub = vax_costs.loc[7, 'China']
    hb3_ub = vax_costs.loc[9, 'China']

    bl_bd_ub, s1_bd_ub, bl_hb3_ub, s1_hb3_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    bl_bd_ub[:, 0] = bl_cent["bd_cov"][:] * bl_cent["births"][:] * bd_ub * discount[:, 1]
    bl_hb3_ub[:, 0] = bl_cent["hb3_cov"][:] * bl_cent["births"][:] * hb3_ub * discount[:, 1]
    s1_bd_ub[:, 0] = cent_s1["bd_cov"][:] * cent_s1["births"][:] * bd_ub * discount[:, 1]
    s1_hb3_ub[:, 0] = cent_s1["hb3_cov"][:] * cent_s1["births"][:] * hb3_ub * discount[:, 1]

    # Difference (to be added to direct costs)

    bl_bd_lbd = bl_bd_lb - bl_bd_pe
    bl_hb3_lbd = bl_hb3_lb - bl_hb3_pe
    s1_bd_lbd = s1_bd_lb - s1_bd_pe
    s1_hb3_lbd = s1_hb3_lb - s1_hb3_pe

    bl_bd_ubd = bl_bd_ub - bl_bd_pe
    bl_hb3_ubd = bl_hb3_ub - bl_hb3_pe
    s1_bd_ubd = s1_bd_ub - s1_bd_pe
    s1_hb3_ubd = s1_hb3_ub - s1_hb3_pe

    # total direct costs for tabulation, NEB, and ICER
    vcost_lb_bl = econ["cbl_dirc"] + (bl_bd_lbd[:, 0] + bl_hb3_lbd[:, 0])
    vcost_ub_bl = econ["cbl_dirc"] + (bl_bd_ubd[:, 0] + bl_hb3_ubd[:, 0])

    vcost_lb_s1 = econ["cs1_dirc"] + (s1_bd_lbd[:, 0] + s1_hb3_lbd[:, 0])
    vcost_ub_s1 = econ["cs1_dirc"] + (s1_bd_ubd[:, 0] + s1_hb3_ubd[:, 0])

    """Treatment costs (TDF + mAVs)"""
    # current
    tx_pe = care_costs.loc[3, 'China']

    tx_bl_pe = bl_cent["tx_cov"][:] * tx_pe * discount[:, 1] ##9.16加【:】
    tx_s1_pe = cent_s1["tx_cov"][:] * tx_pe * discount[:, 1]

    # half
    tx_bl_lb = tx_bl_pe * 0.5
    tx_s1_lb = tx_s1_pe * 0.5

    # imported
    tx_jk = care_costs.loc[9, 'China']

    tx_bl_ub = bl_cent["tx_cov"][:] * tx_jk * discount[:, 1]
    tx_s1_ub = cent_s1["tx_cov"][:] * tx_jk * discount[:, 1]

    """mAV with HBIG + importe"""
    hrs_cost = vax_costs.loc[10, 'China']
    dx_cost = care_costs.loc[0, 'China']
    mav_hbig = vax_costs.loc[12, 'China']
    mav_hbig_ub = vax_costs.loc[15, 'China']
    mav_hbig_lb = vax_costs.loc[16, 'China']

    mv_bl_pe, mv_s1_pe, mv_bl_hbig_lb, mv_s1_hbig_lb, mv_bl_hbig_ub, mv_s1_hbig_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    # current (mav-domestic)
    mv_bl_pe[:, 0] = (bl_cent["mav_n"][:] * mav_hbig * discount[:, 1]) + (bl_cent["prg_scr"][:] * dx_cost * discount[:,1]) + (bl_cent["prg_hrs"][:] * hrs_cost * discount[:, 2])
    mv_s1_pe[:, 0] = (cent_s1["mav_n"][:] * mav_hbig * discount[:, 1]) + (cent_s1["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost * discount[:, 2])

    # lower bound (mav-half)
    mv_bl_hbig_lb[:, 0] = (bl_cent["mav_n"][:] * mav_hbig_lb * discount[:, 1]) + (bl_cent["prg_scr"][:] * dx_cost * discount[:,1])+ (bl_cent["prg_hrs"][:] * hrs_cost * discount[:, 2])
    mv_s1_hbig_lb[:, 0] = (cent_s1["mav_n"][:] * mav_hbig_lb * discount[:, 1]) + (cent_s1["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost * discount[:, 2])

    # upper bound (mav-import)
    mv_bl_hbig_ub[:, 0] = (bl_cent["mav_n"][:] * mav_hbig_ub * discount[:, 1]) + (bl_cent["prg_scr"][:] * dx_cost * discount[:,1])+ (bl_cent["prg_hrs"][:] * hrs_cost * discount[:, 2])
    mv_s1_hbig_ub[:, 0] = (cent_s1["mav_n"][:] * mav_hbig_ub * discount[:, 1]) + (cent_s1["prg_scr"][:] * dx_cost * discount[:, 1]) + (cent_s1["prg_hrs"][:] * hrs_cost * discount[:, 2])

    # Difference (to be added to direct costs)
    tx_bl_lbd_mvd = (tx_bl_lb + mv_bl_hbig_lb) - (tx_bl_pe + mv_bl_pe)
    tx_bl_ubd_mvd = (tx_bl_ub + mv_bl_hbig_ub) - (tx_bl_pe + mv_bl_pe)

    tx_s1_lbd_mvd = (tx_s1_lb + mv_s1_hbig_lb) - (tx_s1_pe + mv_s1_pe)
    tx_s1_ubd_mvd = (tx_s1_ub + mv_s1_hbig_ub) - (tx_s1_pe + mv_s1_pe)

    # total direct costs for tabulation, NEB, and ICER
    tcost_lb_bl = econ["cbl_dirc"] + tx_bl_lbd_mvd[:, 0]
    tcost_ub_bl = econ["cbl_dirc"] + tx_bl_ubd_mvd[:, 0]
    tcost_lb_s1 = econ["cs1_dirc"] + tx_s1_lbd_mvd[:, 0]
    tcost_ub_s1 = econ["cs1_dirc"] + tx_s1_ubd_mvd[:, 0]

    """HCC surveillance costs"""
    # current
    hcc_cost = care_costs.loc[6, 'China']
    hcc_prp = care_costs.loc[8, 'China']
    util = 0.25

    hccs_bl_pe, hccs_s1_pe, hccs_bl_lb, hccs_s1_lb, hccs_bl_ub, hccs_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    hccs_bl_pe[:, 0] = ((bl_cent["pop_hcc"][:] * hcc_cost * util) + (bl_cent["tgt_hcc"][:] * hcc_cost * util) + (bl_cent["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]
    hccs_s1_pe[:, 0] = ((cent_s1["pop_hcc"][:] * hcc_cost * util) + (cent_s1["tgt_hcc"][:] * hcc_cost * util) + (cent_s1["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]

    # half
    hccs_bl_lb = hccs_bl_pe * 0.5
    hccs_s1_lb = hccs_s1_pe * 0.5

    # double
    hccs_bl_ub = hccs_bl_pe * 2
    hccs_s1_ub = hccs_s1_pe * 2

    # Difference (to be added to indirect costs)
    bl_hccs_lbd = hccs_bl_lb - hccs_bl_pe
    bl_hccs_ubd = hccs_bl_ub - hccs_bl_pe

    s1_hccs_lbd = hccs_s1_lb - hccs_s1_pe
    s1_hccs_ubd = hccs_s1_ub - hccs_s1_pe

    # total direct costs for tabulation, NEB, and ICER
    hccs_lb_bl = econ["cbl_dirc"] + bl_hccs_lbd[:, 0]
    hccs_ub_bl = econ["cbl_dirc"] + bl_hccs_ubd[:, 0]

    hccs_lb_s1 = econ["cs1_dirc"] + s1_hccs_lbd[:, 0]
    hccs_ub_s1 = econ["cs1_dirc"] + s1_hccs_ubd[:, 0]

    """Hospitalization costs"""
    # current
    hosp_cost = care_costs.loc[7, 'China']
    tx_hosp = 0.5

    hospc_bl_pe, hospc_s1_pe, hospc_bl_lb, hospc_s1_lb, hospc_bl_ub, hospc_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    hospc_bl_pe[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * util) + (bl_cent["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    hospc_s1_pe[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * util) + (cent_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]

    # half
    hospc_bl_lb = hospc_bl_pe * 0.5
    hospc_s1_lb = hospc_s1_pe * 0.5

    # double
    hospc_bl_ub = hospc_bl_pe * 2
    hospc_s1_ub = hospc_s1_pe * 2

    # Difference (to be added to indirect costs)
    bl_hospc_lbd = hospc_bl_lb - hospc_bl_pe
    bl_hospc_ubd = hospc_bl_ub - hospc_bl_pe

    s1_hospc_lbd = hospc_s1_lb - hospc_s1_pe
    s1_hospc_ubd = hospc_s1_ub - hospc_s1_pe

    # total direct costs for tabulation, NEB, and ICER
    hospc_lb_bl = econ["cbl_dirc"] + bl_hospc_lbd[:, 0]
    hospc_ub_bl = econ["cbl_dirc"] + bl_hospc_ubd[:, 0]

    hospc_lb_s1 = econ["cs1_dirc"] + s1_hospc_lbd[:, 0]
    hospc_ub_s1 = econ["cs1_dirc"] + s1_hospc_ubd[:, 0]

    """Hospitalization rate after treatment"""
    # current
    hosp_cost = care_costs.loc[7, 'China']
    tx_hosp = 0.5
    tx_hosp_lb = 1
    tx_hosp_ub = 0.25

    hospu_bl_pe, hospu_s1_pe, hospu_bl_lb, hospu_s1_lb, hospu_bl_ub, hospu_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    hospu_bl_pe[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * util) + (bl_cent["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    hospu_s1_pe[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * util) + (cent_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]

    # half
    hospu_bl_lb[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * util) + (bl_cent["hsp_tx"][:] * hosp_cost * util * tx_hosp_lb)) * discount[:, 1]
    hospu_s1_lb[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * util) + (cent_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp_lb)) * discount[:, 1]

    # double
    hospu_bl_ub[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * util) + (bl_cent["hsp_tx"][:] * hosp_cost * util * tx_hosp_ub)) * discount[:, 1]
    hospu_s1_ub[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * util) + (cent_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp_ub)) * discount[:, 1]

    # Difference (to be added to indirect costs)
    bl_hospu_lbd = hospu_bl_lb - hospu_bl_pe
    bl_hospu_ubd = hospu_bl_ub - hospu_bl_pe

    s1_hospu_lbd = hospu_s1_lb - hospu_s1_pe
    s1_hospu_ubd = hospu_s1_ub - hospu_s1_pe

    # total direct costs for tabulation, NEB, and ICER
    hospu_lb_bl = econ["cbl_dirc"] + bl_hospu_lbd[:, 0]
    hospu_ub_bl = econ["cbl_dirc"] + bl_hospu_ubd[:, 0]

    hospu_lb_s1 = econ["cs1_dirc"] + s1_hospu_lbd[:, 0]
    hospu_ub_s1 = econ["cs1_dirc"] + s1_hospu_ubd[:, 0]

    '''Disease management rate'''
    dx_dmc = care_costs.loc[4, 'China']
    tx_dmc = care_costs.loc[5, 'China']
    util = 0.25
    tx_hosp = 0.5
    utillb = 0
    utilub = 0.5

    manc_bl_pe, manc_s1_pe, manc_bl_lb, manc_s1_lb, manc_bl_ub, manc_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)),np.zeros((61, 1)), np.zeros((61, 1)),np.zeros((61, 1)), np.zeros((61, 1))
    hospc_bl_pe, hospc_s1_pe, hospc_bl_lb, hospc_s1_lb, hospc_bl_ub, hospc_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))
    hccs_bl_pe, hccs_s1_pe, hccs_bl_lb, hccs_s1_lb, hccs_bl_ub, hccs_s1_ub = np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1)), np.zeros((61, 1))

    ## Disease management costs (diagnosis + treatment)
    manc_bl_pe[:, 0] = ((bl_cent["dm_dx"][:] * dx_dmc * util) + (bl_cent["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]
    manc_s1_pe[:, 0] = ((cent_s1["dm_dx"][:] * dx_dmc * util) + (cent_s1["dm_tx"][:] * tx_dmc * util)) * discount[:, 1]

    ## Hospitalization costs
    hospc_bl_pe[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * util) + (bl_cent["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]
    hospc_s1_pe[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * util) + (cent_s1["hsp_tx"][:] * hosp_cost * util * tx_hosp)) * discount[:, 1]

    ## Cost of HCC surveillance
    hccs_bl_pe[:, 0] = ((bl_cent["pop_hcc"][:] * hcc_cost * util) + (bl_cent["tgt_hcc"][:] * hcc_cost * util) + (bl_cent["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]
    hccs_s1_pe[:, 0] = ((cent_s1["pop_hcc"][:] * hcc_cost * util) + (cent_s1["tgt_hcc"][:] * hcc_cost * util) + (cent_s1["tgt_hcc_b"][:] * hcc_prp * hcc_cost * util)) * discount[:, 1]

    """Management rate is 0"""
    ## Disease management costs (diagnosis + treatment)
    manc_bl_lb[:, 0] = ((bl_cent["dm_dx"][:] * dx_dmc * utillb) + (bl_cent["dm_tx"][:] * tx_dmc * utillb)) * discount[:, 1]
    manc_s1_lb[:, 0] = ((cent_s1["dm_dx"][:] * dx_dmc * utillb) + (cent_s1["dm_tx"][:] * tx_dmc * utillb)) * discount[:, 1]

    ## Hospitalization costs
    hospc_bl_lb[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * utillb) + (bl_cent["hsp_tx"][:] * hosp_cost * utillb * tx_hosp)) * discount[:, 1]
    hospc_s1_lb[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * utillb) + (cent_s1["hsp_tx"][:] * hosp_cost * utillb * tx_hosp)) * discount[:, 1]

    ## Cost of HCC surveillance
    hccs_bl_lb[:, 0] = ((bl_cent["pop_hcc"][:] * hcc_cost * utillb) + (bl_cent["tgt_hcc"][:] * hcc_cost * utillb) + (bl_cent["tgt_hcc_b"][:] * hcc_prp * hcc_cost * utillb)) * discount[:, 1]
    hccs_s1_lb[:, 0] = ((cent_s1["pop_hcc"][:] * hcc_cost * utillb) + (cent_s1["tgt_hcc"][:] * hcc_cost * utillb) + (cent_s1["tgt_hcc_b"][:] * hcc_prp * hcc_cost * utillb)) * discount[:, 1]

    """Management rate is 0.5"""
    ##  Disease management costs (diagnosis + treatment)
    manc_bl_ub[:, 0] = ((bl_cent["dm_dx"][:] * dx_dmc * utilub) + (bl_cent["dm_tx"][:] * tx_dmc * utilub)) * discount[:,1]
    manc_s1_ub[:, 0] = ((cent_s1["dm_dx"][:] * dx_dmc * utilub) + (cent_s1["dm_tx"][:] * tx_dmc * utilub)) * discount[:,1]

    ## Hospitalization costs
    hospc_bl_ub[:, 0] = ((bl_cent["hsp_utx"][:] * hosp_cost * utilub) + (bl_cent["hsp_tx"][:] * hosp_cost * utilub * tx_hosp)) * discount[:, 1]
    hospc_s1_ub[:, 0] = ((cent_s1["hsp_utx"][:] * hosp_cost * utilub) + (cent_s1["hsp_tx"][:] * hosp_cost * utilub * tx_hosp)) * discount[:, 1]

    ## Cost of HCC surveillance
    hccs_bl_ub[:, 0] = ((bl_cent["pop_hcc"][:] * hcc_cost * utilub) + (bl_cent["tgt_hcc"][:] * hcc_cost * utilub) + (bl_cent["tgt_hcc_b"][:] * hcc_prp * hcc_cost * utilub)) * discount[:, 1]
    hccs_s1_ub[:, 0] = ((cent_s1["pop_hcc"][:] * hcc_cost * utilub) + (cent_s1["tgt_hcc"][:] * hcc_cost * utilub) + (cent_s1["tgt_hcc_b"][:] * hcc_prp * hcc_cost * utilub)) * discount[:, 1]

    # Difference (to be added to indirect (hcc surveillance, dis man, hosp) costs)
    bl_util_lbd = (hospc_bl_lb + hccs_bl_lb + manc_bl_lb) - (hospc_bl_pe + hccs_bl_pe + manc_bl_pe)
    bl_util_ubd = (hospc_bl_ub + hccs_bl_ub + manc_bl_ub) - (hospc_bl_pe + hccs_bl_pe + manc_bl_pe)

    s1_util_lbd = (hospc_s1_lb + hccs_s1_lb + manc_s1_lb) - (hospc_s1_pe + hccs_s1_pe + manc_bl_pe)
    s1_util_ubd = (hospc_s1_ub + hccs_s1_ub + manc_s1_ub) - (hospc_s1_pe + hccs_s1_pe + manc_bl_pe)

    # total direct costs for tabulation, NEB, and ICER
    util_lb_bl = econ["cbl_dirc"] + bl_util_lbd[:, 0]
    util_ub_bl = econ["cbl_dirc"] + bl_util_ubd[:, 0]

    util_lb_s1 = econ["cs1_dirc"] + s1_util_lbd[:, 0]
    util_ub_s1 = econ["cs1_dirc"] + s1_util_ubd[:, 0]

    res_dict = {}

    res_dict["dirc_bl"] = econ["cbl_dirc"]
    res_dict["dirc_s1"] = econ["cs1_dirc"]
    res_dict["dx_lb_bl"] = dcost_lb_bl
    res_dict["dx_ub_bl"] = dcost_ub_bl
    res_dict["dx_lb_s1"] = dcost_lb_s1
    res_dict["dx_ub_s1"] = dcost_ub_s1
    res_dict["vax_lb_bl"] = vcost_lb_bl
    res_dict["vax_ub_bl"] = vcost_ub_bl
    res_dict["vax_lb_s1"] = vcost_lb_s1
    res_dict["vax_ub_s1"] = vcost_ub_s1
    res_dict["trt_lb_bl"] = tcost_lb_bl
    res_dict["trt_ub_bl"] = tcost_ub_bl
    res_dict["trt_lb_s1"] = tcost_lb_s1
    res_dict["trt_ub_s1"] = tcost_ub_s1
    res_dict["surv_lb_bl"] = hccs_lb_bl
    res_dict["surv_ub_bl"] = hccs_ub_bl
    res_dict["surv_lb_s1"] = hccs_lb_s1
    res_dict["surv_ub_s1"] = hccs_ub_s1
    res_dict["hospc_lb_bl"] = hospc_lb_bl
    res_dict["hospc_ub_bl"] = hospc_ub_bl
    res_dict["hospc_lb_s1"] = hospc_lb_s1
    res_dict["hospc_ub_s1"] = hospc_ub_s1
    res_dict["hospu_lb_bl"] = hospu_lb_bl
    res_dict["hospu_ub_bl"] = hospu_ub_bl
    res_dict["hospu_lb_s1"] = hospu_lb_s1
    res_dict["hospu_ub_s1"] = hospu_ub_s1
    res_dict["util_lb_bl"] = util_lb_bl
    res_dict["util_ub_bl"] = util_ub_bl
    res_dict["util_lb_s1"] = util_lb_s1
    res_dict["util_ub_s1"] = util_ub_s1
    return res_dict


def sens_nebicer(econ_sens, econ_main):
    import numpy as np
    econ_outs = {}

    keys_to_check = [
        "vax_lb_bl", "vax_ub_bl", "vax_lb_s1", "vax_ub_s1",
        "dx_lb_bl", "dx_ub_bl", "dx_lb_s1", "dx_ub_s1",
        "trt_lb_bl", "trt_ub_bl", "trt_lb_s1", "trt_ub_s1",
        "surv_lb_bl", "surv_ub_bl", "surv_lb_s1", "surv_ub_s1",
        "hospc_lb_bl", "hospc_ub_bl", "hospc_lb_s1", "hospc_ub_s1",
        "hospu_lb_bl", "hospu_ub_bl", "hospu_lb_s1", "hospu_ub_s1",
        "util_lb_bl", "util_ub_bl", "util_lb_s1", "util_ub_s1"
    ]

    for key in keys_to_check:
        if key in econ_sens:
            econ_sens[key] = np.nan_to_num(econ_sens[key], nan=0)

## vax costs
    econ_outs["vax_lb_neb"] = (np.cumsum(econ_sens["vax_lb_bl"]) + np.cumsum(econ_main["cbl_prod"])) - (np.cumsum(econ_sens["vax_lb_s1"]) + np.cumsum(econ_main["cs1_prod"]))
    econ_outs["vax_ub_neb"] = (np.cumsum(econ_sens["vax_ub_bl"]) + np.cumsum(econ_main["cbl_prod"])) - (np.cumsum(econ_sens["vax_ub_s1"]) + np.cumsum(econ_main["cs1_prod"]))

    econ_outs["vax_lb_icer"]= -(np.cumsum(econ_sens["vax_lb_bl"])-np.cumsum(econ_sens["vax_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["vax_ub_icer"]= -(np.cumsum(econ_sens["vax_ub_bl"])-np.cumsum(econ_sens["vax_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

## Diagnosis Costs
    econ_outs["dx_lb_neb"]= (np.cumsum(econ_sens["dx_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["dx_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["dx_ub_neb"]= (np.cumsum(econ_sens["dx_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["dx_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))

    econ_outs["dx_lb_icer"]= -(np.cumsum(econ_sens["dx_lb_bl"])-np.cumsum(econ_sens["dx_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["dx_ub_icer"]= -(np.cumsum(econ_sens["dx_ub_bl"])-np.cumsum(econ_sens["dx_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

## Treatment Costs
    econ_outs["trt_lb_neb"]= (np.cumsum(econ_sens["trt_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["trt_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["trt_ub_neb"]= (np.cumsum(econ_sens["trt_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["trt_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))

    econ_outs["trt_lb_icer"]= -(np.cumsum(econ_sens["trt_lb_bl"])-np.cumsum(econ_sens["trt_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["trt_ub_icer"]= -(np.cumsum(econ_sens["trt_ub_bl"])-np.cumsum(econ_sens["trt_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

## SUV Costs
    econ_outs["surv_lb_neb"]= (np.cumsum(econ_sens["surv_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["surv_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["surv_ub_neb"]= (np.cumsum(econ_sens["surv_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["surv_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))

    econ_outs["surv_lb_icer"]= -(np.cumsum(econ_sens["surv_lb_bl"])-np.cumsum(econ_sens["surv_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["surv_ub_icer"]= -(np.cumsum(econ_sens["surv_ub_bl"])-np.cumsum(econ_sens["surv_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

## hos Costs
    econ_outs["hospc_lb_neb"]= (np.cumsum(econ_sens["hospc_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["hospc_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["hospc_ub_neb"]= (np.cumsum(econ_sens["hospc_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["hospc_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))

    econ_outs["hospc_lb_icer"]= -(np.cumsum(econ_sens["hospc_lb_bl"])-np.cumsum(econ_sens["hospc_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["hospc_ub_icer"]= -(np.cumsum(econ_sens["hospc_ub_bl"])-np.cumsum(econ_sens["hospc_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

## hos util
    econ_outs["hospu_lb_neb"] = (np.cumsum(econ_sens["hospu_lb_bl"]) + np.cumsum(econ_main["cbl_prod"])) - (np.cumsum(econ_sens["hospu_lb_s1"]) + np.cumsum(econ_main["cs1_prod"]))
    econ_outs["hospu_ub_neb"] = (np.cumsum(econ_sens["hospu_ub_bl"]) + np.cumsum(econ_main["cbl_prod"])) - (np.cumsum(econ_sens["hospu_ub_s1"]) + np.cumsum(econ_main["cs1_prod"]))

    econ_outs["hospu_lb_icer"] = -(np.cumsum(econ_sens["hospu_lb_bl"]) - np.cumsum(econ_sens["hospu_lb_s1"])) / (np.cumsum(econ_main["cbl_daly"]) - np.cumsum(econ_main["cs1_daly"]))
    econ_outs["hospu_ub_icer"] = -(np.cumsum(econ_sens["hospu_ub_bl"]) - np.cumsum(econ_sens["hospu_ub_s1"])) / (np.cumsum(econ_main["cbl_daly"]) - np.cumsum(econ_main["cs1_daly"]))

## Management rate Costs
    econ_outs["util_lb_neb"]= (np.cumsum(econ_sens["util_lb_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["util_lb_s1"])+np.cumsum(econ_main["cs1_prod"]))
    econ_outs["util_ub_neb"]= (np.cumsum(econ_sens["util_ub_bl"])+np.cumsum(econ_main["cbl_prod"]))-(np.cumsum(econ_sens["util_ub_s1"])+np.cumsum(econ_main["cs1_prod"]))

    econ_outs["util_lb_icer"]= -(np.cumsum(econ_sens["util_lb_bl"])-np.cumsum(econ_sens["util_lb_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))
    econ_outs["util_ub_icer"]= -(np.cumsum(econ_sens["util_ub_bl"])-np.cumsum(econ_sens["util_ub_s1"]))/(np.cumsum(econ_main["cbl_daly"])-np.cumsum(econ_main["cs1_daly"]))

    return econ_outs

###Data integration into Excel
def process_excel_files(excel_files, selected_sheets, discount_rate=0, base_year=2023):
    DataFrames = {}


    for excel_file in excel_files:
        print(f"Processing file: {excel_file}")

        sheet_names = pd.ExcelFile(excel_file).sheet_names

        for sheet in sheet_names:
            if sheet not in selected_sheets:
                continue

            df = pd.read_excel(excel_file, sheet_name=sheet)

            if 'Year' not in df.columns:
                num_rows = len(df)
                years = list(range(1991, 1991 + num_rows - 1))
                df.insert(0, 'Year', [None] + years)

            file_key = excel_file.split('/')[-1].replace(".xlsx", "")
            key = f"{file_key}_{sheet}"
            DataFrames[key] = df

    sum_results = []

    for sheet_key in DataFrames:
        sheet_name = '_'.join(sheet_key.split('_')[1:])
        df = DataFrames[sheet_key]

        if 'Year' not in df.columns:
            print(f"表 {sheet_name} 缺少年份页")
            continue

        df_filtered = df[df['Year'] >= 2023]
        if df_filtered.empty:
            print(f"No data for {sheet_name} after filtering for Year >= 2023")
            continue

        non_year_columns = [col for col in df_filtered.columns if col != 'Year']
        if all(substring not in sheet_name for substring in ['chc_inc', 'hcc_inc', 'mort']):
            for year in df_filtered['Year'].unique():
                year_mask = df_filtered['Year'] == year
                discount_factor = (1 + discount_rate) ** (year - base_year)
                df_filtered.loc[year_mask, non_year_columns] /= discount_factor

        column_sums = df_filtered.drop('Year', axis=1).sum()
        sum_results.append((sheet_key, column_sums))

    sum_df = pd.DataFrame([result[1] for result in sum_results], index=[result[0] for result in sum_results])

    result_list = []

    for sheet in selected_sheets:
        for file in excel_files:
            file_key = file.split('/')[-1].replace(".xlsx", "")
            key = f"{file_key}_{sheet}"
            print(f"Checking key: {key}")
            if key in sum_df.index:
                data = sum_df.loc[key].dropna()

                # 计算 95% 百分位数的上下限
                sheet_lower_quantile = np.percentile(data, 2.5)
                sheet_upper_quantile = np.percentile(data, 97.5)
                sheet_median_quantile = np.mean(data)

                print(f"Sheet: {key}")
                result_list.append({
                    'Sheet Name': key,
                    'Median': sheet_median_quantile,
                    '95% PI': (sheet_lower_quantile, sheet_upper_quantile)
                })
            else:
                print(f"Key {key} not found in sum_df index")

    return result_list

def calculate_mean_and_ui(data):
    result = {}
    for key in data[0].keys():
        values = [entry[key] for entry in data]
        mean_value = np.mean(values)
        lower_ui, upper_ui = np.percentile(values, [2.5, 97.5])
        result[key] = {
            'mean': mean_value,
            '95% UI': (lower_ui, upper_ui)
        }
    return result


def calculate_percentiles(results, percentiles=[2.5, 50, 97.5]):
    grouped_data = {}
    for record in results:
        owsatype = record['owsa_type']
        total_discounted = record['Yt_total_discounted']
        if owsatype not in grouped_data:
            grouped_data[owsatype] = {'total_discounted_bl_bl_elimination': [],
                                       'total_discounted_bl_s1_no_elimination': []}
        grouped_data[owsatype]['total_discounted_bl_bl_elimination'].append(
            total_discounted['total_discounted_bl_bl_elimination'])
        grouped_data[owsatype]['total_discounted_bl_s1_no_elimination'].append(
            total_discounted['total_discounted_bl_s1_no_elimination'])
    result_percentiles = {}
    for owsatype, data in grouped_data.items():
        result_percentiles[owsatype] = {}
        for key, values in data.items():
            percentiles_result = np.percentile(values, percentiles)
            mean_value = np.mean(values)
            result_percentiles[owsatype][key] = f"{mean_value:.2f} ({percentiles_result[0]:.2f}, {percentiles_result[2]:.2f})"

    return result_percentiles

def plot_yearly_difference(df_bl_bl_elimination,
                           df_bl_s1_no_elimination,
                           df_bl_s2_no_elimination,
                           df_bl_s3_no_elimination,
                           start_year, num_years, file_name):
    years = list(range(start_year, start_year + num_years))

    plt.figure(figsize=(10, 6))

    plt.plot(years, df_bl_bl_elimination['Mean_Difference'] / 10, label='Eliminate scenario', color='#17202A')
    plt.plot(years, -df_bl_s1_no_elimination['Mean_Difference'] / 10, label='WHO coverage target achieved in 2030', color='#8ECFC9')
    plt.plot(years, -df_bl_s2_no_elimination['Mean_Difference'] / 10, label='WHO coverage target achieved in 2040', color='#82B0D2')
    plt.plot(years, -df_bl_s3_no_elimination['Mean_Difference'] / 10, label='WHO coverage target achieved in 2050', color='#BEB8DC')

    plt.xlabel('Year')
    plt.ylabel('Difference in level of GDP (Billions US$)')
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name, format='png')

def compute_average_by_scenario(raw_data):

    all_scenarios = set()
    for scenario_dict in raw_data:
        all_scenarios.update(scenario_dict.keys())
    all_scenarios = list(all_scenarios)


    result_dict = {}

    for scenario in all_scenarios:
        df_list = []

        for scenario_dict in raw_data:
            if scenario in scenario_dict:
                df_list.append(scenario_dict[scenario])
        if not df_list:
            continue

        combined_df = pd.concat(df_list, ignore_index=True)
        mean_df = combined_df.groupby("Year", as_index=False)["Discounted Difference"].mean()
        mean_df.rename(columns={"Discounted Difference": "Mean_Difference"}, inplace=True)
        result_dict[scenario] = mean_df

    return result_dict

def parse_value_ui(value_ui_str):

    match = re.match(r'^([-+]?\d*\.\d+|\d+) \(([-+]?\d*\.\d+|\d+), ([-+]?\d*\.\d+|\d+)\)$', value_ui_str)
    if match:
        mean = float(match.group(1))
        lower = float(match.group(2))
        upper = float(match.group(3))
        return mean, lower, upper
    else:
        return float('nan'), float('nan'), float('nan')

#==================================================================Start calling ===================================================================
# Set working directory and other paths
# Create path dictionary
paths = { }

macro_data_path = 'wd/ + macro_data.xlsx'
wd =
excel_output_path = "wd/ + china_econ_analysis.xlsx"
excel_economic_path = "wd/ + china_econ_analysis.xlsx"
excel_file_path = "wd/ + model_results.xlsx"
os.chdir(wd)
F = at.ProjectFramework("hbv_China_model.xlsx")
runs = 1000

# Setting parameters
p_j = 1
ages = {'0_4': 2, '5_14': 10, '15_49': 32, '50_59': 55, '60': 67}
start_year = 1990
discount_rate = 0
disc_rate = 0.03
disc_rate_upper = 0.05
disc_rate_lower = 0

china_bl_runs, china_bl_cent = model_results(F, "China_db_mav.xlsx", "China_calib.xlsx", "Status Quo", runs)
china_s1_runs, china_s1_cent = model_results(F, "China_db_s1.xlsx", "China_calib.xlsx", "S1: 2030 Target", runs)
china_s2_runs, china_s2_cent = model_results(F, "China_db_s2.xlsx", "China_calib.xlsx", "S2: 2040 Target", runs)
china_s3_runs, china_s3_cent = model_results(F, "China_db_s3.xlsx", "China_calib.xlsx", "S3: 2050 Target", runs)

china_econ = econ_analysis(china_bl_cent, china_s1_cent, china_s2_cent, china_s3_cent,china_bl_runs,
                           china_s1_runs, china_s2_runs, china_s3_runs,
    "costs.xlsx", runs, disc_rate, 0.03
)


scenarios = ['bl', 's1', 's2', 's3']
age_sex_groups = ["0-4M", "5-14M", "15-49M", "50-59M", "60+M",
                  "0-4F", "5-14F", "15-49F", "50-59F", "60+F"]

# Main loop, traversing simulation runs
Yt_results = []
Yt_yearly_results = []
num_runs = runs
for run_index in range(num_runs):

    pop_data_dict = {}
    yld_data_dict = {}
    yll_data_dict = {}
    mort_data_dict = {}

    for scenario, runs_data in zip(scenarios, [china_bl_runs, china_s1_runs, china_s2_runs, china_s3_runs]):

        pop_data_run = extract_data_for_run(runs_data, run_index, 'pop')
        yld_data_run = extract_data_for_run(runs_data, run_index, 'yld')
        yll_data_run = extract_data_for_run(runs_data, run_index, 'yll')
        mort_data_run = extract_data_for_run(runs_data, run_index, 'mort')


        pop_data_dict[scenario] = pop_data_run
        yld_data_dict[scenario] = yld_data_run
        yll_data_dict[scenario] = yll_data_run
        mort_data_dict[scenario] = mort_data_run


    econ_data_run = {}
    for scenario in scenarios:
        dirc_data = china_econ[f"{scenario}_dirc"]
        econ_data_run[scenario] = dirc_data[:, run_index]
    cost_adjustment,_ = generate_economic_comparison(econ_data_run)

    result = integrated_scenario_analysis(
        macro_data_path, ages, start_year, cost_adjustment, discount_rate,
        pop_data_dict, yld_data_dict, yll_data_dict, mort_data_dict
    )

    Yt_results.append(result['Yt_total_discounted'])
    Yt_yearly_results.append(result['yearly_discounted_differences'])

Yearly_result_df = compute_average_by_scenario(Yt_yearly_results)

Determine_results = calculate_mean_and_ui(Yt_results)
for scenario, stats in Determine_results.items():
    print(f"{scenario}: Mean = {stats['mean']}, 95% UI = {stats['95% UI']}")


#=========================================================OWSA===============================================================================
#===============================================================================================================================================
# # Part I - Sensitivity analysis of effects
ci_95_cent_bl_mtctw, ci_95_cent_s1_mtctw, runs_bl_mtctw, runs_s1_mtctw = par_sens(F, "China_db_mav.xlsx", "China_db_S1.xlsx", "China_calib.xlsx",mtct_wc, runs)
ci_95_cent_bl_mtctb, ci_95_cent_s1_mtctb, runs_bl_mtctb, runs_s1_mtctb = par_sens(F, "China_db_mav.xlsx", "China_db_S1.xlsx", "China_calib.xlsx",mtct_bc, runs)
ci_95_cent_bl_mortw, ci_95_cent_s1_mortw, runs_bl_mortw, runs_s1_mortw = par_sens(F, "China_db_mav.xlsx", "China_db_S1.xlsx", "China_calib.xlsx",mort_wc, runs)
ci_95_cent_bl_mortb, ci_95_cent_s1_mortb, runs_bl_mortb, runs_s1_mortb = par_sens(F, "China_db_mav.xlsx", "China_db_S1.xlsx", "China_calib.xlsx",mort_bc, runs)
ci_95_cent_bl_texw,  ci_95_cent_s1_texw,  runs_bl_texw,  runs_s1_texw  = par_sens(F, "China_db_mav.xlsx", "China_db_S1.xlsx", "China_calib.xlsx", tex_wc, runs)
ci_95_cent_bl_texb,  ci_95_cent_s1_texb,  runs_bl_texb,  runs_s1_texb  = par_sens(F, "China_db_mav.xlsx", "China_db_S1.xlsx", "China_calib.xlsx",tex_bc, runs)
# # Part II -Sensitivity Analysis- Economic
china_econ_sens_mtctw = parsens_econ(ci_95_cent_bl_mtctw, ci_95_cent_s1_mtctw, "costs.xlsx", disc_rate)
china_econ_sens_mtctb = parsens_econ(ci_95_cent_bl_mtctb, ci_95_cent_s1_mtctb, "costs.xlsx", disc_rate)
china_econ_sens_mortw = parsens_econ(ci_95_cent_bl_mortw, ci_95_cent_s1_mortw, "costs.xlsx", disc_rate)
china_econ_sens_mortb = parsens_econ(ci_95_cent_bl_mortb, ci_95_cent_s1_mortb, "costs.xlsx", disc_rate)
china_econ_sens_texw  = parsens_econ(ci_95_cent_bl_texw, ci_95_cent_s1_texw, "costs.xlsx", disc_rate)
china_econ_sens_texb  = parsens_econ(ci_95_cent_bl_texb, ci_95_cent_s1_texb, "costs.xlsx", disc_rate)
# # Part 3 - Results that change only economic data
china_econ_sens = econsens_econ(china_bl_cent, china_s1_cent, china_econ, "costs.xlsx", disc_rate)
china_econ_sens_icerneb = sens_nebicer(china_econ_sens, china_econ)
china_econ_discountupper = econsens_econ(china_bl_cent, china_s1_cent, china_econ, "costs.xlsx", disc_rate_upper)
china_econ_discountlower = econsens_econ(china_bl_cent, china_s1_cent, china_econ, "costs.xlsx", disc_rate_lower)

# OWSA
owsa_types = [
    'mtctw', 'mtctb', 'mortw', 'mortb', 'texw', 'texb',
    'dx_lb', 'dx_ub', 'vax_lb', 'vax_ub',
    'trt_lb', 'trt_ub', 'surv_lb', 'surv_ub',
    'hospc_lb', 'hospc_ub', 'hospu_lb', 'hospu_ub',
    'util_lb', 'util_ub', 'discountupper', 'discountlower'
]

Yt_results_owsa = []
OWSA_results = {owsa_type: [] for owsa_type in owsa_types}
owsa_econ_mapping = {
    'mtctw': china_econ_sens_mtctw,
    'mtctb': china_econ_sens_mtctb,
    'mortw': china_econ_sens_mortw,
    'mortb': china_econ_sens_mortb,
    'texw': china_econ_sens_texw,
    'texb': china_econ_sens_texb,
    'dx_lb': china_econ_sens,
    'dx_ub': china_econ_sens,
    'vax_lb': china_econ_sens,
    'vax_ub': china_econ_sens,
    'trt_lb': china_econ_sens,
    'trt_ub': china_econ_sens,
    'surv_lb': china_econ_sens,
    'surv_ub': china_econ_sens,
    'hospc_lb': china_econ_sens,
    'hospc_ub': china_econ_sens,
    'hospu_lb': china_econ_sens,
    'hospu_ub': china_econ_sens,
    'util_lb': china_econ_sens,
    'util_ub': china_econ_sens,
    'discountupper': china_econ_discountupper,
    'discountlower': china_econ_discountlower
}

data_mapping = {
    'mtctw': ('runs_bl_mtctw', 'runs_s1_mtctw'),
    'mtctb': ('runs_bl_mtctb', 'runs_s1_mtctb'),
    'mortw': ('runs_bl_mortw', 'runs_s1_mortw'),
    'mortb': ('runs_bl_mortb', 'runs_s1_mortb'),
    'texw': ('runs_bl_texw', 'runs_s1_texw'),
    'texb': ('runs_bl_texb', 'runs_s1_texb'),
    'dx_lb': ('china_bl_runs', 'china_s1_runs'),
    'dx_ub': ('china_bl_runs', 'china_s1_runs'),
    'vax_lb': ('china_bl_runs', 'china_s1_runs'),
    'vax_ub': ('china_bl_runs', 'china_s1_runs'),
    'trt_lb': ('china_bl_runs', 'china_s1_runs'),
    'trt_ub': ('china_bl_runs', 'china_s1_runs'),
    'surv_lb': ('china_bl_runs', 'china_s1_runs'),
    'surv_ub': ('china_bl_runs', 'china_s1_runs'),
    'hospc_lb': ('china_bl_runs', 'china_s1_runs'),
    'hospc_ub': ('china_bl_runs', 'china_s1_runs'),
    'hospu_lb': ('china_bl_runs', 'china_s1_runs'),
    'hospu_ub': ('china_bl_runs', 'china_s1_runs'),
    'util_lb': ('china_bl_runs', 'china_s1_runs'),
    'util_ub': ('china_bl_runs', 'china_s1_runs'),
    'discountupper': ('china_bl_runs', 'china_s1_runs'),
    'discountlower': ('china_bl_runs', 'china_s1_runs'),
}

runs_bl_XX = {}
runs_s1_XX = {}

for owsa_type, (bl_data_key, s1_data_key) in data_mapping.items():
    if bl_data_key in locals():
        runs_bl_XX[owsa_type] = locals()[bl_data_key]
    else:
        runs_bl_XX[owsa_type] = {}
    if s1_data_key in locals():
        runs_s1_XX[owsa_type] = locals()[s1_data_key]
    else:
        runs_s1_XX[owsa_type] = {}

runs_bl_XX = {}
runs_s1_XX = {}

owsa_types = [
     'mtctw',
    'mtctb', 'mortw', 'mortb', 'texw', 'texb',
    'dx_lb', 'dx_ub', 'vax_lb', 'vax_ub',
    'trt_lb', 'trt_ub', 'surv_lb', 'surv_ub',
    'hospc_lb', 'hospc_ub', 'hospu_lb', 'hospu_ub',
    'util_lb', 'util_ub', 'discountupper', 'discountlower'
]

for owsa_type, (bl_data_key, s1_data_key) in data_mapping.items():
    if bl_data_key in locals():
        runs_bl_XX[owsa_type] = locals()[bl_data_key]
    else:
        runs_bl_XX[owsa_type] = {}
    if s1_data_key in locals():
        runs_s1_XX[owsa_type] = locals()[s1_data_key]
    else:
        runs_s1_XX[owsa_type] = {}


pop_data_dict = {}
yld_data_dict = {}
yll_data_dict = {}
mort_data_dict = {}


for owsa_type in owsa_types:
    pop_data_dict[owsa_type] = {}
    yld_data_dict[owsa_type] = {}
    yll_data_dict[owsa_type] = {}
    mort_data_dict[owsa_type] = {}


OWSA_results = {owsa_type: [] for owsa_type in owsa_types}
OWSA_results_records = []


runs_bl_XX = {}
runs_s1_XX = {}

for owsa_type, (bl_data_key, s1_data_key) in data_mapping.items():
    runs_bl_XX[owsa_type] = locals().get(bl_data_key, {})
    runs_s1_XX[owsa_type] = locals().get(s1_data_key, {})

pop_data_dict = {owsa_type: {} for owsa_type in owsa_types}
yld_data_dict = {owsa_type: {} for owsa_type in owsa_types}
yll_data_dict = {owsa_type: {} for owsa_type in owsa_types}
mort_data_dict = {owsa_type: {} for owsa_type in owsa_types}

for run_index in range(num_runs):
    for owsa_type in owsa_types:
        for scenario, runs_data in zip(scenarios, [runs_bl_XX, runs_s1_XX]):
            pop_data_run = extract_data_for_run(runs_data[owsa_type], run_index, 'pop')
            yld_data_run = extract_data_for_run(runs_data[owsa_type], run_index, 'yld')
            yll_data_run = extract_data_for_run(runs_data[owsa_type], run_index, 'yll')
            mort_data_run = extract_data_for_run(runs_data[owsa_type], run_index, 'mort')

            pop_data_dict[owsa_type][scenario] = pop_data_run
            yld_data_dict[owsa_type][scenario] = yld_data_run
            yll_data_dict[owsa_type][scenario] = yll_data_run
            mort_data_dict[owsa_type][scenario] = mort_data_run

    for owsa_type in owsa_types:
        if owsa_type in ['discountupper', 'discountlower']:
            owsa_type_econ = 'dirc'
        elif owsa_type in [
            'dx_lb', 'dx_ub', 'vax_lb', 'vax_ub',
            'trt_lb', 'trt_ub', 'surv_lb', 'surv_ub',
            'hospc_lb', 'hospc_ub', 'hospu_lb', 'hospu_ub',
            'util_lb', 'util_ub'
        ]:
            owsa_type_econ = owsa_type
        else:
            owsa_type_econ = 'cdirc'

        econ_sens = owsa_econ_mapping.get(owsa_type, china_econ_sens)
        try:
            #  OWSA_analysis
            owsa_result = OWSA_analysis(
                macro_data_path,
                ages,
                start_year,
                econ_sens,
                discount_rate,
                is_owsa=True,
                owsa_type=owsa_type,
                owsa_type_econ=owsa_type_econ,
                yld_data=yld_data_dict[owsa_type],
                yll_data=yll_data_dict[owsa_type],
                pop_data=pop_data_dict[owsa_type],
                mort_data=mort_data_dict[owsa_type]
            )

            if isinstance(owsa_result, dict):
                Yt_total_discounted = owsa_result.get('Yt_total_discounted', np.nan)
            else:
                print(f"OWSA_analysis for {owsa_type} did not return a dictionary. Returned type: {type(owsa_result)}")
                Yt_total_discounted = np.nan

            if not isinstance(OWSA_results[owsa_type], list):
                print(f"Error: OWSA_results[{owsa_type}] is not a list. Current type: {type(OWSA_results[owsa_type])}")
                OWSA_results[owsa_type] = []

            OWSA_results[owsa_type].append(Yt_total_discounted)
            Yt_results_owsa.append(Yt_total_discounted)

            OWSA_results_records.append({
                'owsa_type': owsa_type,
                'Yt_total_discounted': Yt_total_discounted
            })

        except Exception as e:
            print(f"Error during OWSA analysis for {owsa_type}: {e}")
            continue

OWSA_Yt_result = calculate_percentiles(OWSA_results_records)
