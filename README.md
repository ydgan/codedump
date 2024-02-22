# codedump

get_data - 获取个股以及申万行业指数日频行情

get_factors - 因子计算以及预处理

factors_eva - 因子评估

############################

  此页旨在详细评估基于量价数据的因子模型在过去几年内的表现，分析因子有效性，稳健性及潜在风险。评估时间段为 2018 年至 2023 年，换仓周期为 T+3；目标股票池为沪深两市除去 ST，新股外的所有个股；个股行情数据来自开源 Python 库 BaoStock；个股申万行业数据来自开源 Python 库 AkShare 以及申万宏源研究所官网；因子库主要由WorldQuant，国君以及广发等机构所发布的量价因子研报和网络分享组成，约 450 个日频因子。
  在完成预处理后（包括市值中性化、行业中性化以及标准化），为了评估每个因子对股票未来收益的预测能力，计算了 IC 值并进行了 T 检验，确保其统计显著性。进一步地，将股票按照因子得分从高到低分为五组，并在每组内部对所有个股实施等权配置，以计算各组的投资收益。综合考量 IC 值、T 检验结果以及分组收益情况，筛选出 18 个表现优异的单因子，这些因子在预测股票下一期收益方面展现出了显著的优势和稳健性。下图为因子“20日换手率均值”历史表现。
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/liq_turn_20_ic.png)
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/liq_turn_20_ret.png)
  多因子合成旨在结合不同因子的独特优势，以提高整体策略的预测能力和稳健性。为消除因子间的相关性，避免信息冗余和多重共线性问题，采用对称正交化方法对因子进行处理。通过对因子矩阵进行正交分解，得到了相关性极低因子，为后续的合成提供了更加纯净的因子输入。在因子正交化的基础上，尝试了多种多因子合成方法（等权合成，历史 IC/IR 加权，机器学习合成等），以寻求最佳的合成方式。通过对比不同合成因子在评估时间段内的 IC值，收益等指标，评估了各策略的优劣。下表为不同合成方法在评估时间段内 IC/IR 表现以及分组能力。
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/combine.png)
  下图为半衰 IR 加权方法历史表现
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/half_ir_ic.png)
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/half_ir_ret.png)
  下图为 XGBoost 合成因子历史表现
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/xgb_ic.png)
  ![image](https://github.com/ydgan/codedump/blob/main/eva_img/xgb_ret.png)
