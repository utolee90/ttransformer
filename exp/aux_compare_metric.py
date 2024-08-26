# 메트릭 그래프 비교
import numpy as np
import matplotlib.pyplot as plt

# 비교 데이터 목록 : 리벨명 : 데이터명
comparison_list = {
    'weather_96': 'long_term_forecast_traffic_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el4_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0',
    'weather_192': 'long_term_forecast_traffic_96_192_iTransformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el4_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0'
}

chart_type='barplot'

# 데이터 가져오기

fig, ax = plt.subplots(layout='constrained')

metrics = ("MSE", "MAE")
metric_indices = {"MAE": 0, "MSE": 1, "SMAE": 5, "CORR": 7, "IRR_RATIO": 6}
x = np.arange(len(metrics))
width = 0.25 # size
multiplier = 0

comp_new = {}

for comp_label, comp_id in comparison_list.items():
    # 메트릭 불러오기
    metric_result = np.load(f'./results/{comp_id}/metrics.npy')
    comp_new[comp_label] = metric_result

# barplot 그리기
used_indices = list(map(lambda x: metric_indices[x], metrics))
metric_map = {comp_label: list(map(lambda x: comp_new[comp_label][x], used_indices)) for comp_label in comparison_list}

if chart_type == 'barplot':
    for metric, res in metric_map.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, res, width, label=metric)
        multiplier += 1


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('value')
ax.set_title('Metric between Preds and Trues')
ax.set_xticks(x + width, metrics)
ax.legend(loc='upper left', ncols=len(comp_new))
ax.set_ylim(-0.2, 0.8)

plt.show()
fig.savefig("aux_compare_metric.png")