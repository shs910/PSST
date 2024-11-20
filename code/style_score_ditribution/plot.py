import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from copy import deepcopy

# sns.set_theme(font_scale=1.25)
def concate_diff_data(baselines_df,other_models_df_dict):
    """
    baselines_df: ['id', 'text_type', 'position', 'interactivity_score']
    other_models_df_dict df columns: ['id', 'text_type', 'position', 'interactivity_score', 'model_name']
    """
    all_df_to_concate = []
    model_name_mapping = {"Llama-2-7b-chat-ms": "Llama-2-7b-chat",
              "Llama-2-13b-chat-ms": "Llama-2-13b-chat",
              "Llama-2-70B-Chat-GPTQ": "Llama-2-70b-chat",
              "Llama-3-8B-Instruct": "Llama-3-8b-instruct",
              "Llama-3-70B-Instruct-GPTQ": "Llama-3-70b-instruct",
              "gpt-3.5": "GPT-3.5-turbo",
              }
    for k,v in other_models_df_dict.items():
        all_df_to_concate.append(v)
        baselines_df["model_name"] = model_name_mapping.get(k,k)
        all_df_to_concate.append(deepcopy(baselines_df))
    
    return pd.concat(all_df_to_concate,ignore_index=True)
    

def plot(all_data_df,save_path):
    fig = sns.FacetGrid(all_data_df, col="model_name",row="dimension",margin_titles=True)
    fig.map_dataframe(sns.lineplot, x="position", y=f"score", hue="text_type")
    fig.set_titles(col_template="{col_name}", row_template="{row_name}",size=15)
    fig.add_legend(fontsize=15)
    sns.move_legend(
        fig, "lower center",
        bbox_to_anchor=(.4, 1), 
        ncol=5, title=None, frameon=False,fontsize=15
    )
    fig.set(ylim=(0, 100))
    fig.set_ylabels("score", fontsize=15)
    fig.set_xlabels("position", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.cla()


def main(plot_type):
    tgt_dim_list = ["interactivity","orality"]
    for valid_only in [False,True]:
        data_type_list = ["800_delta_200"]
        base_path  = "./code/PSST/style_score_ditribution"
        data_dir_path = f"{base_path}/sampl_for_plot/{plot_type}"
        tgt_dir_path = f"{base_path}/plot_res/{plot_type}"
        baselines_data_path = f"{data_dir_path}/baselines"
        other_model_names = ["Llama-2-7b-chat-ms","Llama-3-8B-Instruct","Llama-2-13b-chat-ms","Llama-2-70B-Chat-GPTQ","Llama-3-70B-Instruct-GPTQ", "gpt-3.5"]
        for data_type in data_type_list:
            tmp_file_name = f"{data_type}_baselines_valid_only.csv" if valid_only  else f"{data_type}_baselines.csv"
            baseline_df = pd.read_csv(os.path.join(baselines_data_path,tmp_file_name))
            other_models_df_dict = {}
            for model_name in other_model_names:
                tmp_file_name = f"{data_type}_{model_name}_valid_only.csv" if valid_only else f"{data_type}_{model_name}.csv"
                other_models_df_dict[model_name] = pd.read_csv(os.path.join(data_dir_path,model_name,tmp_file_name))
            all_data_df = concate_diff_data(baseline_df,other_models_df_dict)
            
            all_data_df = all_data_df.loc[all_data_df["dimension"].isin(tgt_dim_list)]
            print(all_data_df.columns)

            tmp_tgt_dir = os.path.join(tgt_dir_path,f"{data_type}_valid_only_{valid_only}")
            os.makedirs(tmp_tgt_dir,exist_ok=True)
            all_data_df.to_csv(os.path.join(tmp_tgt_dir,"+".join(tgt_dim_list)+"_baseline_"+"+".join(other_model_names)+".csv"))
            save_path = os.path.join(tmp_tgt_dir,"+".join(tgt_dim_list)+"_baseline_"+"+".join(other_model_names)+".png")
            plot(all_data_df,save_path)

if __name__ == "__main__" :
    # "MODE_4_CHUNK_5","MODE_4_CHUNK_7","MODE_4_CHUNK_10"
    # for t in ["MODE_4_CHUNK_5","MODE_4_CHUNK_7","MODE_4_CHUNK_10"]:
    for t in ["MODE_4_CHUNK_5"]:
        main(t)        