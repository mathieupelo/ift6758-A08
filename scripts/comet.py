import os
import pandas as pd
from comet_ml import Experiment



experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='Milestone_2',
    workspace='me-pic',
)

data = pd.read_csv('data/derivatives/dataframe_milestone_2.csv')

subset_df = data[data['gameId'] == '2017021065']

experiment.log_dataframe_profile(
    subset_df,
    name='wpg_v_wsh_2017021065',
    dataframe_format='csv'
)

experiment.end()
