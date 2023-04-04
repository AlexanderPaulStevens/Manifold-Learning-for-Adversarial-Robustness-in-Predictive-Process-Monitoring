# The code and architecture in this file stems from: hhttps://github.com/irhete/predictive-monitoring-benchmark
# I have only added this file to my own GitHub, to make my own code reproducible.
# I thank the authors for their valuable work.
from encoders.AggregateTransformer import AggregateTransformer
        
def get_encoder(method, case_id_col=None, static_cat_cols=None, static_num_cols=None, dynamic_cat_cols=None,
                dynamic_num_cols=None, fillna=True, max_events=None, activity_col=None, resource_col=None, timestamp_col=None,
                scale_model=None):

    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols, fillna=fillna)

    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, boolean=False, fillna=fillna)


    else:
        print("Invalid encoder type")
        return None
