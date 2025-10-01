###
### 分类模型
###
########################################################################################################################
###### RF
########################################################################################################################
from sklearn.ensemble import RandomForestClassifier


def RF(train_X,
       train_y,
       val_X,
       val_y,
       model_arg_dict=dict(),
       fit_arg_dict=dict()):
    model = RandomForestClassifier(random_state=random_state_num,
                                   **model_arg_dict)
    _ = model.fit(train_X, train_y, **fit_arg_dict)
    return model


########################################################################################################################
###### LGBM
########################################################################################################################
from lightgbm import LGBMClassifier


def LGBM(train_X,
         train_y,
         val_X,
         val_y,
         model_arg_dict=dict(),
         fit_arg_dict=dict()):
    model = LGBMClassifier(random_state=random_state_num, **model_arg_dict)
    _ = model.fit(train_X, train_y, eval_set=[(val_X, val_y)], **fit_arg_dict)
    return model