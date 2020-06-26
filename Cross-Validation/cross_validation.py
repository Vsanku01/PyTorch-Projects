from sklearn import model_selection
import pandas as pd

class Crossvalidation:
    def __init__(
        self,
        df,
        target_columns,
        problem_type='binary_classification',
        mulilabel_delimiter = ',',
        n_folds=5,
        shuffle=True,
        random_state=24
        ):
        self.dataframe = df
        self.target_columns = target_columns
        self.num_targets = len(target_columns)
        self.problem_type = problem_type
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state =random_state
        self.mulilabel_delimiter =mulilabel_delimiter


        if shuffle == True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe['kfold'] = -1
        print(self.num_targets)

    def split(self):
        if self.problem_type in ("binary_classification","multi_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for binary classification or multi classification problems")
            target = self.target_columns[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!..Check the problem type again")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(
                                            n_splits=self.n_folds,
                                            shuffle= False,
                )

                for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.dataframe,y=self.dataframe['target'].values)):
                    self.dataframe.loc[valid_idx,'kfold'] = fold

        elif self.problem_type in ("single_column_regression", "multi_column_regression"):
            if self.num_targets != 1 and self.problem_type == "single_column_regression":
                raise Exception("Invalid number of targets for this problem_type")
            if self.num_targets < 2 and self.problem_type == "multi_column_regression":
                raise Exception("Invalid number of targets for this problem_type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[valid_idx,'kfold'] = fold

        elif self.problem == 'multilabel_classification':
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem_type")
            targets = self.dataframe[self.target_columns[0]].apply(lambda x: len(str(x).split(self.mulilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)
            for fold,(train_idx,valid_idx) in enumerate(kf.split(X=self.dataframe,y=targets)):
                self.dataframe[valid_idx,'kfold'] = fold

        else:
            raise Exception("Problem not found")

                
                


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    cv = Crossvalidation(df,target_columns=['target'])
    df_split = cv.split()


