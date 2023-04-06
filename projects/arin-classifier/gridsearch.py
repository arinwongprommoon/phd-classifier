#!/usr/bin/env python3

from sklearn.model_selection import GridSearchCV


# Grid search
# TODO: convert to parameters-process structure fully
# TODO: documentation
class GridSearchHandler:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def fit(self, pipeline, features, targets):
        grid_pipeline = GridSearchCV(pipeline, self.param_grid, cv=5)
        grid_pipeline.fit(features, targets)
        pipeline.set_params(**grid_pipeline.best_params_)
        return pipeline
