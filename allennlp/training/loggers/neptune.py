# -*- coding: utf-8 -*-
import time

import neptune


class Neptune:
    def __init__(
        self,
        project_name=None,
        experiment_name=None,
        params=None,
        upload_source_files=None,
        **kwargs
    ):
        neptune.init(project_qualified_name=project_name)

        self._experiment = neptune.create_experiment(
            name=experiment_name, params=params, upload_source_files=upload_source_files, **kwargs
        )

    def log_metrics(self, train_metrics, val_metrics=None, epoch=None):
        for metric, value in train_metrics.items():
            self._experiment.log_metric("train/" + metric, epoch, value)
            time.sleep(0.01)

        if val_metrics:
            for metric, value in val_metrics.items():
                self._experiment.log_metric("val/" + metric, epoch, value)
                time.sleep(0.01)

    def finalize(self):
        self._experiment.stop()
