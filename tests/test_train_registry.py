from __future__ import annotations


def test_train_module_exposes_registry_helper():
    from matching_service.application.train import _try_mlflow_log

    assert callable(_try_mlflow_log)
    sig = __import__("inspect").signature(_try_mlflow_log)
    assert "registered_name" in sig.parameters
    assert "stage" in sig.parameters


def test_registered_name_constant_in_code():
    import inspect

    import matching_service.application.train as t

    src = inspect.getsource(t)
    assert "matching-deal-flat-catboost" in src
    assert "mlflow.register_model" in src
    assert "set_registered_model_alias" in src
