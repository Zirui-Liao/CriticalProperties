from __future__ import annotations

from typing import Any, Dict

from chemprop import nn as cp_nn, models as cp_models


def build_mpnn_from_params(
    n_tasks: int,
    scaler,
    hparams: Dict[str, Any],
) -> cp_models.MPNN:
    """Build a Chemprop MPNN from combined structure and training params."""
    mp = cp_nn.BondMessagePassing(
        d_h=hparams["message_hidden_dim"],
        depth=hparams["depth"],
        dropout=hparams["MPNN_DROPOUT"],
    )
    agg = cp_nn.NormAggregation()
    output_transform = cp_nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = cp_nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim,
        hidden_dim=hparams["ffn_hidden_dim"],
        n_layers=hparams["ffn_num_layers"],
        dropout=hparams["FFN_DROPOUT"],
        output_transform=output_transform,
    )
    return cp_models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        warmup_epochs=hparams["warmup_epochs"],
        init_lr=hparams["init_lr"],
        max_lr=hparams["max_lr"],
        final_lr=hparams["final_lr"],
    )
