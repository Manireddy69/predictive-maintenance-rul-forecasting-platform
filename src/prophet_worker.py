from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .sequence_attention_model import ProphetEnsembleConfig, _load_prophet_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a Prophet baseline and emit serialized forecasts.")
    parser.add_argument("--train-csv", required=True, help="CSV with ds/y training columns.")
    parser.add_argument("--score-csv", required=True, help="CSV with ds scoring column.")
    parser.add_argument("--config-json", required=True, help="Serialized ProphetEnsembleConfig payload.")
    parser.add_argument("--output-json", required=True, help="JSON file for forecast outputs.")
    parser.add_argument("--seasonality-column", default="cycle_seasonality", help="Custom seasonality column name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Prophet = _load_prophet_runtime()

    train_csv = Path(args.train_csv)
    score_csv = Path(args.score_csv)
    config_json = Path(args.config_json)
    output_json = Path(args.output_json)

    config = ProphetEnsembleConfig(**json.loads(config_json.read_text(encoding="utf-8")))
    training_frame = pd.read_csv(train_csv)
    scoring_frame = pd.read_csv(score_csv)

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=config.changepoint_prior_scale,
        seasonality_prior_scale=config.seasonality_prior_scale,
    )
    model.add_seasonality(
        name=args.seasonality_column,
        period=config.seasonality_period,
        fourier_order=config.seasonality_fourier_order,
    )
    model.fit(training_frame.loc[:, ["ds", "y"]])
    forecast = model.predict(scoring_frame.loc[:, ["ds"]])
    seasonality_column = args.seasonality_column if args.seasonality_column in forecast.columns else "additive_terms"

    output_json.write_text(
        json.dumps(
            {
                "predictions": forecast["yhat"].astype(float).tolist(),
                "trend": forecast["trend"].astype(float).tolist(),
                "seasonality": forecast[seasonality_column].astype(float).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
