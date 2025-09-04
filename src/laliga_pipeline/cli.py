import argparse, logging
from pathlib import Path
from .config import Config
from .logging_conf import setup_logging
from .loaders import load_all
from .features import to_long, rolling_form, assemble
from .backtest import rolling_origin

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[5,10])
    p.add_argument("--xi", type=float, default=0.001)
    p.add_argument("--max_goals", type=int, default=10)
    p.add_argument("--min_train_matches", type=int, default=380)
    args=p.parse_args()
    setup_logging(); log=logging.getLogger("laliga_pipeline")

    cfg=Config(args.data_dir, args.out_dir, tuple(args.windows), args.xi, args.max_goals, args.min_train_matches)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df=load_all(cfg.data_dir)
    df.to_csv(cfg.out_dir/"matches_clean.csv", index=False)
    log.info("Loaded %d matches across %d seasons", len(df), df["Season"].nunique())

    long=to_long(df)
    form=rolling_form(long, cfg.windows)
    form.to_csv(cfg.out_dir/"team_form.csv", index=False)
    features=assemble(df, form)
    features.to_csv(cfg.out_dir/"matches_features.csv", index=False)

    metrics=rolling_origin(df[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]], cfg.min_train_matches, cfg.xi, cfg.max_goals)
    log.info("Backtest | n=%d | log_loss=%.4f | brier=%.4f", metrics["n"], metrics["log_loss"], metrics["brier"])

if __name__ == "__main__":
    main()
