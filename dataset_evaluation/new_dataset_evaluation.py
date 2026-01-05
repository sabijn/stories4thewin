import pandas as pd
import numpy as np

from pathlib import Path
from tqdm.notebook import tqdm


import ast

from vendi_score import text_utils

from diversity import (
	compression_ratio
)

def load_or_run_eval(eval_f, dataset, column, path_name, *, run=False):
	if Path(path_name).exists() and not run:
		df = pd.read_csv(path_name)
		if 'lexical_diversity' in df.columns:
			df['lexical_diversity'] = df['lexical_diversity'].apply(ast.literal_eval)
			
		return df

	dataset = eval_f.run_pipeline_on_df(dataset, column)
	dataset.to_csv(path_name, index=False)
	return dataset

def sample_and_run_eval(df, outdir, eval_f, n_runs, sample_size, column):
	run_summaries = []
	
	for run in tqdm(range(1, n_runs + 1)):
		seed = 1000 + run	# reproducible but different each run
		sample = df.sample(n=sample_size, replace=False, random_state=seed).copy()

		save_name = outdir / f'eval_results_new_dataset_run{run}_seed{seed}.csv'
		metrics_df = load_or_run_eval(eval_f, sample, column, save_name)

		metrics_df["lexical_diversity"] = metrics_df["lexical_diversity"].apply(lambda d: next(iter(d.values())))
		run_summary = metrics_df.mean(numeric_only=True).to_dict()
		run_summary["run"] = run
		run_summary["seed"] = seed

		run_summaries.append(run_summary)

	return run_summaries

def calculate_overall_statistics(run_summaries, out_dir):
	runs_df = pd.DataFrame(run_summaries).sort_values("run")
	runs_df_path = out_dir / "runs_summary.csv"
	runs_df.to_csv(runs_df_path, index=False)

	metric_cols = [c for c in runs_df.columns if c not in ["run", "seed"]]

	final_summary = pd.DataFrame({
		"metric": metric_cols,
		"mean_over_runs": [runs_df[c].mean() for c in metric_cols],
		"std_over_runs":  [runs_df[c].std(ddof=1) for c in metric_cols],  # sample std across runs
	})

	final_summary_path = out_dir / "final_mean_std.csv"
	final_summary.to_csv(final_summary_path, index=False)

	return final_summary

def new_dataset_eval(df, n_runs, sample_size, text_col, out_dir, eval_f):
	run_summaries = sample_and_run_eval(df, out_dir, eval_f, n_runs, sample_size, text_col)
	final_summary = calculate_overall_statistics(run_summaries, out_dir)

	return final_summary

def run_overall_dataset_statistics(df, outdir, n_runs, sample_size, column):
	dataset_run_rows = []

	for run in range(1, n_runs + 1):
		seed = 1000 + run
		sample = df.sample(n=sample_size, random_state=seed).copy()

		texts = list(map(str, sample[column].to_list()))

		# --- dataset-level metrics (one value per run) ---
		embed_vs = text_utils.embedding_vendi_score(
			texts, model_path="jegormeister/bert-base-dutch-cased"
		)
		comp_score = compression_ratio(texts, algorithm="gzip")

		dataset_run_rows.append({
			"run": run,
			"seed": seed,
			"vendi": embed_vs,
			"compression": comp_score,
		})

	dataset_runs_df = pd.DataFrame(dataset_run_rows)
	dataset_runs_df.to_csv(outdir / "dataset_metrics_runs.csv", index=False)
	
	return dataset_runs_df