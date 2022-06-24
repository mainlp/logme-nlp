import argparse, csv, json
import numpy as np

from collections import OrderedDict
from scipy.stats import weightedtau


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Human LM Ranking Evaluation')
	arg_parser.add_argument('target_path', help='path to gold rankings in JSON format')
	arg_parser.add_argument('survey_path', help='path to ranking survey results in CSV format')
	arg_parser.add_argument(
		'-pp', '--print_participants', action='store_true', default=False,
		help='flag for printing participant-level correlations')
	return arg_parser.parse_args()


def load_target_rankings(path):
	# rankings {'setup': {'task': [model_0_f1, ...] } } (num_setups -> num_tasks -> num_models)
	rankings = OrderedDict()
	logme_scores = OrderedDict()
	# map of model and task names to indices
	models = None
	# iterate over setups
	with open(path, 'r') as fp:
		results = json.load(fp)
		for setup in results:
			rankings[setup] = OrderedDict()
			logme_scores[setup] = OrderedDict()
			for task in results[setup]:
				if models is None:
					models = {name: idx for idx, name in enumerate(results[setup][task]['model'])}
				# rankings[setup][task] = np.argsort(results[setup][task]["f1"], )
				rankings[setup][task] = results[setup][task]['f1']
				logme_scores[setup][task] = results[setup][task]['logme']
	return rankings, logme_scores, models


def load_survey_rankings(path, models):
	# rankings [ [ [model_0_rank, ...] ,... ], ...] (num_participants -> num_tasks -> num_models)
	rankings = []
	# load file
	with open(path, 'r', encoding='utf8', newline='') as fp:
		csv_reader = csv.reader(fp)
		# iterate over participants
		for ridx, row in enumerate(csv_reader):
			if ridx == 0: continue
			participant_rankings = []
			# split by task
			for task_start_idx in range(2, len(row), len(models)):
				task_ranking = row[task_start_idx:task_start_idx+len(models)]
				# check if models were selected twice
				if len(set(task_ranking)) != len(models):
					print(f"[Warning] Some models were selected more than once for participant {ridx}.")
				# create reverse ranked list (i.e. rank 7 of 7 has score 0, rank 0 of 7 has score 7)
				participant_rankings.append(
					[len(models) - task_ranking.index(m) - 1 for m in models]
				)
			rankings.append(participant_rankings)

	return rankings


def print_tau_statistics(taus, indent=0):
	taus = np.array(taus)
	tau_mean = np.arctanh(np.mean(np.tanh(taus)))
	print(f"{' ' * indent}Mean tau: {tau_mean:.4f}")
	print(f"{' ' * indent}Range of tau: {min(taus):.4f} (min) / {max(taus):.4f} (max)")
	print(f"{' ' * indent}Correlation sign: {np.sum(taus <= 0)} (-) / {np.sum(taus > .5)} (>.5) / {np.sum(taus > 0)} (+)")


def main():
	args = parse_arguments()

	# load rankings
	target_rankings, logme_scores, models = load_target_rankings(args.target_path)
	print(f"Loaded target rankings for {len(target_rankings)} setups, {len(models)} models.")
	survey_rankings = load_survey_rankings(args.survey_path, models)
	print(f"Loaded survey rankings for {len(survey_rankings)} participants.")

	# iterate over setups
	taus, logme_taus = [], []
	num_logme_wins = 0
	for setup in target_rankings:
		print(f"Setup '{setup}':")
		setup_taus, setup_logme_taus = [], []
		# iterate over tasks
		for tidx, task in enumerate(target_rankings[setup]):
			print(f"  Task '{task}':")
			# retrieve relevant scores
			task_target_ranking = target_rankings[setup][task]
			# iterate over participants
			task_taus = []
			for pidx, rankings in enumerate(survey_rankings):
				# retrieve relevant rankings
				task_survey_ranking = rankings[tidx]
				# compute correlations
				tau, p = weightedtau(task_target_ranking, task_survey_ranking)
				if args.print_participants:
					print(f"    Participant {pidx+1}: tau = {tau}, p = {p}")
				task_taus.append(tau)
			# print statistics
			print_tau_statistics(task_taus, indent=4)
			# compute majority ranking
			tau, p = weightedtau(task_target_ranking, np.sum(rankings, axis=0))
			print(f"    Majority tau: tau = {tau:.4f}, p = {p}")
			# compute LogME tau
			tau, p = weightedtau(task_target_ranking, logme_scores[setup][task])
			print(f"    LogME: tau = {tau}, p = {p}")
			setup_taus += task_taus
			setup_logme_taus.append(tau)
			num_logme_wins += np.sum(np.array(task_taus) > tau)
		# print statistics
		print("  * Humans:")
		print_tau_statistics(setup_taus, indent=2)
		print("  * LogME:")
		print_tau_statistics(setup_logme_taus, indent=2)
		taus += setup_taus
		logme_taus += setup_logme_taus
	print("Overall:")
	print("* Humans:")
	print_tau_statistics(taus)
	print("* LogME:")
	print_tau_statistics(logme_taus)
	print(f"LogME wins: {num_logme_wins}/{len(taus)} ({(num_logme_wins * 100)/len(taus):.2f}%)")


if __name__ == '__main__':
	main()
