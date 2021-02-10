import PerfumeRecommender as pir
import argparse
import train_recommender as trainer


parser = argparse.ArgumentParser()
parser.add_argument("--train_models", type=bool, default=False, help="Generate pickle weight files")
parser.add_argument("--model_dir", type=str, default="models", help="Path where models will-be/are stored")
parser.add_argument("--input_data_csv", type=str, default="final_perfume_data.csv", help="Path to input csv file with perfume data")
parser.add_argument("--query_string", type=str, default="I want something with jasmine, vanilla and cedar but I do not want tobacco", help="Query string for searching perfumes")
parser.add_argument("--num_recommendations", type=int, default=5, help="Number of perfumes to be recommended.")
args = parser.parse_args()

if args.train_models:
	trainer.train_models(args.input_data_csv, args.model_dir)

pir.load_models(args.input_data_csv, args.model_dir)
recommended_perfumes = pir.find_similar_perfumes(args.query_string, args.num_recommendations)
print(recommended_perfumes)
recommendations = pir.details_of_recommendations(recommended_perfumes)
print(len(recommendations))
for row in recommendations:
	print(row[0])
	print(row[1])
	print(row[2])