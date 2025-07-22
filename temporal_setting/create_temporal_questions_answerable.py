import datasets
import torch
import requests
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHARED_PROMPT = "Generate a question based on the passage below that will contain the main event in the passage using the entities. Feel include to incorporate temporal entities like date if provided, etc, but be specific, DO NOT use phrases like `this year` or `this month` or specify the day of the week if you give a month, day, or week include the exact date and include the year."


def load_model_tokenizer(model_path, quant_type=torch.bfloat16):
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=quant_type, device_map="auto"
    )
    print(model.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_news_articles(total_articles, api_key):
    total_articles = min(
        total_articles, 200
    )  # only 200 articles can be fetched at a time
    # Get the news articles from the Guardian API
    i = 0
    print("Getting news articles", total_articles)
    body_texts = []
    years = []
    # url = f'https://content.guardianapis.com/search?order-by=relevance&from-date=2024-06-17&page-size={total_articles}&type=article&api-key={api_key}&show-fields=all'
    for year in range(1990, 2020):
        print("On Year: ", year)
        month = np.random.choice(
            ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        )
        print("On Month: ", month)

        url = f"https://content.guardianapis.com/search?order-by=relevance&from-date={year}-{month}-01&to-date={year}-{month}-01&page-size={total_articles}&type=article&api-key={api_key}&show-fields=all"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        for article in news_data["response"]["results"]:
            body_text = article["fields"]["bodyText"]
            body_texts.append(body_text)
            years.append(year)
            i += 1
    return body_texts, years


def generate_updated_question(example, model, tokenizer, system_prompt=SHARED_PROMPT):
    article = example["article"]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    message = [
        {
            "role": "system",
            "content": f"{SHARED_PROMPT}. The passage is from the year {example['year']}. Be specific and ONLY return the question.",
        },
        {"role": "user", "content": f"{article}"},
    ]
    example["question"] = pipe(message)[0]["generated_text"][-1]["content"].strip()
    print("New Question: ", example["question"])
    return example


def get_temporal_questions(model, tokenizer, system_prompt, guardian_key):
    import nltk
    from nltk.tokenize import sent_tokenize

    nltk.download("punkt")
    articles, years = get_news_articles(100, guardian_key)

    all_articles_tenSent = []
    for text, year in zip(articles, years):
        sentences = sent_tokenize(text)
        first_ten_sent_string = " ".join(sentences[:10])
        all_articles_tenSent.append(
            {
                "article": first_ten_sent_string,
                "question_category": "temporal_question",
                "year": year,
            }
        )

    dataset_arts = datasets.Dataset.from_list(all_articles_tenSent)
    return dataset_arts.map(
        generate_updated_question,
        fn_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
        },
    )


if __name__ == "__main__":
    import argparse

    MODEL_PATH = "meta-llama/Meta-Llama-3-70B-Instruct"
    SHARED_PROMPT = "Generate a question based on the passage below that will contain the main event in the passage using the entities. Feel include to incorporate temporal entities like date if provided, etc, but be specific, DO NOT use phrases like `this year` or `this month` or specify the day of the week if you give a month, day, or week include the exact date and include the year."

    parser = argparse.ArgumentParser(
        prog="create_temporal_questions_answerable.py",
        description="This script evaluates the resonses for refusal messages",
    )
    parser.add_argument("-o", "--output_data", type=str, default=None)
    parser.add_argument("-sp", "--system_prompt", type=str, default=SHARED_PROMPT)
    parser.add_argument("-m", "--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--guardian_key", type=str, required=True, default=None)
    args = parser.parse_args()
    print(args)
    # if output data none then rewrite input data
    if args.output_data is None:
        args.output_data = "refusal_eval_data/temporal_split_answerable.jsonl"
    print("----------Getting Model----------")
    model, tokenizer = load_model_tokenizer(args.model_path, quant_type=torch.bfloat16)
    ##### Now we make the actual questions #####
    temporal_dataset = get_temporal_questions(
        model, tokenizer, args.system_prompt, args.guardian_key
    )
    print("------------HF DATASET------------")
    print(temporal_dataset)

    if "json" in args.output_data:
        temporal_dataset.to_json(args.output_data)
    elif "csv" in args.output_data:
        temporal_dataset.to_csv(args.output_data)
    elif "jsonl" in args.output_data:
        temporal_dataset.to_json(args.output_data)
    else:
        raise NotImplementedError("Only handles `csv` and `json` output formats")
