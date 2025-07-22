import datasets
import torch
import json, copy, requests
from transformers import pipeline, LlamaForCausalLM, AutoTokenizer
import pandas as pd
import sys

sys.path.append("../")

from utils import turn_on_proxy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHARED_PROMPT = "Generate a question based on the passage below that will contain the main event in the passage using the entities. Feel include to incorporate temporal entities like the current year, date, etc. Feel free to say `this year`, `2024`, `next month`, `today`, `this week`, etc. Be specific and ONLY return the question."


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
    # These should be live but there is a way to do from DATE in the url search
    # url = f'https://content.guardianapis.com/search?order-by=relevance&from-date=2024-06-17&page-size={total_articles}&type=article&api-key={api_key}&show-fields=all'
    for month in ["04", "05", "06"]:
        for day in ["01", "07", "14", "21"]:
            url = f"https://content.guardianapis.com/search?order-by=relevance&from-date=2024-{month}-{day}&page-size={total_articles}&type=article&api-key={api_key}&show-fields=all"
            response = requests.get(url)
            response.raise_for_status()
            news_data = response.json()
            for article in news_data["response"]["results"]:
                body_text = article["fields"]["bodyText"]
                body_texts.append(body_text)
                i += 1
    return body_texts


def generate_updated_question(example, model, tokenizer, system_prompt=SHARED_PROMPT):
    article = example["article"]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    message = [
        {"role": "system", "content": f"{SHARED_PROMPT}"},
        {"role": "user", "content": f"{article}"},
    ]
    example["question"] = pipe(message)[0]["generated_text"][-1]["content"].strip()
    print("New Question: ", example["question"])
    return example


def get_temporal_questions(model, tokenizer, system_prompt):
    import nltk
    from nltk.tokenize import sent_tokenize

    nltk.download("punkt")
    articles = get_news_articles(200, "39e5b80c-7cf8-43ae-bbb0-63ab6c4a5370")

    all_articles_tenSent = []
    for text in articles:
        sentences = sent_tokenize(text)
        first_ten_sent_string = " ".join(sentences[:10])
        all_articles_tenSent.append(
            {"article": first_ten_sent_string, "question_category": "temporal_question"}
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
    parser = argparse.ArgumentParser(
        prog="create_temporal_questions.py",
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
        args.output_data = "refusal_eval_data/temporal_split_unanswerable.jsonl"

    print("----------Getting Model----------")
    model, tokenizer = load_model_tokenizer(args.model_path, quant_type=torch.bfloat16)
    ##### Now we make the actual questions #####
    temporal_dataset = get_temporal_questions(model, tokenizer, args.system_prompt)
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
