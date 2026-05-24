from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
import ollama

model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(texts):
    return model.encode(texts)


def add_embeddings(graph_df):

    graph_df["combined_text"] = graph_df.apply(
        lambda row: f"{row['node_1']} {row['edge']} {row['node_2']}",
        axis=1
    )

    graph_df["embedding"] = list(
        generate_embeddings(
            graph_df["combined_text"].tolist()
        )
    )

    return graph_df


def get_context(chunk_ids, chunk_df):

    texts = []

    for cid in chunk_ids:

        try:

            cid = int(cid.strip())

            match = chunk_df[
                chunk_df["chunk_id"] == cid
            ]

            if not match.empty:
                texts.append(
                    match.iloc[0]["content"]
                )

        except:
            pass

    return "\n".join(texts)


def answer_query(query, graph_df, chunk_df):

    query_embedding = generate_embeddings(
        [query]
    )[0]

    graph_df["similarity"] = graph_df[
        "embedding"
    ].apply(
        lambda emb:
        1 - cosine(query_embedding, emb)
    )

    top_rows = graph_df.sort_values(
        by="similarity",
        ascending=False
    ).head(3)

    relationships = []
    context_parts = []

    for _, row in top_rows.iterrows():

        relationships.append(
            f"{row['node_1']} -> {row['edge']} -> {row['node_2']}"
        )

        chunk_ids = str(
            row["chunk_id"]
        ).split(",")

        context_parts.append(
            get_context(chunk_ids, chunk_df)
        )

    relationships_text = "\n".join(
        relationships
    )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
    Answer the question using only the information below.

    Relationships:
    {relationships_text}

    Context:
    {context}

    Question:
    {query}
    """

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


dfg2 = add_embeddings(dfg2)

print("GraphRAG system ready!")

while True:

    query = input("\nAsk Question: ")

    if query.lower() in ["exit", "quit"]:
        break

    answer = answer_query(
        query,
        dfg2,
        df
    )

    print("\nAnswer:\n")
    print(answer)