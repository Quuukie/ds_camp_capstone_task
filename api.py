# Standart library imports
from contextlib import asynccontextmanager

# Thirdparty imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

# Local imports
from config import LLMConfig
from config import SemanticSearchConfig as config
from ds_capstone.data_loader import load_and_preprocess_data
from ds_capstone.models import SearchOutput, SearchResultItem, SummaryOutput, TextInput
from ds_capstone.search_index import FaissSearchHandler
from ds_capstone.summarizer_graph import SummarizerAgent
from ds_capstone.vectorizer import TextVectorizer


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    # This function is called before the application starts
    app.state.summarizer_graph = SummarizerAgent(LLMConfig.MODEL_NAME, base_url=LLMConfig.OLLAMA_BASE_URL)
    app.state.vectorizer = TextVectorizer(config.EMBEDDING_MODEL_NAME, query_prefix=config.QUERY_PREFIX)
    app.state.products_df = load_and_preprocess_data(config.PRODUCTS_CSV_PATH)
    print(f"Loaded {len(app.state.products_df)} products from {config.PRODUCTS_CSV_PATH}")

    dimension = app.state.vectorizer.model.config.hidden_size
    app.state.search_index = FaissSearchHandler(dimension)
    index_path = config.FAISS_INDEX_PATH

    index_loaded = app.state.search_index.load(index_path)
    if index_loaded and app.state.search_index.index.ntotal != len(app.state.products_df):
        # The catalog changed since the index was built, so the stored vectors no longer line up with the rows
        print(f"FAISS index at {index_path} is stale. Rebuilding...")
        app.state.search_index = FaissSearchHandler(dimension)
        index_loaded = False

    if index_loaded:
        print(f"Successfully loaded FAISS index from {index_path}")
    else:
        print(f"Creating a new FAISS index at {index_path}...")
        print("Vectorizing product descriptions...")
        texts = app.state.products_df['text'].tolist()
        embeddings = app.state.vectorizer.embed(texts)
        print("Building FAISS index...")
        app.state.search_index.build(embeddings)
        print(f"Saving new index to {index_path}...")
        app.state.search_index.save(index_path)

    yield
    print("Application is shutting down...")


app = FastAPI(lifespan=lifespan)


# Define a route for the root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():  # type: ignore
    return """
    <html>
        <head>
            <title>FastAPI</title>
        </head>
        <body>
            <h1>Welcome to FastAPI!</h1>
        </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


# Plain `def` routes run in FastAPI's threadpool, so blocking model calls don't stall the event loop
@app.post("/summarize/", response_model=SummaryOutput)
def summarize_text(input_data: TextInput, request: Request):  # type: ignore
    summary = request.app.state.summarizer_graph.execute(input_message=input_data.text)
    return SummaryOutput(summary=summary)


@app.post("/semantic_search/", response_model=SearchOutput)
def semantic_search(query: TextInput, request: Request):  # type: ignore
    # Load items from lifespan
    vectorizer = request.app.state.vectorizer
    search_index = request.app.state.search_index
    products_df = request.app.state.products_df

    query_embedding = vectorizer.embed([query.text], is_query=True)
    distances, indices = search_index.search(query_embedding, k=config.TOP_K)
    results: list[SearchResultItem] = []
    for idx, score in zip(indices[0], distances[0]):
        # FAISS pads with -1 when the index holds fewer than k vectors
        if idx < 0 or score < config.SIMILARITY_THRESHOLD:
            continue
        product = products_df.iloc[idx]
        results.append(
            SearchResultItem(
                rank=len(results) + 1,
                title=product['title'],
                brand=product['brand'],
                category=product['category'],
                description=product['description'],
                similarity=min(float(score), 1.0),
            )
        )

    if not results:
        raise HTTPException(status_code=404, detail="No matches found.")

    return SearchOutput(results=results)
