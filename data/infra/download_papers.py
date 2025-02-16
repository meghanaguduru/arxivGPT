import arxiv

search = arxiv.Search(query="transformers NLP", max_results=10, sort_by=arxiv.SortCriterion.Relevance)

for paper in search.results():
    print(f"Downloading {paper.title}")
    paper.download_pdf(dirpath="data/")