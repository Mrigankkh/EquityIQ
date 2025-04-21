# EquityIQ

EquityIQ is an AI‑powered, multi‑agent financial analysis system that automates extraction, analysis, and summarization of complex financial documents. By leveraging CrewAI for agent orchestration, Retrieval‑Augmented Generation (RAG) for grounded retrieval, and domain‑fine‑tuned language models, EquityIQ enables efficient, accurate, and customizable investment research.

## Features

- **Multi‑Agent Architecture**  
  - **Researcher Agent**: Retrieves relevant document passages via dense vector retrieval (BAAI/bge‑small‑en‑v1.5) and a top‑K RAG pipeline.  
  - **Quant Agent**: Parses filings, extracts structured financial metrics (revenue, EBITDA, year‑over‑year changes, etc.) into CSV/Excel.  
  - **Writer Agent**: Synthesizes narrative responses in selectable personas (Analyst, Advisor, Journalist, Report Writer) using LLaMA 3.3 70B (Groq API).  
  - **Briefing Agent**: Generates concise, domain‑aware summaries with a BART‑base model fine‑tuned on the FINDSum dataset.

- **Retrieval‑Augmented Generation**  
  - Vector embeddings indexed via llama_index for fast, scalable similarity search.  
  - Deterministic retrieval (temperature 0) to minimize hallucinations.

- **Domain‑Fine‑Tuned Models**  
  - **LLaMA‑3** variants for retrieval, analytics, and generation.  
  - **BART‑base** fine‑tuned on FINDSum for high‑quality financial summarization (ROUGE‑1: 0.3984, ROUGE‑2: 0.1382, ROUGE‑L: 0.2091).

- **Transparent & Extensible**  
  - End‑to‑end open‑source pipeline, reproducible with publicly available models and datasets.  
  - Modular agent definitions allow easy customization and extension to other domains.

## Architecture Overview

1. **Vector Store Initialization**  
   - Parse raw SEC filings, embed with BAAI/bge‑small‑en‑v1.5, and index via `llama_index`.  
2. **Query Processing**  
   - **Researcher** retrieves top‑K chunks (K = 5) for a user query.  
   - **Writer** drafts a user‑facing response based on retrieved context.  
   - **Quant** extracts tabular metrics via structured prompts and LlamaParse.  
   - **Briefing** summarizes full reports with the fine‑tuned BART model.

## Installation

```bash
git clone https://github.com/Mrigankkh/EquityIQ.git
cd EquityIQ
pip install -r requirements.txt
```

### Environment Variables

- `GROQ_API_KEY` – API key for Groq access to LLaMA 3.  
- `HUGGINGFACE_API_TOKEN` – Token for Hugging Face model and dataset access.

## Usage

1. **Build or Load Vector Store**  
   ```bash
   python scripts/create_vector_store.py --data-dir data/filings --index-dir vector_store/
   ```

2. **Run an Interactive Query**  
   ```bash
   python scripts/run_query.py \
     --query "What are Tesla’s key risk factors for 2024?" \
     --persona FinancialAnalyst \
     --output results/tesla_risk_analysis.md
   ```

3. **Extract Structured Metrics**  
   ```bash
   python scripts/run_quant_agent.py \
     --company Amazon \
     --years 2022 2023 2024 \
     --output results/amazon_metrics.csv
   ```

4. **Generate a Briefing Summary**  
   ```bash
   python scripts/run_briefing_agent.py \
     --company Microsoft \
     --year 2023 \
     --output results/msft_summary.md
   ```

## Datasets

- **FINDSum** (Financial Report Document Summarization)  
  Thousands of document–summary pairs covering earnings reports, regulatory filings, and announcements. Available via [Papers with Code](https://paperswithcode.com/dataset/findsum).

## Empirical Results

- **RAG Evaluation** (40-question test set):  
  - Faithfulness: 0.8428  
  - Context Precision: 0.8717  
  - Context Recall: 0.9067  

- **Summarization (BART‑base fine‑tuned)**:  
  - ROUGE‑1: 0.3984  
  - ROUGE‑2: 0.1382  
  - ROUGE‑L: 0.2091  
  - Perplexity: 7.80  

## Repository Structure

```
EquityIQ/
├── data/                  # Raw filings and documents
├── vector_store/          # Persisted llama_index vector indices
├── scripts/               # Agent entrypoints and utility scripts
│   ├── create_vector_store.py
│   ├── run_query.py
│   ├── run_quant_agent.py
│   ├── run_briefing_agent.py
├── models/                # Local checkpoints or fine‑tuned models
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open issues or pull requests for:

- New agent personas or roles  
- Integration with additional data sources (e.g., real‑time market feeds)  
- Enhanced hallucination detection or numerical reasoning modules  
- Expanded evaluation metrics or user studies

## Authors & Contact

- **Mrigank Khandelwal** – khandelwal.mr@northeastern.edu  
- **Gargi Kelaskar** – kelaskar.g@northeastern.edu  
- **Shashank Joshi** – joshi.shash@northeastern.edu  

Northeastern University, DS 5983

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
