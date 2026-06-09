\# Tools and Implementation



This document describes the software, models, libraries, and infrastructure used in the implementation and evaluation of the intra-paper claim verification framework.



\## Document Processing



\### MinerU



Scientific papers in PDF format were converted into structured Markdown using MinerU.



Repository:

https://github.com/opendatalab/MinerU



Purpose:



\* PDF-to-Markdown conversion

\* Extraction of Introduction and Methods sections

\* Preservation of document structure



\## Large Language Model



\### GPT-4o



GPT-4o was used throughout the framework for:



\* Novelty claim extraction

\* Reviewer concern analysis

\* Evaluation category discovery

\* Claim–method verification

\* Structured review generation

\* Review summarization



Access was provided through the OpenAI API.



\## Human Review Data



\### OpenReview



Human reviewer comments were collected from OpenReview.



https://openreview.net



The review corpus consisted of:



\* 182 ICLR 2025 papers

\* 786 individual reviews



These reviews were used to derive the human-inspired evaluation categories employed throughout the framework.



\## Semantic Similarity Models



\### Sentence-BERT (SBERT)



Model:



```text

all-mpnet-base-v2

```



Library:



```text

sentence-transformers

```



Usage:



\* Category-level semantic similarity analysis

\* Human vs. framework-generated review comparison



\### BERTScore



Library:



```text

bert-score

```



Usage:



\* Review-level semantic similarity analysis

\* Real-vs-fake review discrimination experiments



\## Statistical Analysis



Statistical analyses were implemented in Python using:



\### pandas



\* Data manipulation

\* Dataset preparation



\### NumPy



\* Numerical computation



\### SciPy



\* One-sample hypothesis testing

\* Paired t-tests

\* Wilcoxon signed-rank tests

\* Correlation analysis



\## Visualization



\### Matplotlib



Used for:



\* Similarity distributions

\* Experimental result visualizations



\### Seaborn



Used for:



\* Statistical plots

\* Distribution analysis



\## Compute Infrastructure



Large-scale document processing and analysis were performed using the Bridges-2 system at the Pittsburgh Supercomputing Center (PSC).



https://www.psc.edu/resources/bridges-2/



\## Programming Language



Python 3.11



The framework was implemented using modular Python scripts corresponding to each processing stage.



