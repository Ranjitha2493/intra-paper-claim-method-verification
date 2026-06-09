# Intra-Paper Claim Verification

This repository contains the code, prompts, evaluation materials, and supporting artifacts for the paper:



\*\*Do Methods Support the Claims? Intra-Paper Verification for Peer Review\*\*



\## Overview



Large language models (LLMs) are increasingly being explored as tools for supporting scientific peer review. Existing automated novelty-assessment systems primarily evaluate whether a paper's claimed contributions differ from prior literature. However, reviewers frequently challenge novelty claims because the methodological evidence presented in a paper does not adequately support those claims.



This repository accompanies a framework for \*\*intra-paper claim verification\*\*, a task that evaluates whether novelty claims described in a paper are substantiated by the methodological evidence provided within the same paper.



The framework extracts novelty claims from the Introduction, retrieves supporting methodological evidence, performs claim substantiation analysis, and generates structured reviewer-style assessments that can be compared with human reviewer concerns.



\## Framework Components



\### Novelty Claim Extraction



Novelty claims are extracted from the Introduction section of scientific papers using GPT-4o.



\### Human-Inspired Evaluation Categories



Reviewer comments from ICLR 2025 papers are analyzed to identify recurring reviewer concerns. This process yields four evaluation categories:



\* Novelty Concern

\* Methodological Issue

\* Clarity Issue

\* Other Issues



These categories are used to organize both human reviewer comments and framework-generated assessments.



\### Claim–Method Verification



Methodological evidence is retrieved for each extracted novelty claim and analyzed to determine whether the paper provides sufficient support for the claimed contribution.



\### Structured Review Generation



Claim-verification findings are transformed into structured reviewer-style assessments organized according to the four evaluation categories.



\### Evaluation



Framework-generated assessments are compared with human reviewer feedback using:



\* Human evaluator assessment

\* SBERT-based category-level semantic similarity

\* BERTScore-based review-level semantic similarity



\## Repository Structure



```text

01\_extract\_introduction\_methods/

02\_extract\_novelty\_claims/

03\_claim\_method\_verification/

04\_generate\_structured\_reviews/

05\_extract\_human\_review\_categories/

06\_generate\_review\_summaries/

07\_human\_inspired\_evaluation\_categories/

prompts/

evaluation\_materials/

sample\_outputs/

```



\## Data Sources



\* Scientific papers from ICLR 2025

\* Human reviewer comments collected from OpenReview

\* PDF-to-Markdown conversion performed using MinerU



\## Citation



If you use this repository, please cite the associated paper.



