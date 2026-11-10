# ScamShield US — Architecture Deck Narrative Plan

## Audience
GW NLP course instructors, teammates, and reviewers evaluating the ScamShield US final project.

## Objective
Explain the end-to-end architecture, offline training/indexing pipeline, online investigation workflow, tech stack, S3 layout, evaluation results, and deployment path.

## Narrative arc
1. What the system is and why it exists
2. High-level architecture (Angular → FastAPI → S3 / models)
3. Offline pipeline that builds durable artifacts
4. Online agent workflow for a user message
5. Classification approaches and hybrid cascade
6. RAG retrieval + grounded answer generation
7. Storage / stack / metrics / how to run

## Slide list
1. Title
2. Problem & goals
3. System architecture
4. Offline pipeline workflow
5. Online investigation workflow
6. Classification approaches
7. RAG & answer generation
8. S3 data layout
9. Technology stack
10. Evaluation results
11. Deployment & safety

## Source plan
- Repository README and scripts (`run_pipeline.sh`, `setup_venv.sh`)
- `backend/app/agents/workflow.py`, classifier, retriever, LLM modules
- S3 metrics: `approach_metrics.csv` / `approach_summary.json`
- Course stack constraint: EC2 + S3 only, open-source models

## Visual system
- Modern sans: Poppins titles, Lato body
- Palette: deep navy `#0B1F2A`, teal accent `#1FA87A`, slate cards, white panels
- Architecture diagrams as editable cards/arrows; one native column chart for approach macro-F1
