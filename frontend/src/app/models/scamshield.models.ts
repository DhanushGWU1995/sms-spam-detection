export interface AnalyzeRequest {
  message: string;
  model?: string;
}

export interface AskRequest {
  message: string;
  question: string;
}

export interface MessageSignals {
  urls: string[];
  emails: string[];
  phones: string[];
  money_mentions: string[];
  urgency_cues: string[];
  credential_requests: string[];
  payment_methods: string[];
  impersonated_orgs: string[];
  asks_for_sensitive_info: boolean;
  score_hints: Record<string, number>;
}

export interface AnalyzeResponse {
  model: string;
  label: string;
  confidence: number;
  top_margin?: number;
  weak_confidence_margin?: number;
  llm_would_trigger?: boolean;
  escalate_to_advanced?: boolean;
  probabilities: Record<string, number>;
  signals: MessageSignals;
  disclaimer: string;
}

export interface LabelMetricRow {
  label: string;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ApproachMetrics {
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  macro_precision: number;
  macro_recall: number;
  confusion: number[][];
  label_metrics?: LabelMetricRow[];
  routed_to_llm?: number;
  routed_to_advanced?: number;
  description?: string;
  flips_correct?: number;
  flips_wrong?: number;
  routed_messages?: number;
  messages?: number;
}

export interface DatasetUsed {
  name: string;
  url?: string;
  rows?: number;
}

export interface ApproachReportsResponse {
  available: boolean;
  note?: string;
  margin?: number | null;
  hybrid_margin?: number | null;
  labels?: string[];
  order?: string[];
  primary_metric?: string;
  dataset_name?: string;
  datasets_used?: DatasetUsed[];
  comparison_split?: string;
  weak_confidence_rule?: string;
  hybrid_rule?: string;
  splits?: {
    test?: {
      messages: number;
      approaches: Record<string, ApproachMetrics>;
    };
    val?: {
      messages: number;
      approaches: Record<string, ApproachMetrics>;
    };
  };
  summary?: Record<string, unknown>;
}

export interface AttentionTerm {
  term: string;
  docs: number;
  share: number;
}

export interface DashboardVizResponse {
  available: boolean;
  note?: string;
  wordclouds?: {
    spam_wordcloud?: string;
    ham_wordcloud?: string;
  };
  attention_terms?: {
    spam?: AttentionTerm[];
    ham?: AttentionTerm[];
    note?: string;
  };
  label_mix?: Array<{ label: string; count: number }>;
  binary_mix?: { spam: number; ham: number };
}

export interface Citation {
  n: number;
  title?: string;
  agency?: string;
  source_url?: string;
  retrieved_date?: string;
}

export interface LlmInputPayload {
  model?: string;
  system?: string;
  user?: string;
  messages?: Array<{ role: string; content: string }>;
  chat_template_prompt?: string | null;
  retrieved_passage_count?: number;
  context_preview?: string;
  consumer_question_bundle?: string;
  submitted_message?: string;
  user_question?: string;
  classification?: {
    label?: string;
    confidence?: number;
    model?: string;
  };
  generation?: Record<string, unknown>;
  note?: string;
}

export interface InvestigateResponse {
  classification: AnalyzeResponse;
  stats: Record<string, unknown>;
  retrieval: Array<Record<string, unknown>>;
  answer: string;
  model: string;
  citations: Citation[];
  llm_input?: LlmInputPayload;
  validation: {
    ok: boolean;
    warnings: string[];
    evidence_count: number;
  };
  trace: string[];
  guardrails: Record<string, boolean>;
}

export interface HealthResponse {
  status: string;
  project: string;
  llm: string;
  embedding: string;
  classifier_models: string[];
  rag_ready: boolean;
  open_source_only: boolean;
}

export interface KnowledgeSourceDoc {
  doc_id?: string;
  agency?: string;
  title?: string;
  source_url?: string;
  topic?: string;
  retrieved_date?: string;
}

export interface KnowledgeIndexChunk {
  chunk_id?: string;
  doc_id?: string;
  agency?: string;
  title?: string;
  topic?: string;
  source_url?: string;
  retrieved_date?: string;
  char_count?: number;
  text?: string;
}

export interface KnowledgeSourcesResponse {
  available: boolean;
  note?: string;
  retrieved_date?: string;
  count?: number;
  documents?: KnowledgeSourceDoc[];
  index?: {
    available?: boolean;
    chunk_count?: number;
    embedding_model?: string;
    dim?: number;
    faiss_type?: string;
    total_chars?: number;
  };
  chunks?: KnowledgeIndexChunk[];
}
