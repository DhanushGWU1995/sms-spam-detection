import { CommonModule } from '@angular/common';
import { Component, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  AnalyzeResponse,
  ApproachMetrics,
  ApproachReportsResponse,
  AttentionTerm,
  DashboardVizResponse,
  HealthResponse,
  InvestigateResponse,
  KnowledgeSourcesResponse,
  LabelMetricRow,
  LimeFeature,
} from './models/scamshield.models';
import { ScamshieldApiService } from './services/scamshield-api.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  private readonly api = inject(ScamshieldApiService);

  title = 'ScamShield US';
  message = '';
  question = '';
  status = 'Checking API…';
  busy = false;
  health: HealthResponse | null = null;
  analysis: AnalyzeResponse | null = null;
  investigation: InvestigateResponse | null = null;
  approachReports: ApproachReportsResponse | null = null;
  dashboardViz: DashboardVizResponse | null = null;
  knowledgeSources: KnowledgeSourcesResponse | null = null;
  selectedApproach = '';
  selectedSampleId = '';
  error = '';

  /** Curated from the test set using Traditional (LR) top-margin vs hybrid threshold. */
  readonly sampleMessages: Array<{
    id: string;
    title: string;
    kind: 'spam' | 'ham';
    route: 'traditional' | 'escalate';
    blurb: string;
    text: string;
  }> = [
    {
      id: 'spam-trad',
      title: 'Spam · Traditional only',
      kind: 'spam',
      route: 'traditional',
      blurb: 'Clear phishing cues → strong Traditional margin, no DistilBERT escalate.',
      text: 'CITI Msg A hold have be place on your banking Click on the link below for more information',
    },
    {
      id: 'ham-trad',
      title: 'Ham · Traditional only',
      kind: 'ham',
      route: 'traditional',
      blurb: 'Ordinary personal SMS → strong Traditional confidence, stays on Traditional.',
      text: "Ohh okay we'll reach about the same time!",
    },
    {
      id: 'spam-esc',
      title: 'Spam · Escalate to Advanced',
      kind: 'spam',
      route: 'escalate',
      blurb: 'Edge-case spam wording → weak Traditional margin → Hybrid routes to DistilBERT.',
      text: 'Hello James, your FEDEX package with tracking code GB-6412-GH83 is waiting for you to set delivery preferences:',
    },
    {
      id: 'ham-esc',
      title: 'Ham · Escalate to Advanced',
      kind: 'ham',
      route: 'escalate',
      blurb: 'Ambiguous everyday SMS → weak Traditional margin → Hybrid routes to DistilBERT.',
      text: 'Daddy at home, what time you will be back to fetch me to airport.',
    },
  ];

  ngOnInit(): void {
    this.api.health().subscribe({
      next: (health) => {
        this.health = health;
        this.status = health.open_source_only
          ? `Ready · ${health.project} · open-source models only`
          : `Ready · ${health.project}`;
      },
      error: () => {
        this.status = 'API offline. Start FastAPI on port 8000, then refresh.';
      },
    });
    this.api.approachMetrics().subscribe({
      next: (reports) => {
        this.approachReports = reports;
        if (reports.available && reports.order?.length) {
          this.selectedApproach = reports.order[0];
        }
      },
      error: () => {
        this.approachReports = {
          available: false,
          note: 'Could not load approach metrics.',
        };
      },
    });
    this.api.dashboardViz().subscribe({
      next: (viz) => {
        this.dashboardViz = viz;
      },
      error: () => {
        this.dashboardViz = {
          available: false,
          note: 'Could not load dashboard visualizations.',
        };
      },
    });
    this.api.knowledgeSources().subscribe({
      next: (sources) => {
        this.knowledgeSources = sources;
      },
      error: () => {
        this.knowledgeSources = {
          available: false,
          note: 'Could not load RAG knowledge sources.',
          documents: [],
        };
      },
    });
  }

  useSample(sampleId: string): void {
    const sample = this.sampleMessages.find((item) => item.id === sampleId);
    if (!sample) {
      return;
    }
    this.selectedSampleId = sample.id;
    this.message = sample.text;
    this.analysis = null;
    this.investigation = null;
    this.error = '';
    this.status = `Loaded sample: ${sample.title}. Click Analyze, then add a question to Investigate.`;
  }

  onMessageChange(): void {
    // Editing the message invalidates the prior Analyze result used by Investigate/LLM.
    if (this.analysis || this.investigation) {
      this.analysis = null;
      this.investigation = null;
    }
  }

  canAnalyze(): boolean {
    return this.message.trim().length > 0;
  }

  isSpamPrediction(): boolean {
    const label = (this.analysis?.label || '').trim().toLowerCase();
    return !!label && label !== 'legitimate';
  }

  canInvestigate(): boolean {
    return (
      this.message.trim().length > 0 &&
      this.question.trim().length > 0 &&
      this.analysis !== null &&
      this.isSpamPrediction()
    );
  }

  investigateDisabledReason(): string {
    if (this.canInvestigate()) {
      return 'Run the agent workflow with scam RAG + local Qwen';
    }
    if (this.analysis && !this.isSpamPrediction()) {
      return 'Investigate is only available for spam / scam predictions (RAG is scam-focused)';
    }
    const missing: string[] = [];
    if (!this.message.trim()) {
      missing.push('a suspicious message');
    }
    if (!this.question.trim()) {
      missing.push('a question');
    }
    if (!this.analysis) {
      missing.push('a completed Analyze with a spam prediction');
    }
    return `Need ${missing.join(', ')} before Investigate`;
  }

  analyze(): void {
    const text = this.message.trim();
    if (!text) {
      this.status = 'Paste a message first.';
      return;
    }
    this.busy = true;
    this.error = '';
    this.investigation = null;
    // Use Traditional LR so escalate_to_advanced reflects Hybrid Traditional→Advanced routing.
    this.status = 'Running Traditional NLP (logistic regression)…';
    this.api.analyze({ message: text, model: 'logistic_regression' }).subscribe({
      next: (result) => {
        this.analysis = result;
        const trigger = result.escalate_to_advanced
          ? ' Weak Traditional margin → Hybrid would escalate to DistilBERT (Advanced).'
          : ' Strong Traditional margin → Hybrid keeps Traditional only.';
        this.status = `Predicted ${result.label} (${(result.confidence * 100).toFixed(1)}%) via ${result.model}.${trigger}`;
        this.busy = false;
      },
      error: (err: unknown) => {
        this.error = this.readError(err);
        this.status = 'Analysis failed.';
        this.busy = false;
      },
    });
  }

  investigate(): void {
    if (!this.canInvestigate()) {
      this.status = this.investigateDisabledReason();
      return;
    }
    const text = this.message.trim();
    const question = this.question.trim();
    this.busy = true;
    this.error = '';
    this.status = 'Running agent workflow (classifier → stats → hybrid RAG → local Qwen)…';
    this.api.investigate({ message: text, question }).subscribe({
      next: (result) => {
        this.investigation = result;
        this.analysis = result.classification;
        this.status = 'Investigation finished.';
        this.busy = false;
      },
      error: (err: unknown) => {
        this.error = this.readError(err);
        this.status = 'Investigation failed.';
        this.busy = false;
      },
    });
  }

  approachNames(): string[] {
    return this.approachReports?.order ?? [];
  }

  testApproaches(): Record<string, ApproachMetrics> | null {
    return this.approachReports?.splits?.test?.approaches ?? null;
  }

  selectedConfusion(): number[][] | null {
    const approaches = this.testApproaches();
    if (!approaches || !this.selectedApproach) {
      return null;
    }
    return approaches[this.selectedApproach]?.confusion ?? null;
  }

  selectedLabelMetrics(): LabelMetricRow[] {
    const approaches = this.testApproaches();
    if (!approaches || !this.selectedApproach) {
      return [];
    }
    return approaches[this.selectedApproach]?.label_metrics ?? [];
  }

  labelName(index: number): string {
    return this.approachReports?.labels?.[index] ?? String(index);
  }

  cellIntensity(value: number, matrix: number[][]): number {
    const max = Math.max(...matrix.flat(), 1);
    return Math.max(0.12, value / max);
  }

  pct(value: number | undefined | null): string {
    return `${((value ?? 0) * 100).toFixed(1)}%`;
  }

  spamCloudSrc(): string | null {
    return this.dashboardViz?.wordclouds?.spam_wordcloud ?? null;
  }

  hamCloudSrc(): string | null {
    return this.dashboardViz?.wordclouds?.ham_wordcloud ?? null;
  }

  spamLimeCloudSrc(): string | null {
    return (
      this.dashboardViz?.lime?.images?.spam_lime_wordcloud ??
      this.dashboardViz?.wordclouds?.spam_lime_wordcloud ??
      null
    );
  }

  hamLimeCloudSrc(): string | null {
    return (
      this.dashboardViz?.lime?.images?.ham_lime_wordcloud ??
      this.dashboardViz?.wordclouds?.ham_lime_wordcloud ??
      null
    );
  }

  spamLimeExampleSrc(): string | null {
    return (
      this.dashboardViz?.lime?.images?.spam_lime_example ??
      this.dashboardViz?.wordclouds?.spam_lime_example ??
      null
    );
  }

  hamLimeExampleSrc(): string | null {
    return (
      this.dashboardViz?.lime?.images?.ham_lime_example ??
      this.dashboardViz?.wordclouds?.ham_lime_example ??
      null
    );
  }

  limeAvailable(): boolean {
    return Boolean(this.dashboardViz?.lime?.available);
  }

  limeNote(): string {
    return this.dashboardViz?.lime?.note ?? '';
  }

  spamLimeFeatures(): LimeFeature[] {
    return this.dashboardViz?.lime?.top_features?.spam ?? [];
  }

  hamLimeFeatures(): LimeFeature[] {
    return this.dashboardViz?.lime?.top_features?.ham ?? [];
  }

  spamTerms(): AttentionTerm[] {
    return this.dashboardViz?.attention_terms?.spam ?? [];
  }

  hamTerms(): AttentionTerm[] {
    return this.dashboardViz?.attention_terms?.ham ?? [];
  }

  termBarWidth(term: AttentionTerm, pool: AttentionTerm[]): number {
    const max = Math.max(...pool.map((item) => item.share), 0.0001);
    return Math.max(8, (term.share / max) * 100);
  }

  limeBarWidth(feature: LimeFeature, pool: LimeFeature[]): number {
    const weights = pool.map((item) => Math.abs(item.mean_weight ?? item.weight ?? 0));
    const max = Math.max(...weights, 0.0001);
    const value = Math.abs(feature.mean_weight ?? feature.weight ?? 0);
    return Math.max(8, (value / max) * 100);
  }

  limeWeightLabel(feature: LimeFeature): string {
    const value = feature.mean_weight ?? feature.weight ?? 0;
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(3)}`;
  }

  private readError(err: unknown): string {
    if (typeof err === 'string') {
      return err;
    }
    if (err && typeof err === 'object' && 'message' in err) {
      return String((err as { message: string }).message);
    }
    return 'Request failed.';
  }
}
