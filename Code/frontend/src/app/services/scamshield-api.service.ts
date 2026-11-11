import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import {
  AnalyzeRequest,
  AnalyzeResponse,
  ApproachReportsResponse,
  AskRequest,
  DashboardVizResponse,
  HealthResponse,
  InvestigateResponse,
  KnowledgeSourcesResponse,
} from '../models/scamshield.models';

@Injectable({ providedIn: 'root' })
export class ScamshieldApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  analyze(payload: AnalyzeRequest): Observable<AnalyzeResponse> {
    return this.http.post<AnalyzeResponse>(`${this.baseUrl}/analyze`, payload);
  }

  investigate(payload: AskRequest): Observable<InvestigateResponse> {
    return this.http.post<InvestigateResponse>(`${this.baseUrl}/ask`, payload);
  }

  approachMetrics(): Observable<ApproachReportsResponse> {
    return this.http.get<ApproachReportsResponse>(`${this.baseUrl}/metrics/approaches`);
  }

  dashboardViz(): Observable<DashboardVizResponse> {
    return this.http.get<DashboardVizResponse>(`${this.baseUrl}/viz/overview`);
  }

  knowledgeSources(): Observable<KnowledgeSourcesResponse> {
    return this.http.get<KnowledgeSourcesResponse>(`${this.baseUrl}/knowledge/sources`);
  }
}
