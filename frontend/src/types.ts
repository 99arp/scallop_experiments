export interface Series {
  series_id: string;
  source_kind: string | null;
  source_name: string | null;
  source_signal: string | null;
  fluent: string | null;
  value: string | null;
  probability: number | null;
  declaration_chain: string | null;
  intervals: Array<[number, number]>;
}

export interface Snapshot {
  updated_at: string;
  status: string;
  graph_svg: string;
  sample_index: number | null;
  timestamp_ns: number | null;
  poll_ms: number;
  axis: { start: number; end: number };
  series: Series[];
}

export interface EcInterval {
  source_kind: string;
  name: string;
  start: number;
  end: number;
  p_min: number;
  p_max: number;
}

export interface EcSnapshot {
  updated_at: string;
  sample_index: number | null;
  intervals: EcInterval[];
  axis: { start: number; end: number };
}

export interface ScallopAnswer {
  probability: number;
  tuple: (string | number | boolean)[];
}

export interface ScallopQueryResult {
  ok: boolean;
  query?: string;
  answers?: ScallopAnswer[];
  answer_count?: number;
  error?: string;
}

export interface KnowledgeBaseAnswer {
  term: string;
  probability: number;
}

export interface KnowledgeBaseQueryResult {
  ok: boolean;
  query?: string;
  answers?: KnowledgeBaseAnswer[];
  answer_count?: number;
  truncated?: boolean;
  error?: string;
  error_type?: string;
}
