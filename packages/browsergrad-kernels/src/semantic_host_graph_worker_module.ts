import {
  installSemanticHostGraphWorkerEntry,
  type SemanticHostGraphWorkerEntryScope,
} from "./semantic_host_graph_worker_transport.js";

installSemanticHostGraphWorkerEntry(
  globalThis as unknown as SemanticHostGraphWorkerEntryScope,
);
