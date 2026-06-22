export const VALID_PROFILE = {
  id: "cs336-assignment1",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  metadata: {
    title: "Stanford CS336 Assignment 1: Basics",
    course: "Stanford CS336",
    source_url: "https://github.com/stanford-cs336/assignment1-basics",
    lecture_urls: ["https://www.youtube.com/watch?v=example"],
    tags: ["language-modeling", "tokenization"],
  },
  runtime_packages: ["numpy", "regex", "pytest"],
  files: {
    root: "/assignments/cs336-assignment1",
    rubric_path: "rubric.py",
    starter_path: "assignment.py",
    reference_path: "reference.py",
    fixtures_path: "fixtures",
  },
  timeouts: {
    setup_ms: 10_000,
    test_ms: 30_000,
    worker_ms: 60_000,
  },
  allowed_tests: ["test_train_bpe_tiny", "test_encode_iterable_streams"],
  oracles: [
    {
      name: "_bg_tokenizers",
      js_module: "/assets/tokenizer-oracle.js",
      export_name: "oracle",
    },
  ],
  gates: [
    {
      name: "browser_runtime",
      kind: "capability",
      options: { requires: ["pyodide"] },
    },
    {
      name: "encode_iterable_streaming",
      kind: "streaming",
      options: { max_chunks_before_first_yield: 2 },
    },
  ],
  datasets: [{ name: "tiny", url: "/fixtures/tiny.txt", hash: "sha256:abc" }],
};

