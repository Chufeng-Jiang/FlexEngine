"""Optional Prometheus registry with no metrics in tests if import unavailable."""
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover
    CollectorRegistry = None


class RuntimeMetrics:
    def __init__(self):
        if CollectorRegistry is None:
            self.registry = None
            return
        self.registry = CollectorRegistry()
        self.requests = Counter("flex_requests_total", "Requests received", registry=self.registry)
        self.completed = Counter("flex_completed_total", "Terminated requests", ["reason"], registry=self.registry)
        self.tokens = Counter("flex_output_tokens_total", "Output tokens produced", registry=self.registry)
        self.preemptions = Counter("flex_preemptions_total", "Recompute-based preemptions", registry=self.registry)
        self.prefix_hits = Counter("flex_prefix_cache_hits_total", "Prefix cache hits", registry=self.registry)
        self.prefix_saved = Counter("flex_prefix_saved_tokens_total", "Prefill tokens skipped", registry=self.registry)
        self.prefix_evictions = Counter("flex_prefix_cache_evictions_total", "Prefix entries evicted", registry=self.registry)
        self.queue = Gauge("flex_waiting_requests", "Waiting requests", registry=self.registry)
        self.running = Gauge("flex_running_requests", "Executing requests", registry=self.registry)
        self.blocks = Gauge("flex_used_kv_blocks", "KV physical blocks in use", registry=self.registry)
        self.shared_blocks = Gauge("flex_shared_prefix_blocks", "KV blocks owned by prefix entries", registry=self.registry)
        self.ttft = Histogram("flex_ttft_seconds", "Time to first output token", registry=self.registry)
        self.e2e = Histogram("flex_e2e_seconds", "Request E2E latency", registry=self.registry)

    def export(self) -> tuple[bytes, str]:
        if self.registry is None:
            return b"# prometheus_client not installed\n", "text/plain"
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def update_gauges(self, waiting: int, running: int, blocks: int, shared_blocks: int = 0) -> None:
        if self.registry is not None:
            self.queue.set(waiting)
            self.running.set(running)
            self.blocks.set(blocks)
            self.shared_blocks.set(shared_blocks)
