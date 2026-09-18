package net.faulj.web;

import java.time.Instant;
import java.util.Deque;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.TimeUnit;

/**
 * Simple in-memory sliding-window rate limiter keyed by client IP.
 * Note: for multi-node deployments use Redis or a distributed store.
 */
public class IpRateLimiter {
    private final ConcurrentHashMap<String, Deque<Long>> map = new ConcurrentHashMap<>();
    private final long windowMs;
    private final int maxRequestsPerWindow;

    public IpRateLimiter() {
        this(TimeUnit.MINUTES.toMillis(1), 5);
    }

    public IpRateLimiter(long windowMs, int maxRequestsPerWindow) {
        this.windowMs = windowMs;
        this.maxRequestsPerWindow = Math.max(1, maxRequestsPerWindow);
    }

    public boolean allowRequest(String ip) {
        if (ip == null) ip = "unknown";
        long now = Instant.now().toEpochMilli();
        Deque<Long> dq = map.computeIfAbsent(ip, k -> new ConcurrentLinkedDeque<>());
        // remove expired
        while (true) {
            Long head = dq.peekFirst();
            if (head == null) break;
            if (now - head > windowMs) dq.pollFirst(); else break;
        }
        if (dq.size() >= maxRequestsPerWindow) return false;
        dq.addLast(now);
        return true;
    }
}
