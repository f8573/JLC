package net.faulj.web;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Component;

/**
 * Lightweight per-key fixed-window rate limiter for API abuse throttling.
 */
@Component
public class InMemoryRateLimiter {
    private static final int CLEANUP_TRIGGER_SIZE = 8_192;
    private static final long DEFAULT_STALE_MS = 60L * 60L * 1000L;

    private static final class Window {
        private long windowStartMs;
        private int count;
        private long lastSeenMs;

        Window(long now) {
            this.windowStartMs = now;
            this.lastSeenMs = now;
            this.count = 0;
        }
    }

    private final ConcurrentHashMap<String, Window> windows = new ConcurrentHashMap<>();

    /**
     * Return true when request is allowed for key within fixed window.
     */
    public boolean allow(String key, int limit, long windowMs) {
        if (key == null || key.isBlank() || limit <= 0 || windowMs <= 0) {
            return false;
        }

        long now = System.currentTimeMillis();
        Window win = windows.computeIfAbsent(key, k -> new Window(now));
        boolean allowed;

        synchronized (win) {
            if ((now - win.windowStartMs) >= windowMs) {
                win.windowStartMs = now;
                win.count = 0;
            }
            win.lastSeenMs = now;
            if (win.count >= limit) {
                allowed = false;
            } else {
                win.count++;
                allowed = true;
            }
        }

        if (windows.size() > CLEANUP_TRIGGER_SIZE) {
            cleanup(now, Math.max(windowMs, DEFAULT_STALE_MS));
        }
        return allowed;
    }

    private void cleanup(long now, long staleMs) {
        Iterator<Map.Entry<String, Window>> it = windows.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, Window> entry = it.next();
            Window win = entry.getValue();
            if (win == null) {
                it.remove();
                continue;
            }

            boolean stale;
            synchronized (win) {
                stale = (now - win.lastSeenMs) > staleMs;
            }
            if (stale) {
                it.remove();
            }
        }
    }
}
