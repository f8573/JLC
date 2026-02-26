package net.faulj.web;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.http.HttpServletRequest;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ContactController {
    private static final Logger LOG = LoggerFactory.getLogger(ContactController.class);
    private static final long ONE_HOUR_MS = 60L * 60L * 1000L;
    private static final Pattern MENTION_PATTERN = Pattern.compile("<@!?\\d+>|<@&\\d+>|<#\\d+>");
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NEVER)
            .build();

    private final InMemoryRateLimiter rateLimiter;
    private final int perIpLimitPerHour;
    private final int maxFieldLength;
    private final int maxMessageLength;
    private final boolean blockCrossSite;

    public ContactController(
            InMemoryRateLimiter rateLimiter,
            @Value("${app.contact.rate-limit-per-hour:12}") int perIpLimitPerHour,
            @Value("${app.contact.max-field-length:256}") int maxFieldLength,
            @Value("${app.contact.max-message-length:4000}") int maxMessageLength,
            @Value("${app.contact.block-cross-site:true}") boolean blockCrossSite) {
        this.rateLimiter = rateLimiter;
        this.perIpLimitPerHour = Math.max(1, perIpLimitPerHour);
        this.maxFieldLength = Math.max(64, maxFieldLength);
        this.maxMessageLength = Math.max(256, maxMessageLength);
        this.blockCrossSite = blockCrossSite;
    }

    @PostMapping("/api/contact")
    public ResponseEntity<Map<String, String>> contact(
            @RequestBody(required = false) Map<String, Object> payload,
            HttpServletRequest request) {
        if (blockCrossSite && isCrossSiteFetch(request)) {
            return ResponseEntity.status(403).body(Map.of(
                    "status", "error",
                    "message", "Cross-site invocation is blocked"));
        }

        if (payload == null || payload.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "status", "error",
                    "message", "Request body is required"));
        }

        String clientIp = clientIp(request);
        if (!rateLimiter.allow("contact:" + clientIp, perIpLimitPerHour, ONE_HOUR_MS)) {
            return ResponseEntity.status(429).body(Map.of(
                    "status", "error",
                    "message", "Rate limit exceeded. Please try again later."));
        }

        URI webhookUri = parseWebhookUri(System.getenv("DISCORD_WEBHOOK_URL"));
        if (webhookUri == null) {
            LOG.error("DISCORD_WEBHOOK_URL is missing or invalid. Contact relay is disabled.");
            return ResponseEntity.status(500).body(Map.of(
                    "status", "error",
                    "message", "Contact delivery is not configured"));
        }

        String content = buildContent(payload, clientIp);
        if (content.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "status", "error",
                    "message", "Message content is empty"));
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("content", content);
        // Prevent webhook mention abuse from user-provided text.
        body.put("allowed_mentions", Map.of("parse", new String[0]));

        String json;
        try {
            json = MAPPER.writeValueAsString(body);
        } catch (Exception ex) {
            LOG.error("Failed to serialize contact payload", ex);
            return ResponseEntity.status(500).body(Map.of(
                    "status", "error",
                    "message", "Failed to process request"));
        }

        boolean delivered = sendWebhookWithRetry(webhookUri, json, clientIp);
        if (!delivered) {
            return ResponseEntity.status(502).body(Map.of(
                    "status", "error",
                    "message", "Failed to deliver message"));
        }

        return ResponseEntity.ok(Map.of("status", "ok"));
    }

    private boolean sendWebhookWithRetry(URI webhookUri, String json, String clientIp) {
        int maxAttempts = 3;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                HttpRequest httpReq = HttpRequest.newBuilder()
                        .uri(webhookUri)
                        .timeout(Duration.ofSeconds(15))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(json))
                        .build();

                HttpResponse<String> resp = CLIENT.send(httpReq, HttpResponse.BodyHandlers.ofString());
                int status = resp.statusCode();
                if (status == 204 || status == 200 || status == 201) {
                    return true;
                }

                if (status == 429) {
                    long retryMs = parseRetryAfterMs(resp.body());
                    sleepQuietly(retryMs);
                    continue;
                }

                LOG.warn("Discord webhook rejected contact payload (status={}) for client {}", status, clientIp);
                return false;
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                LOG.warn("Interrupted while sending contact webhook for client {}", clientIp);
                return false;
            } catch (IOException ioe) {
                LOG.warn("Transient webhook delivery error for client {} (attempt {}/{})",
                        clientIp, attempt, maxAttempts);
                sleepQuietly(Math.min(3000L, 300L * attempt));
            }
        }
        return false;
    }

    private static long parseRetryAfterMs(String body) {
        try {
            JsonNode node = MAPPER.readTree(body);
            double retryRaw = node.path("retry_after").asDouble(1.0);
            long retryMs = retryRaw > 1000.0 ? (long) retryRaw : (long) (retryRaw * 1000.0);
            return Math.max(250L, Math.min(5000L, retryMs));
        } catch (Exception ignored) {
            return 1000L;
        }
    }

    private static void sleepQuietly(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
    }

    private String buildContent(Map<String, Object> payload, String clientIp) {
        StringBuilder sb = new StringBuilder();
        sb.append("New contact form submission\n");

        appendIfPresent(sb, "Type", safeField(payload, "formType", maxFieldLength, false));
        appendIfPresent(sb, "Reporter", safeField(payload, "reporterEmail", maxFieldLength, false));
        appendIfPresent(sb, "Name", safeField(payload, "name", maxFieldLength, false));
        appendIfPresent(sb, "Subject", safeField(payload, "subject", maxFieldLength, false));

        String matrix = safeField(payload, "matrixContext", Math.min(maxMessageLength, 2000), true);
        if (!matrix.isBlank()) {
            sb.append("Matrix:\n```").append(matrix).append("```\n");
        }

        appendIfPresent(sb, "Details", safeField(payload, "details", maxMessageLength, true));
        appendIfPresent(sb, "Message", safeField(payload, "message", maxMessageLength, true));
        appendIfPresent(sb, "Timestamp", safeField(payload, "timestamp", maxFieldLength, false));
        sb.append("IP: ").append(clientIp).append('\n');

        String content = sb.toString();
        // Discord webhook content max length is 2000 characters.
        if (content.length() > 1900) {
            content = content.substring(0, 1900);
        }
        return content;
    }

    private static void appendIfPresent(StringBuilder sb, String label, String value) {
        if (!value.isBlank()) {
            sb.append(label).append(": ").append(value).append('\n');
        }
    }

    private String safeField(Map<String, Object> payload, String key, int limit, boolean multiline) {
        Object raw = payload.get(key);
        if (raw == null) {
            return "";
        }

        String text = Objects.toString(raw, "");
        String normalized = multiline
                ? text.replace('\r', '\n')
                : text.replace('\r', ' ').replace('\n', ' ');
        normalized = normalized.trim();
        if (!multiline) {
            normalized = normalized.replaceAll("\\s+", " ");
        }

        normalized = normalized
                .replace("@everyone", "[@everyone]")
                .replace("@here", "[@here]");
        normalized = MENTION_PATTERN.matcher(normalized).replaceAll("[mention]");

        int max = Math.max(1, limit);
        if (normalized.length() > max) {
            normalized = normalized.substring(0, max);
        }
        return normalized;
    }

    private static URI parseWebhookUri(String webhook) {
        if (webhook == null || webhook.isBlank()) {
            return null;
        }
        try {
            URI uri = URI.create(webhook.trim());
            String scheme = uri.getScheme();
            String host = uri.getHost();
            String path = uri.getPath();
            if (!"https".equalsIgnoreCase(scheme) || host == null || path == null) {
                return null;
            }

            String hostLower = host.toLowerCase();
            boolean allowedHost = hostLower.equals("discord.com")
                    || hostLower.endsWith(".discord.com")
                    || hostLower.equals("discordapp.com")
                    || hostLower.endsWith(".discordapp.com");
            if (!allowedHost) {
                return null;
            }
            if (!path.startsWith("/api/webhooks/")) {
                return null;
            }
            return uri;
        } catch (Exception ex) {
            return null;
        }
    }

    private static boolean isCrossSiteFetch(HttpServletRequest request) {
        if (request == null) {
            return false;
        }

        String secFetchSite = request.getHeader("Sec-Fetch-Site");
        if (secFetchSite != null && "cross-site".equalsIgnoreCase(secFetchSite.trim())) {
            return true;
        }

        String origin = request.getHeader("Origin");
        if (origin == null || origin.isBlank()) {
            return false;
        }
        String hostHeader = request.getHeader("Host");
        if (hostHeader == null || hostHeader.isBlank()) {
            return true;
        }

        String requestHost = hostHeader.split(":")[0].toLowerCase(Locale.ROOT);
        try {
            URI originUri = URI.create(origin);
            String originHost = originUri.getHost();
            if (originHost == null || originHost.isBlank()) {
                return true;
            }
            return !originHost.equalsIgnoreCase(requestHost);
        } catch (Exception ex) {
            return true;
        }
    }

    private static String clientIp(HttpServletRequest request) {
        if (request == null) {
            return "unknown";
        }
        String ip = request.getRemoteAddr();
        return (ip == null || ip.isBlank()) ? "unknown" : ip;
    }
}
