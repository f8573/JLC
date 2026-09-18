package net.faulj.web;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jakarta.servlet.http.HttpServletRequest;

@RestController
@CrossOrigin(originPatterns = {
        "http://localhost",
        "http://localhost:*",
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://0.0.0.0",
        "http://0.0.0.0:*",
        "https://localhost",
        "https://localhost:*",
        "https://127.0.0.1",
        "https://127.0.0.1:*",
        "https://0.0.0.0",
        "https://0.0.0.0:*",
        "http://lambdacompute.org",
        "http://lambdacompute.org:*",
        "https://lambdacompute.org",
        "https://lambdacompute.org:*",
        "http://www.lambdacompute.org",
        "https://www.lambdacompute.org"
})
public class ContactController {
    private static final Logger LOG = LoggerFactory.getLogger(ContactController.class);

    static {
        String webhookEnv = System.getenv("DISCORD_WEBHOOK_URL");
        LOG.info("DISCORD_WEBHOOK_URL present: {}", webhookEnv != null && !webhookEnv.isEmpty());
    }
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    @PostMapping("/api/contact")
    public ResponseEntity<Map<String, String>> contact(@RequestBody Map<String, Object> payload,
            HttpServletRequest request) {
        String webhook = System.getenv("DISCORD_WEBHOOK_URL");
        if (webhook == null || webhook.isEmpty()) {
            Map<String, String> resp = new HashMap<>();
            resp.put("status", "error");
            resp.put("message", "Webhook not configured");
            return ResponseEntity.status(500).body(resp);
        }

        String content = buildContent(payload, request);

        Map<String, Object> body = new HashMap<>();
        body.put("content", content);

        String json;
        try {
            json = MAPPER.writeValueAsString(body);
        } catch (Exception e) {
            Map<String, String> resp = new HashMap<>();
            resp.put("status", "error");
            return ResponseEntity.status(500).body(resp);
        }

        int maxAttempts = 3;
        int attempt = 0;
        while (attempt < maxAttempts) {
            attempt++;
            try {
                HttpRequest httpReq = HttpRequest.newBuilder()
                        .uri(URI.create(webhook))
                        .timeout(Duration.ofSeconds(15))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(json))
                        .build();

                HttpResponse<String> resp = CLIENT.send(httpReq, HttpResponse.BodyHandlers.ofString());
                int status = resp.statusCode();
                if (status == 204 || status == 200 || status == 201) {
                    break; // success
                }

                if (status == 429) {
                    // rate limited; parse retry_after
                    try {
                        JsonNode node = MAPPER.readTree(resp.body());
                        long retryMs = node.path("retry_after").asLong(1000L);
                        Thread.sleep(Math.max(500L, retryMs));
                    } catch (Exception ex) {
                        Thread.sleep(1000L);
                    }
                    continue;
                }

                // non-retryable error: break and log
                System.err.println("Discord webhook failed status=" + status + " body=" + resp.body());
                break;
            } catch (IOException | InterruptedException ex) {
                System.err.println("Discord webhook exception: " + ex.getMessage());
                try {
                    Thread.sleep(500L * attempt);
                } catch (InterruptedException ie) {
                    // ignore
                }
            }
        }

        Map<String, String> ok = new HashMap<>();
        ok.put("status", "ok");
        return ResponseEntity.ok(ok);
    }

    private String buildContent(Map<String, Object> payload, HttpServletRequest request) {
        StringBuilder sb = new StringBuilder();
        sb.append("New contact form submission\n");

        Object formType = payload.get("formType");
        if (formType != null) {
            sb.append("Type: ").append(formType.toString()).append("\n");
        }

        Object reporter = payload.get("reporterEmail");
        if (reporter != null) {
            sb.append("Reporter: ").append(reporter.toString()).append("\n");
        }

        Object name = payload.get("name");
        if (name != null) {
            sb.append("Name: ").append(name.toString()).append("\n");
        }

        Object subject = payload.get("subject");
        if (subject != null) {
            sb.append("Subject: ").append(subject.toString()).append("\n");
        }

        Object matrix = payload.get("matrixContext");
        if (matrix != null) {
            sb.append("Matrix:\n```").append(matrix.toString()).append("```\n");
        }

        Object details = payload.get("details");
        if (details != null) {
            sb.append("Details: ").append(details.toString()).append("\n");
        }

        Object message = payload.get("message");
        if (message != null) {
            sb.append("Message: ").append(message.toString()).append("\n");
        }

        Object ts = payload.get("timestamp");
        if (ts != null) {
            sb.append("Timestamp: ").append(ts.toString()).append("\n");
        }

        sb.append("IP: ").append(request.getRemoteAddr()).append('\n');
        return sb.toString();
    }
}
