package net.faulj.web;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ReadListener;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletInputStream;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Enforce hard body-size limits for JSON API payloads before deserialization.
 */
@Component
public class RequestSizeLimitFilter extends OncePerRequestFilter {
    @Value("${app.security.max-json-bytes:1048576}")
    private int maxJsonBytes;

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getRequestURI();
        String contextPath = request.getContextPath();
        if (path == null) {
            return true;
        }
        if (contextPath != null && !contextPath.isBlank() && path.startsWith(contextPath)) {
            path = path.substring(contextPath.length());
        }
        if (!path.startsWith("/api/")) {
            return true;
        }

        String method = request.getMethod();
        if (!"POST".equalsIgnoreCase(method)
                && !"PUT".equalsIgnoreCase(method)
                && !"PATCH".equalsIgnoreCase(method)) {
            return true;
        }

        String contentType = request.getContentType();
        if (contentType == null) {
            return true;
        }
        String normalized = contentType.toLowerCase(Locale.ROOT);
        return !(normalized.contains(MediaType.APPLICATION_JSON_VALUE) || normalized.contains("+json"));
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        int max = Math.max(1, maxJsonBytes);
        long contentLength = request.getContentLengthLong();
        if (contentLength > max) {
            reject(response, max);
            return;
        }

        byte[] body;
        try {
            body = readLimited(request.getInputStream(), max);
        } catch (PayloadTooLargeException ex) {
            reject(response, max);
            return;
        }

        CachedBodyRequest wrapped = new CachedBodyRequest(request, body);
        filterChain.doFilter(wrapped, response);
    }

    private static byte[] readLimited(InputStream in, int maxBytes) throws IOException, PayloadTooLargeException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.min(maxBytes, 16 * 1024));
        byte[] buf = new byte[8 * 1024];
        int total = 0;
        int read;
        while ((read = in.read(buf)) != -1) {
            total += read;
            if (total > maxBytes) {
                throw new PayloadTooLargeException();
            }
            out.write(buf, 0, read);
        }
        return out.toByteArray();
    }

    private static void reject(HttpServletResponse response, int maxBytes) throws IOException {
        response.setStatus(HttpServletResponse.SC_REQUEST_ENTITY_TOO_LARGE);
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.getWriter().write("{\"error\":\"payload too large\",\"details\":\"Maximum JSON payload is "
                + maxBytes + " bytes.\"}");
    }

    private static final class PayloadTooLargeException extends Exception {
        private static final long serialVersionUID = 1L;
    }

    private static final class CachedBodyRequest extends HttpServletRequestWrapper {
        private final byte[] body;

        CachedBodyRequest(HttpServletRequest request, byte[] body) {
            super(request);
            this.body = body == null ? new byte[0] : body;
        }

        @Override
        public ServletInputStream getInputStream() {
            ByteArrayInputStream bis = new ByteArrayInputStream(body);
            return new ServletInputStream() {
                @Override
                public int read() {
                    return bis.read();
                }

                @Override
                public boolean isFinished() {
                    return bis.available() == 0;
                }

                @Override
                public boolean isReady() {
                    return true;
                }

                @Override
                public void setReadListener(ReadListener listener) {
                    // Synchronous read only.
                }
            };
        }

        @Override
        public BufferedReader getReader() {
            return new BufferedReader(new InputStreamReader(getInputStream(), StandardCharsets.UTF_8));
        }
    }
}
