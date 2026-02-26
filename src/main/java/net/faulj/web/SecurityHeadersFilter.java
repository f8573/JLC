package net.faulj.web;

import java.io.IOException;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Add baseline security headers to all backend responses.
 */
@Component
public class SecurityHeadersFilter extends OncePerRequestFilter {
    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        response.setHeader("X-Content-Type-Options", "nosniff");
        response.setHeader("X-Frame-Options", "DENY");
        response.setHeader("Referrer-Policy", "no-referrer");
        response.setHeader("Permissions-Policy", "geolocation=(), microphone=(), camera=()");
        response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
        response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        response.setHeader("X-Permitted-Cross-Domain-Policies", "none");
        String uri = request.getRequestURI();
        if (uri != null && uri.startsWith("/api/")) {
            response.setHeader("Cache-Control", "no-store, max-age=0");
            response.setHeader("Pragma", "no-cache");
        }

        if (request.isSecure()) {
            response.setHeader("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
        }

        filterChain.doFilter(request, response);
    }
}
