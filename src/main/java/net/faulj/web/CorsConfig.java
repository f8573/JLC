package net.faulj.web;

import java.util.Arrays;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Configure CORS for API endpoints used by the frontend.
 * SECURITY PATCH: Strict CORS configuration using explicit origins from properties.
 */
@Configuration
public class CorsConfig {

    // Inject allowed origins from properties, defaulting to production URL
    @Value("${app.cors.allowed-origins:https://lambdacompute.org,https://www.lambdacompute.org}")
    private String[] allowedOrigins;

    /**
     * Provide a WebMvcConfigurer with strict CORS settings.
     *
     * @return configured WebMvcConfigurer
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                String[] sanitizedOrigins = Arrays.stream(allowedOrigins == null ? new String[0] : allowedOrigins)
                        .map(origin -> origin == null ? "" : origin.trim())
                        .filter(origin -> !origin.isBlank() && !"*".equals(origin))
                        .toArray(String[]::new);
                // SECURITY PATCH: Strict CORS definition
                registry.addMapping("/api/**")
                        .allowedOrigins(sanitizedOrigins)
                        // Restrict methods to only those actually used by the API
                        .allowedMethods("GET", "POST", "OPTIONS")
                        // Restrict headers to standard safe sets rather than wildcard "*"
                        .allowedHeaders("Content-Type", "Authorization", "Accept", "Origin")
                        .allowCredentials(false)
                        .maxAge(3600); // Cache pre-flight requests for 1 hour
            }
        };
    }

}
