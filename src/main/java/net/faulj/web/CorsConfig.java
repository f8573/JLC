package net.faulj.web;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
/**
 * Configure CORS for API endpoints used by the frontend.
 */
public class CorsConfig {

    /**
     * Provide a WebMvcConfigurer that allows local Vite dev origins.
     *
     * @return configured WebMvcConfigurer
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/api/**")
                        .allowedOriginPatterns(
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
                                "https://www.lambdacompute.org")
                        .allowedMethods("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
                        .allowedHeaders("*");
            }
        };
    }

}
