package net.faulj.web;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "net.faulj")
/**
 * Spring Boot entry point for the JLAC web service.
 */
public class Application {
    /**
     * Launch the Spring application context.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
