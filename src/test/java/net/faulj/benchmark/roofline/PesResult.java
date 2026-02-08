package net.faulj.benchmark.roofline;

final class PesResult {
    final String kernel;
    final int n;
    final double arithmeticIntensity;
    final String boundType;
    final String memoryLevel;
    final double computeUtilization;
    final double memoryUtilization;
    final double algorithmicEfficiency;
    final double portableEfficiencyScore;
    final double measuredGflops;
    final double roofGflops;
    final double selectedMemoryRoofGbps;
    final double elapsedSeconds;

    PesResult(String kernel,
              int n,
              double arithmeticIntensity,
              String boundType,
              String memoryLevel,
              double computeUtilization,
              double memoryUtilization,
              double algorithmicEfficiency,
              double portableEfficiencyScore,
              double measuredGflops,
              double roofGflops,
              double selectedMemoryRoofGbps,
              double elapsedSeconds) {
        this.kernel = kernel;
        this.n = n;
        this.arithmeticIntensity = arithmeticIntensity;
        this.boundType = boundType;
        this.memoryLevel = memoryLevel;
        this.computeUtilization = computeUtilization;
        this.memoryUtilization = memoryUtilization;
        this.algorithmicEfficiency = algorithmicEfficiency;
        this.portableEfficiencyScore = portableEfficiencyScore;
        this.measuredGflops = measuredGflops;
        this.roofGflops = roofGflops;
        this.selectedMemoryRoofGbps = selectedMemoryRoofGbps;
        this.elapsedSeconds = elapsedSeconds;
    }
}
