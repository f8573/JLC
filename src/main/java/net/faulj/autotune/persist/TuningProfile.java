package net.faulj.autotune.persist;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Serializable tuning profile persisted to disk.
 */
public final class TuningProfile {

    public final String fingerprint;
    public final String timestamp;
    public final boolean converged;
    public final String terminationReason;

    public final int mr;
    public final int nr;
    public final int kc;
    public final int kUnroll;
    public final int mc;
    public final int nc;

    public final double gflops;
    public final double cv;

    public final String schemaVersion;
    public final String pipelineVersion;

    @JsonCreator
    public TuningProfile(@JsonProperty("fingerprint") String fingerprint,
                         @JsonProperty("timestamp") String timestamp,
                         @JsonProperty("converged") boolean converged,
                         @JsonProperty("terminationReason") String terminationReason,
                         @JsonProperty("mr") int mr,
                         @JsonProperty("nr") int nr,
                         @JsonProperty("kc") int kc,
                         @JsonProperty("kUnroll") int kUnroll,
                         @JsonProperty("mc") int mc,
                         @JsonProperty("nc") int nc,
                         @JsonProperty("gflops") double gflops,
                         @JsonProperty("cv") double cv,
                         @JsonProperty("schemaVersion") String schemaVersion,
                         @JsonProperty("pipelineVersion") String pipelineVersion) {
        this.fingerprint = fingerprint;
        this.timestamp = timestamp;
        this.converged = converged;
        this.terminationReason = terminationReason;
        this.mr = mr;
        this.nr = nr;
        this.kc = kc;
        this.kUnroll = kUnroll;
        this.mc = mc;
        this.nc = nc;
        this.gflops = gflops;
        this.cv = cv;
        this.schemaVersion = schemaVersion;
        this.pipelineVersion = pipelineVersion;
    }
}
