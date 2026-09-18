package net.faulj.benchmark.roofline;

/**
 * Describes the confidence level of the roofline model based on which hardware
 * probes succeeded at runtime.
 *
 * <p>The tier determines how much of the roofline ceiling is grounded in
 * measurement versus assumption.  A lower-numbered tier means more of the
 * model is backed by actual hardware observation.</p>
 *
 * <ul>
 *   <li><b>FULLY_MEASURED</b> — all probes succeeded: STREAM bandwidths,
 *       sustained frequency, SIMD width, FMA, and issue width.  Produces
 *       true portable PES.</li>
 *   <li><b>PARTIALLY_MEASURED</b> — some probes failed.  Conservative
 *       defaults fill the gaps (e.g. issue_width=1).  PES is usable but
 *       marked ESTIMATED for the assumed parameters.</li>
 *   <li><b>MINIMAL</b> — only core count and clock are known.  Scalar-only
 *       compute roof.  Answers: "how efficiently does this kernel use what
 *       we can prove exists?"</li>
 *   <li><b>SAFE_MODE</b> — no reliable hardware info.  Roofline ceilings
 *       are not computed; only raw measured metrics are reported.  PES is
 *       UNDEFINED.</li>
 * </ul>
 */
enum CapabilityTier {

    /** All hardware probes succeeded — full roofline model. */
    FULLY_MEASURED("Tier 0: all probes measured"),

    /** Some probes failed; conservative defaults used. */
    PARTIALLY_MEASURED("Tier 1: partially measured, conservative defaults"),

    /** Only core count and clock known; scalar-only roof. */
    MINIMAL("Tier 2: minimal detection, scalar-only roof"),

    /** No reliable hardware info; PES undefined. */
    SAFE_MODE("Tier 3: safe mode, PES undefined");

    final String description;

    CapabilityTier(String description) {
        this.description = description;
    }
}
