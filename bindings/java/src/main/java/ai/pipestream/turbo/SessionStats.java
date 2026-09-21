package ai.pipestream.turbo;

/**
 * Session counters ({@code turbo_session_stats}). {@code providerAllocs} is
 * null when the provider cannot observe its own allocations.
 */
public record SessionStats(long runs, long hostAllocs, long h2dBytes, long d2hBytes, long inputBytes, long outputBytes, Long providerAllocs) {}
