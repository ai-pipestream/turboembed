package ai.pipestream.turbo;

/** One chat message ({@code turbo_message}): a role such as {@code user} and its content. */
public record Message(String role, String content) {
    public static Message user(String content) {
        return new Message("user", content);
    }

    public static Message system(String content) {
        return new Message("system", content);
    }

    public static Message assistant(String content) {
        return new Message("assistant", content);
    }
}
